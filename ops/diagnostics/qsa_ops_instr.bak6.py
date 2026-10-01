# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA sparse attention and cache updates."""

from __future__ import annotations

from functools import lru_cache

import torch

try:
    from vllm.compilation import cg_instr as _cg_instr
except Exception:
    _cg_instr = None

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton


@lru_cache(maxsize=1)
def _is_sm120() -> bool:
    """True on sm_120 (RTX PRO 6000 Blackwell): selects the sm_120 tuning table."""
    return current_platform.get_device_capability() == (12, 0)


@lru_cache(maxsize=1)
def _is_sm90() -> bool:
    """True on sm_90 (H100/H200/H20): selects the sm_90 tuning table."""
    return current_platform.get_device_capability() == (9, 0)


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    softmax_scale,
    output_scale,
    output_gate_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    # The packed selection buffer carries one TRAILING COUNT COLUMN per row
    # (column TOPK of a TOPK+1-wide buffer): the row's valid-entry count,
    # written by the expand kernel. It is never a token index — the tile loop
    # and the index load below only ever cover columns [0, TOPK).
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    # softmax_scale is the host-side attention scale (1/sqrt(head_dim), with the
    # fp8 K dequant scale already pre-multiplied in); convert to log2 units once
    # here for the exp2-based online softmax.
    score_scale = softmax_scale * 1.4426950408889634

    tile_end = tl.minimum(NUM_TILES, tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N))

    for tile in range(split_id, tile_end, NUM_SPLITS):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if IS_FP8:
            # e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype (fp8 QK
            # measured slower here and less accurate).
            keys = keys.to(query.dtype)
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16; for fp8
        # caches the K dequant scale is already folded into softmax_scale on the
        # host.
        scores *= score_scale
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if IS_FP8:
            # Dequant V to fp16 (not bf16) for the PV dot: P <= 1 (online
            # softmax) so fp16 has the range, its wider mantissa is more
            # accurate, and the fp8->fp16 upcast with an fp16 PV dot is faster.
            values = values.to(tl.float16)
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    # Fold the fp8 V dequant scale (output_scale, 1.0 for bf16) into a per-row
    # reciprocal normalizer, so the output is a per-row multiply rather than a
    # HEAD_DIM-wide scale. The split-K LSE below keeps the unscaled normalizer,
    # so the merge stays correct.
    inv_normalizer = output_scale / tl.maximum(normalizer, 1.0e-20)
    normalized_output = tl.where(
        has_values[:, None],
        accumulator * inv_normalizer[:, None],
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        # Preserve the unfused path's BF16 attention-output rounding before
        # applying the gate in FP32.
        normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + (first_head + head_offsets[:, None]) * stride_output_gate_head
            + dim_offsets[None, :],
            mask=output_mask,
            other=0.0,
        ).to(tl.float32)
        normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_row = (split_id * num_rows + row).to(tl.int64)
        tl.store(
            partial_output_ptr
            + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_output = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    # Preserve the unfused path's BF16 attention-output rounding before
    # applying the gate in FP32.
    merged = merged.to(output_ptr.dtype.element_ty)
    output_gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


def _select_sm120_config(
    base_programs: int, use_prefill_config: bool, is_fp8: bool
) -> tuple[int, int, int]:
    """(block_n, target_splits, num_warps) retuned on sm_120 (RTX PRO 6000
    Blackwell), split by cache dtype: bf16 and fp8 favour different configs on
    sm_120, most visibly on the large-prefill region. Each entry is the fastest
    config that stays correct on its own path; the main win is more warps.
    """
    if is_fp8:
        if base_programs > 2048:
            return (32, 2, 1) if use_prefill_config else (64, 1, 2)
        if base_programs <= 24:
            return 64, 64, 8
        if base_programs <= 32:
            return 128, 8, 4
        if base_programs <= 64:
            return 64, 8, 4
        if base_programs <= 128:
            return 32, 4, 4
        if base_programs <= 256:
            return 128, 4, 4
        if base_programs <= 512:
            return 64, 4, 4
        return 32, 2, 1
    # bf16 K/V cache.
    if base_programs > 2048:
        return 64, 2, 2
    if base_programs <= 24:
        return 64, 64, 8
    if base_programs <= 64:
        return 64, 8, 8
    if base_programs <= 128:
        return 32, 8, 8
    if base_programs <= 256:
        return 32, 8, 1
    if base_programs <= 512:
        return 64, 4, 2
    return 32, 4, 1


def _select_sm90_config(
    base_programs: int, use_prefill_config: bool, is_fp8: bool
) -> tuple[int, int, int]:
    """(block_n, target_splits, num_warps) tuned on sm_90 (H20)."""
    if base_programs > 2048:
        return (32, 1, 1) if use_prefill_config else (64, 1, 2)
    if is_fp8:
        if base_programs <= 24:
            return 64, 64, 2
        if base_programs <= 32:
            return 32, 16, 1
        if base_programs <= 64:
            return 32, 8, 1
        if base_programs <= 128:
            return 32, 4, 1
        if base_programs <= 256:
            return 32, 8, 1
        return 32, 4, 1
    if base_programs <= 32:
        return 64, 64, 2
    if base_programs <= 64:
        return 32, 16, 1
    if base_programs <= 256:
        return 32, 8, 1
    return 32, 4, 1


def _select_config(
    num_rows: int,
    num_kv_heads: int,
    use_prefill_config: bool,
    num_columns: int,
    is_fp8: bool = False,
) -> tuple[int, int, int, int]:
    """Select (block_n, num_warps, num_tiles, num_splits) for the kernel.

    Keyed on base_programs = num_rows * num_kv_heads. The bp > 2048 region splits
    on use_prefill_config (capture-stable: at FULL-graph capture max_query_len is
    the uniform decode/verify length). This default table was tuned on GB300;
    sm_120 (RTX PRO 6000 Blackwell) dispatches to _select_sm120_config instead,
    and sm_90 (Hopper) to _select_sm90_config.
    """
    base_programs = num_rows * num_kv_heads
    if _is_sm120():
        BLOCK_N, target_splits, num_warps = _select_sm120_config(
            base_programs, use_prefill_config, is_fp8
        )
    elif _is_sm90():
        BLOCK_N, target_splits, num_warps = _select_sm90_config(
            base_programs, use_prefill_config, is_fp8
        )
    elif base_programs > 2048:
        BLOCK_N, target_splits, num_warps = (
            (32, 1, 1) if use_prefill_config else (64, 1, 2)
        )
    elif base_programs <= 24:
        BLOCK_N, target_splits, num_warps = 32, 64, 4
    elif base_programs <= 32:
        BLOCK_N, target_splits, num_warps = 32, 16, 1
    elif base_programs <= 64:
        BLOCK_N, target_splits, num_warps = 32, 8, 1
    elif base_programs <= 128:
        BLOCK_N, target_splits, num_warps = 32, 4, 1
    elif base_programs <= 256:
        BLOCK_N, target_splits, num_warps = 32, 8, 1
    elif base_programs <= 512:
        BLOCK_N, target_splits, num_warps = 64, 4, 2
    else:
        BLOCK_N, target_splits, num_warps = 64, 1, 2
    num_tiles = triton.cdiv(num_columns, BLOCK_N)
    # Never more splits than tiles, never empty.
    num_splits = min(target_splits, num_tiles)
    return BLOCK_N, num_warps, num_tiles, num_splits


# ---------------------------------------------------------------------------
# Diagnostic: frozen-input "placement x launch-mode" crossover (Astra #4).
# This module is only mounted in diagnostic containers, so the probe defaults
# ON; set QSA_XPROBE=0 to silence it.  It never reads model state back before
# timing, and it only writes the model's output buffer in arms whose K/V
# contents are identical to the production ones.
# ---------------------------------------------------------------------------
import os as _os
import time as _time

from vllm.logger import init_logger as _init_logger

_xlog = _init_logger("qsa_xprobe")
_XP_ON = _os.environ.get("QSA_XPROBE", "1") not in ("0", "false", "False")
_XP_CALL = [0]          # probes actually fired
_XP_QUAL = [0]          # probe-eligible calls seen (replay, prefill-chunk shaped)
_TRACE_KERNEL = [None]
_TR_SLOT = 16
_LAT_CTA = 296
_LAT_ITERS = 1000


_TH_N = 1 << 17          # samples per arm per round
_TH_ROUNDS = 2
_TH_SM = 74
# (workers per SM, iterations)  -> workers/SM == BLK because grid == 74 CTAs
# (KG=in-flight requests per warp, BLK, iters) with grid=74 CTAs
_TH_ARMS = ((1, 32, 21), (4, 32, 21), (32, 32, 21), (32, 512, 1),
            (32, 1024, 1))

# ---------------------------------------------------------------------------
# Astra #7 battery: virtual-address x physical-backing crossover via CUDA VMM,
# an aliased large-VA-span arm, and an empty_cache recovery test.  Armed by
# /tmp/qsa_vmm_arm.  Allocations are made *inside the engine's own CUDA context*
# on purpose: the question is whether fresh mappings in the poisoned context are
# already slow, and whether the slowness follows the VA or the physical block.
# ---------------------------------------------------------------------------
import ctypes as _ctypes
import gc as _gc
import torch.utils.dlpack as _dlpack

_VMM_ARM_DEFAULT = "/tmp/qsa_vmm_arm"


class _CULoc(_ctypes.Structure):
    _fields_ = [("type", _ctypes.c_int), ("id", _ctypes.c_int)]


class _CUAllocFlags(_ctypes.Structure):
    _fields_ = [("compressionType", _ctypes.c_ubyte),
                ("gpuDirectRDMACapable", _ctypes.c_ubyte),
                ("usage", _ctypes.c_ushort),
                ("reserved", _ctypes.c_ubyte * 4)]


class _CUAllocProp(_ctypes.Structure):
    _fields_ = [("type", _ctypes.c_int), ("requestedHandleTypes", _ctypes.c_int),
                ("location", _CULoc), ("win32HandleMetaData", _ctypes.c_void_p),
                ("allocFlags", _CUAllocFlags)]


class _CUAccessDesc(_ctypes.Structure):
    _fields_ = [("location", _CULoc), ("flags", _ctypes.c_int)]


class _DLDevice(_ctypes.Structure):
    _fields_ = [("device_type", _ctypes.c_int), ("device_id", _ctypes.c_int)]


class _DLDataType(_ctypes.Structure):
    _fields_ = [("code", _ctypes.c_uint8), ("bits", _ctypes.c_uint8),
                ("lanes", _ctypes.c_uint16)]


class _DLTensor(_ctypes.Structure):
    _fields_ = [("data", _ctypes.c_void_p), ("device", _DLDevice),
                ("ndim", _ctypes.c_int), ("dtype", _DLDataType),
                ("shape", _ctypes.POINTER(_ctypes.c_int64)),
                ("strides", _ctypes.POINTER(_ctypes.c_int64)),
                ("byte_offset", _ctypes.c_uint64)]


class _DLManagedTensor(_ctypes.Structure):
    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", _ctypes.c_void_p),
                ("deleter", _ctypes.c_void_p)]


_ctypes.pythonapi.PyCapsule_New.restype = _ctypes.py_object


class _Vmm:
    """Minimal CUDA driver-API VMM: phys blocks, VA ranges, remap, DLPack wrap."""

    def __init__(self):
        self._cu = _ctypes.CDLL("libcuda.so.1")
        self._keep = []
        prop = _CUAllocProp()
        prop.type = 1                    # CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = 0    # CU_MEM_HANDLE_TYPE_NONE
        prop.location.type = 1           # CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = 0
        prop.allocFlags.compressionType = 0
        self.prop = prop
        g = _ctypes.c_size_t()
        self._ck(self._cu.cuMemGetAllocationGranularity(
            _ctypes.byref(g), _ctypes.byref(prop), 0), "gran")
        self.gran = g.value

    @staticmethod
    def _ck(r, what):
        if r != 0:
            raise RuntimeError(f"vmm {what} -> CUDA error {r}")

    def _al(self, n):
        return (n + self.gran - 1) // self.gran * self.gran

    def phys(self, nbytes):
        n = self._al(nbytes)
        h = _ctypes.c_ulonglong()
        self._ck(self._cu.cuMemCreate(_ctypes.byref(h), _ctypes.c_size_t(n),
                                      _ctypes.byref(self.prop),
                                      _ctypes.c_ulonglong(0)), "cuMemCreate")
        return int(h.value), n

    def reserve(self, nbytes):
        n = self._al(nbytes)
        va = _ctypes.c_ulonglong()
        self._ck(self._cu.cuMemAddressReserve(
            _ctypes.byref(va), _ctypes.c_size_t(n),
            _ctypes.c_size_t(self.gran), _ctypes.c_ulonglong(0),
            _ctypes.c_ulonglong(0)), "cuMemAddressReserve")
        return int(va.value), n

    def map(self, va, handle, nbytes):
        self._ck(self._cu.cuMemMap(_ctypes.c_ulonglong(va),
                                   _ctypes.c_size_t(nbytes), _ctypes.c_size_t(0),
                                   _ctypes.c_ulonglong(handle),
                                   _ctypes.c_ulonglong(0)), "cuMemMap")
        d = _CUAccessDesc()
        d.location.type = 1
        d.location.id = 0
        d.flags = 3                      # CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._ck(self._cu.cuMemSetAccess(_ctypes.c_ulonglong(va),
                                         _ctypes.c_size_t(nbytes),
                                         _ctypes.byref(d),
                                         _ctypes.c_size_t(1)), "cuMemSetAccess")

    def unmap(self, va, nbytes):
        self._ck(self._cu.cuMemUnmap(_ctypes.c_ulonglong(va),
                                     _ctypes.c_size_t(nbytes)), "cuMemUnmap")

    def release(self, handle):
        self._ck(self._cu.cuMemRelease(_ctypes.c_ulonglong(handle)), "cuMemRelease")

    def free_va(self, va, nbytes):
        self._ck(self._cu.cuMemAddressFree(_ctypes.c_ulonglong(va),
                                           _ctypes.c_size_t(nbytes)),
                 "cuMemAddressFree")

    def tensor(self, va, nwords):
        t = _DLManagedTensor()
        shp = (_ctypes.c_int64 * 1)(nwords)
        t.dl_tensor.data = _ctypes.c_void_p(va)
        t.dl_tensor.device = _DLDevice(2, 0)          # kDLCUDA
        t.dl_tensor.ndim = 1
        t.dl_tensor.dtype = _DLDataType(0, 64, 1)     # kDLInt / 64 bit
        t.dl_tensor.shape = shp
        t.dl_tensor.strides = None
        t.dl_tensor.byte_offset = 0
        cap = _ctypes.pythonapi.PyCapsule_New(_ctypes.byref(t), b"dltensor", None)
        self._keep.extend([t, shp])
        if len(self._keep) > 4096:
            del self._keep[:2048]
        return _dlpack.from_dlpack(cap)


@triton.jit
def _xp_vmm_scatter(src, dst, idx, ROWS_PC: tl.constexpr, VEC: tl.constexpr,
                    STRIDE_W: tl.constexpr):
    """Identical logical work in every arm: ROWS_PC rows of VEC*8 bytes per CTA."""
    pid = tl.program_id(0)
    lanes = tl.arange(0, VEC)
    for r in range(ROWS_PC):
        row = tl.load(idx + pid * ROWS_PC + r)
        tl.store(dst + (pid * ROWS_PC + r) * VEC + lanes,
                 tl.load(src + row * STRIDE_W + lanes))


@triton.jit
def _vmm_fill_kernel(ptr, n, val):
    """Pre-touch a mapping without involving torch on foreign memory."""
    pid = tl.program_id(0)
    off = pid * 1024 + tl.arange(0, 1024)
    tl.store(ptr + off, tl.full((1024,), 7, tl.int64), mask=off < n)


def _mem_battery(timed, dev):
    """Astra #7, torch-only (safe inside a live graph replay).

    Answers three of the four rows of Astra's decision table without any
    driver-level VMM call:
      V(8)  eight independently allocated buffers -> distinct VA placements:
            all equally slow => context-wide; some fast => VA-region attached.
      VB    one ~1 GiB buffer, same gather spread over the whole span:
            slow while small spans are fine => VA span / page-coverage effect.
      EC    release cached blocks, allocate fresh, re-measure (recovery test).
    """
    ROWS, ROWW = 4096, 2048          # 4096 rows of 2 KiB (engine FSC/STR shape)
    out = {}

    def rep(tag, t, idx, reps=4):
        ts, _ = timed(lambda: fo.copy_(t[idx]), reps=reps)
        out[tag] = (round(ts[0], 3), round(ts[-1], 3))
        _xlog.warning("QXPROBE mem cell %s = %s ms  va=0x%x", tag, out[tag],
                      t.data_ptr())

    fo = torch.empty(ROWS, ROWW, dtype=torch.int64, device=dev)
    fi = torch.randint(0, ROWS, (ROWS,), dtype=torch.int64, device=dev)
    _xlog.warning("QXPROBE mem part0 begin rows=%d roww=%d free=%dMiB",
                  ROWS, ROWW, torch.cuda.mem_get_info()[0] >> 20)
    bufs = []
    try:
        for i in range(8):
            b = torch.randint(-(2 ** 62), 2 ** 62, (ROWS, ROWW),
                              dtype=torch.int64, device=dev)
            bufs.append(b)
        torch.cuda.synchronize()
        _xlog.warning("QXPROBE mem part1 8 buffers allocated (%d MiB each, VA %.2f GiB "
                      ".. %.2f GiB apart)", ROWS * ROWW * 8 >> 20,
                      (bufs[-1].data_ptr() - bufs[0].data_ptr()) / 2 ** 30,
                      (bufs[1].data_ptr() - bufs[0].data_ptr()) / 2 ** 30)
        for i, b in enumerate(bufs):
            rep(f"V{i}", b, fi)
            ts, _ = timed(lambda: fo.view(-1).copy_(b.view(-1)), reps=2)
            out[f"D{i}"] = (round(ts[0], 3), round(ts[-1], 3))
        _xlog.warning("QXPROBE mem part2 dense controls = %s",
                      " ".join(f"D{i}={out[f'D{i}']}" for i in range(8)))
        del bufs
        bufs = None
        # ---- VB: one big buffer, same gather spread over the whole span
        nrow_big = (1 << 30) // (ROWW * 8)          # 1 GiB worth of rows
        big = torch.randint(-(2 ** 62), 2 ** 62, (nrow_big, ROWW),
                            dtype=torch.int64, device=dev)
        jdx = torch.randint(0, nrow_big, (ROWS,), dtype=torch.int64, device=dev)
        rep("VB", big, jdx)
        ts, _ = timed(lambda: fo.view(-1).copy_(big.view(-1)[: ROWS * ROWW]), reps=2)
        out["DB"] = (round(ts[0], 3), round(ts[-1], 3))
        _xlog.warning("QXPROBE mem part3 big span %dMiB: gather VB=%s dense DB=%s",
                      nrow_big * ROWW * 8 >> 20, out["VB"], out["DB"])
        del big, jdx
        # ---- EC: release cached blocks, allocate fresh, re-measure
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _gc.collect()
        torch.cuda.synchronize()
        fresh = torch.randint(-(2 ** 62), 2 ** 62, (ROWS, ROWW),
                              dtype=torch.int64, device=dev)
        rep("EC", fresh, fi)
        rep("EC2", fresh, fi)
        del fresh
    except Exception as e:  # noqa: BLE001
        _xlog.warning("QXPROBE mem partial failure: %r", e)
    _xlog.warning("QXPROBE mem DONE free=%dMiB | %s",
                  torch.cuda.mem_get_info()[0] >> 20,
                  " ".join(f"{k}={a}/{b}" for k, (a, b) in out.items()))


def _alloc_battery(timed, dev):
    """Allocator-vs-execution discriminator (the temporary-tensor confound).

    Every slow arm so far was `dst.copy_(src[idx])`, which makes torch allocate a temporary
    result before launching; every fast arm (slices/views, Triton kernels, GEMM with out=) did
    not.  `timed()` brackets with CUDA events on an otherwise-empty stream, so a *host-side*
    allocator stall between ev0 and the kernel launch lands inside the measured interval.
    T1 (temp) vs T2 (index_select with out=, no temp) separates CPU/allocator from GPU execution.
    """
    ROWS, ROWW = 4096, 2048
    st = torch.cuda.memory_stats()
    snap = torch.cuda.memory_snapshot()
    nblk = sum(len(seg.get("blocks", ())) for seg in snap)
    nseg = len(snap)
    lrg = sum(len(seg.get("blocks", ())) for seg in snap if seg.get("pool", "") is None)
    _xlog.warning(
        "QXPROBE alloc part0 free=%dMiB alloc=%.2fGiB resv=%.2fGiB retries=%d "
        "segments=%d blocks=%d large_pool_blocks=%d max_split=%.1fMiB",
        torch.cuda.mem_get_info()[0] >> 20,
        st.get("allocated_bytes.all.current", 0) / 2 ** 30,
        st.get("reserved_bytes.all.current", 0) / 2 ** 30,
        st.get("num_alloc_retries", 0), nseg, nblk, lrg,
        st.get("max_split_size", 0) / 2 ** 20)
    src = torch.randint(-(2 ** 62), 2 ** 62, (ROWS * 2, ROWW), dtype=torch.int64, device=dev)
    idx = torch.randint(0, ROWS * 2, (ROWS,), dtype=torch.int64, device=dev)
    dst = torch.empty(ROWS, ROWW, dtype=torch.int64, device=dev)

    def hostalloc(reps=5):
        hs = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            x = torch.empty(ROWS, ROWW, dtype=torch.int64, device=dev)
            hs.append((_time.perf_counter() - t0) * 1e3)
            del x
        return [round(v, 3) for v in hs]

    t1, h1 = timed(lambda: dst.copy_(src[idx]), reps=4)             # temp allocation
    t2, h2 = timed(lambda: torch.index_select(src, 0, idx, out=dst), reps=4)  # no temp
    _xlog.warning("QXPROBE alloc T1 advanced-index (temp)  event=%s ms host=%s ms",
                  [round(v, 3) for v in t1], [round(v, 3) for v in h1])
    _xlog.warning("QXPROBE alloc T2 index_select(out=) (no temp) event=%s ms host=%s ms",
                  [round(v, 3) for v in t2], [round(v, 3) for v in h2])
    _xlog.warning("QXPROBE alloc T3 bare empty(32MiB) host_ms=%s", hostalloc())
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t4, h4 = timed(lambda: dst.copy_(src[idx]), reps=4)
    _xlog.warning("QXPROBE alloc T4 AFTER empty_cache: T1=%s ms host=%s ms T3=%s",
                  [round(v, 3) for v in t4], [round(v, 3) for v in h4], hostalloc())
    t5, h5 = timed(lambda: dst.copy_(src[:ROWS]), reps=4)
    _xlog.warning("QXPROBE alloc T5 dense view (no alloc) event=%s ms host=%s ms",
                  [round(v, 3) for v in t5], [round(v, 3) for v in h5])
    t6, h6 = timed(lambda: src[idx], reps=4)                        # pure allocation, no copy
    _xlog.warning("QXPROBE alloc T6 src[idx] alone (alloc only) event=%s ms host=%s ms",
                  [round(v, 3) for v in t6], [round(v, 3) for v in h6])
    _xlog.warning("QXPROBE alloc DONE free=%dMiB", torch.cuda.mem_get_info()[0] >> 20)


def _vmm_battery(timed, dev):
    """Astra #7: VA-region x physical-block crossover + aliased VA span + recovery.

    Design note: the four crossover cells are built with **four distinct VA
    ranges established once** (V0a<-P0, V0b<-P1, V1a<-P0, V1b<-P1) and then only
    *measured* -- no mid-flight remapping inside a live engine, which crashed the
    engine on the first attempt.  Every phase logs its numbers immediately, so a
    later failure still keeps the earlier data.
    """
    BLK = 64 << 20                  # 64 MiB per physical block / VA range
    GAP = 32 << 20
    ROWS, VEC, SW = 16384, 32, 512  # 16384 rows of 256 B at 4 KiB stride
    out = {}
    v = _Vmm()
    _xlog.warning("QXPROBE vmm part0 gran=%dMiB ctypes-vmm-ready", v.gran >> 20)

    def run(tag, t, idx, rpc, reps=3):
        ts, _ = timed(lambda: _xp_vmm_scatter[(ROWS // rpc,)](
            t, dst, idx, rpc, VEC, SW, num_warps=1), reps=reps)
        out[tag] = (round(ts[0], 3), round(ts[-1], 3))
        _xlog.warning("QXPROBE vmm cell %s = %s ms", tag, out[tag])

    p0, n = v.phys(BLK)
    p1, _ = v.phys(BLK)
    # V0 region (VA bits low-ish): two adjacent 64 MiB ranges sharing high bits
    r0, _ = v.reserve(BLK * 2 + GAP)
    v0a, v0b = r0, r0 + BLK + GAP
    # spacer forces the V1 region to differ in the high VA bits
    sp, sn = v.reserve(16 << 30)
    v.free_va(sp, sn)
    r1, _ = v.reserve(BLK * 2 + GAP)
    v1a, v1b = r1, r1 + BLK + GAP
    cells = (("P0V0", v0a, p0), ("P1V0", v0b, p1),
             ("P0V1", v1a, p0), ("P1V1", v1b, p1))
    WORDS = BLK // 8
    blk_rows = BLK // (SW * 8)
    dst = torch.empty(ROWS * VEC, dtype=torch.int64, device=dev)
    idxs = {rpc: torch.randint(0, WORDS // SW - 1, (ROWS,), dtype=torch.int64,
                               device=dev) for rpc in (1, 16)}
    # ---- establish all four mappings once, then pre-touch each
    tens = {}
    try:
        for tag, va, ph in cells:
            v.map(va, ph, n)
            t = v.tensor(va, WORDS)
            _vmm_fill_kernel[(WORDS // 1024 + 1,)](t, WORDS, 7, num_warps=4)
            tens[tag] = t
        torch.cuda.synchronize()
        _xlog.warning("QXPROBE vmm part1 mapped+touched 4 cells va=%s/%s/%s/%s "
                   "apart=%sGiB", *(f"0x{x:x}" for x in (v0a, v0b, v1a, v1b)),
                   round(abs(v1a - v0a) / 2 ** 30, 1))
        # measure twice, second pass in reverse order (time-resolved)
        for p_i, order in enumerate((cells, tuple(reversed(cells)))):
            for tag, _va, _ph in order:
                for rpc in (1, 16):
                    run(f"{tag}_{rpc}_p{p_i}", tens[tag], idxs[rpc], rpc)
        # ---- aliased VA span: ONE physical block, irregular, over 4 GiB
        SPAN, step = 4 << 30, BLK * 3 // 2       # 96 MiB spacing -> 32 MiB gaps
        va, nn = v.reserve(SPAN)
        nmap, off = 0, 0
        while off + BLK <= SPAN:
            v.map(va + off, p0, n)
            nmap += 1
            off += step
        torch.cuda.synchronize()
        al = v.tensor(va, nn // 8)
        step_rows = step // (SW * 8)
        ak = {rpc: torch.randint(0, nmap, (ROWS,), dtype=torch.int64, device=dev)
              for rpc in (1, 16)}
        ar = {rpc: torch.randint(0, blk_rows - 1, (ROWS,), dtype=torch.int64,
                                 device=dev) for rpc in (1, 16)}
        for rpc in (1, 16):
            run(f"AL_{rpc}", al, ak[rpc] * step_rows + ar[rpc], rpc)
        _xlog.warning("QXPROBE vmm part2 alias=%dx%sMiB spread over %sGiB (irregular)",
                   nmap, BLK >> 20, SPAN >> 30)
        # ---- engine-pool-like torch gather (self-contained control)
        fr = torch.randint(-(2 ** 62), 2 ** 62, (4096, 2048), dtype=torch.int64,
                           device=dev)
        fo = torch.empty(4096, 2048, dtype=torch.int64, device=dev)
        fi = torch.randint(0, 4095, (4096,), dtype=torch.int64, device=dev)
        ts, _ = timed(lambda: fo.copy_(fr[fi]), reps=2)
        out["GG1"] = (round(ts[0], 3), round(ts[-1], 3))
        _xlog.warning("QXPROBE vmm part3 engine-pool gather GG1=%s ms", out["GG1"])
        # ---- recovery test: release cached blocks, then re-measure BOTH pools
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _gc.collect()
        torch.cuda.synchronize()
        ts, _ = timed(lambda: fo.copy_(fr[fi]), reps=2)
        out["GG2"] = (round(ts[0], 3), round(ts[-1], 3))
        for rpc in (1, 16):
            run(f"AL2_{rpc}", al, ak[rpc] * step_rows + ar[rpc], rpc)
        _xlog.warning("QXPROBE vmm part4 after-empty_cache GG2=%s AL2=%s",
                   out["GG2"], (out.get("AL2_1"), out.get("AL2_16")))
        del al, fr, fo, fi
    except Exception as e:  # noqa: BLE001
        _xlog.warning("QXPROBE vmm partial failure: %r", e)
    finally:
        try:
            del tens
        except Exception:
            pass
        torch.cuda.synchronize()
        try:
            v.release(p0)
            v.release(p1)
        except Exception:
            pass
    _xlog.warning("QXPROBE vmm DONE free=%sMiB | %s",
               round(torch.cuda.mem_get_info()[0] / 2 ** 20),
               " ".join(f"{k}={a}/{b}" for k, (a, b) in out.items()))


@triton.jit
def _xp_thermo_kernel(base_ptr, perm_ptr, out_ptr, n_off, iters, slot_base,
                      KG: tl.constexpr, BLK: tl.constexpr):
    """Per-access COMPLETION latency thermometer.

    The second ``%globaltimer`` read is guarded by a predicate derived from the
    loaded value, so the warp cannot issue that S2R until the load's data has
    arrived (an ungated S2R times instruction *issue*, not completion).
    Each lane group of ``KG`` lanes shares one address inside its warp, so the
    number of in-flight requests per warp equals ``KG``; ``out`` stores
    (dt_ns, word_offset) pairs so every sample carries a page tag.
    """
    pid = tl.program_id(0)
    i = tl.arange(0, BLK)
    lid = (i // 32) * 64 + (i % 32) % KG
    for j in range(iters):
        off = tl.load(perm_ptr + (j * 7919 + pid * 4096 + lid) % n_off)
        t0, v, t1 = tl.inline_asm_elementwise(
            "{\n"
            ".reg .pred %pp;\n"
            ".reg .b64 %vv;\n"
            "mov.u64 $0, %clock64;\n"
            "ld.global.u64 %vv, [$3];\n"
            "setp.ne.u64 %pp, %vv, 0;\n"
            "@%pp mov.u64 $2, %clock64;\n"
            "@!%pp mov.u64 $2, %clock64;\n"
            "mov.u64 $1, %vv;\n"
            "}",
            "=l,=l,=l,l", [base_ptr + off],
            dtype=(tl.int64, tl.int64, tl.int64), is_pure=False, pack=1
        )
        s = slot_base + pid * (BLK * iters) + j * BLK + i
        tl.store(out_ptr + 2 * s, t1 - t0)
        tl.store(out_ptr + 2 * s + 1, off)
        t0 += v * 0


@triton.jit
def _xp_lat_kernel(buf_ptr, out_ptr, mask, iters, BLK: tl.constexpr):
    """Dependent-load chain: loaded memory latency (ns per access)."""
    pid = tl.program_id(0)
    i = tl.arange(0, BLK)
    a = (pid * BLK + i).to(tl.int64) % mask
    t0 = tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l,r', [i],
                                   dtype=tl.int64, is_pure=False, pack=1)
    for _j in range(iters):
        a = tl.load(buf_ptr + a)
    t1 = tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l,r', [i],
                                   dtype=tl.int64, is_pure=False, pack=1)
    o = pid * BLK + i
    tl.store(out_ptr + 2 * o, t1 - t0)
    tl.store(out_ptr + 2 * o + 1, a)
_XP_LAST = [0.0]
_XP_MAX = int(_os.environ.get("QSA_XPROBE_MAX", "100000"))
_XP_MIN_NQ = int(_os.environ.get("QSA_XPROBE_MIN_NQ", "1024"))
_XP_ARM_PATH = _os.environ.get("QSA_XPROBE_ARM", "/tmp/qsa_probe_arm")
_XP_TRACE_ARM = _os.environ.get("QSA_TRACE_ARM", "/tmp/qsa_trace_arm")
_XP_ARM_C = [0.0, 0]
_XP_TRACE_ON = [0]
_XP_GATE_T = [0.0]

try:
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture as _BCC,
    )
except Exception:  # pragma: no cover
    _BCC = None


def _xp_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _xp_active() -> bool:
    try:
        return bool(_BCC is not None and _BCC.current() is not None)
    except Exception:
        return False


def _xp_stride() -> int:
    """Arming file controls probing without a restart.

    Missing file -> 0 (disabled).  Content N -> probe every Nth eligible call.
    """
    now = _time.time()
    if now - _XP_ARM_C[0] < 0.25:
        return _XP_ARM_C[1]
    try:
        with open(_XP_ARM_PATH) as f:
            raw = f.read().strip()
        v = int(raw) if raw else 1
    except Exception:
        v = 0
    _XP_ARM_C[0] = now
    _XP_ARM_C[1] = v
    return v


def _xp_trace_armed() -> bool:
    try:
        return _os.path.exists(_XP_TRACE_ARM)
    except Exception:
        return False


def _xp_ready(q, use_prefill_config, num_splits) -> bool:
    if not _XP_ON or _XP_CALL[0] > _XP_MAX:
        return False
    if num_splits != 1 or q.shape[0] < _XP_MIN_NQ:
        return False
    if _xp_capturing() or _xp_active():
        return False
    _XP_QUAL[0] += 1
    stride = _xp_stride()
    if stride <= 0 or (_XP_QUAL[0] % stride):
        return False
    return True


def _xp_gate(q, k_cache, block_table, logical_indices, use_prefill_config, num_splits) -> None:
    """Throttled dump of the live call, armed or not."""
    now = _time.time()
    if now - _XP_GATE_T[0] < 2.0:
        return
    _XP_GATE_T[0] = now
    quiet = _xp_capturing() or _xp_active()
    extra = ""
    if not quiet:
        try:
            cnt = logical_indices[:, -1]
            li0 = [int(x) for x in logical_indices[0, :4].tolist()]
            bt0 = [int(x) for x in block_table[0, :8].tolist()]
            extra = (
                f" cnt_max={int(cnt.max())} cnt_sum={int(cnt.sum())} "
                f"li0={li0} bt0={bt0} bt_min={int(block_table.min())} "
                f"bt_max={int(block_table.max())} "
                f"bt_uniq={int(torch.unique(block_table).numel())}"
            )
        except Exception as e:
            extra = f" dump_err={e!r}"
    print(
        f"QXGATE t={now:.1f} nq={int(q.shape[0])} nhead={int(q.shape[1])} "
        f"kvh={int(k_cache.shape[2])} prefill={int(bool(use_prefill_config))} "
        f"splits={int(num_splits)} cap={int(_xp_capturing())} "
        f"active={int(_xp_active())} qual={_XP_QUAL[0]} arm={_XP_ARM_C[1]}"
        f" fired={_XP_CALL[0]}{extra}",
        flush=True,
    )


def _qx_trace_kernel():
    """Build (once) a copy of the splitk kernel with per-CTA phase timestamps.

    The copy is generated by transforming this module's own source, so the
    compiled body is identical to production except for inline-asm reads
    (``%globaltimer``, ``%smid``) and the trace stores.  Per-CTA slots
    (stride ``_TR_SLOT``): 0=t0, 1=t1, 2=smid, 3=prologue, 4=index/table,
    5=K/V loads, 6=math+loop, 7=num_tiles, 8=epilogue+output store.
    """
    if _TRACE_KERNEL[0] is not None:
        return _TRACE_KERNEL[0]
    key = "def _qsa_sparse_paged_gqa_splitk_kernel("
    with open(globals()["__file__"]) as _f:
        text = _f.read()
    i = text.index(key)
    j = text.rindex("@triton.jit", 0, i)
    k = text.index("@triton.jit", i + len(key))
    src = text[j:k].replace(key, "def _qsa_trace_kernel(", 1)
    src = src.replace(
        "\n    num_requests,\n", "\n    num_requests,\n    trace_ptr,\n", 1
    )
    lines = src.split("\n")

    def find(pred, start=0):
        for idx in range(start, len(lines)):
            if pred(lines[idx]):
                return idx
        raise RuntimeError("trace transform: anchor not found")

    i_sig = find(lambda L: L.rstrip() == ") -> None:")
    i_loop = find(lambda L: L.strip().startswith("for tile in range(split_id"), i_sig)
    i_keys = find(lambda L: L.strip().startswith("keys = tl.load("), i_loop)
    i_dot = find(lambda L: L.strip().startswith("scores = tl.dot(query, keys)"), i_keys)
    i_end = len(lines) - 1
    while i_end > 0 and not lines[i_end].strip():
        i_end -= 1
    S = _TR_SLOT

    def stamp(acc, ind=4):
        p = " " * ind
        out = [p + "_tr_tt = _tr_time()"]
        if acc:
            out.append(f"{p}_tr_{acc} += _tr_tt - _tr_prev")
        out.append(p + "_tr_prev = _tr_tt")
        return out

    pro = [
        "    _tr_pid = (tl.program_id(0) + tl.program_id(1) * num_rows",
        f"               + tl.program_id(2) * num_rows * {S // 2})",
        "    _tr_one = tl.arange(0, 1)",
        "    _tr_t0 = _tr_time()",
        f"    tl.store(trace_ptr + _tr_pid * {S} + _tr_one, _tr_t0)",
        "    _tr_sm = tl.inline_asm_elementwise('mov.u32 $0, %smid;',",
        "        '=r,r', [_tr_one], dtype=tl.int32, is_pure=False, pack=1)",
        f"    tl.store(trace_ptr + _tr_pid * {S} + 2 + _tr_one, _tr_sm.to(tl.int64))",
        "    _tr_prev = _tr_t0",
        "    _tr_pr = tl.zeros((1,), dtype=tl.int64)",
        "    _tr_ix = tl.zeros((1,), dtype=tl.int64)",
        "    _tr_kv = tl.zeros((1,), dtype=tl.int64)",
        "    _tr_mt = tl.zeros((1,), dtype=tl.int64)",
        "    _tr_nt = tl.zeros((1,), dtype=tl.int64)",
    ]
    tail = [
        "    _tr_tt = _tr_time()",
        "    _tr_mt += _tr_tt - _tr_prev",
        "    _tr_prev = _tr_tt",
        f"    tl.store(trace_ptr + _tr_pid * {S} + 3 + _tr_one, _tr_pr)",
        f"    tl.store(trace_ptr + _tr_pid * {S} + 4 + _tr_one, _tr_ix)",
        f"    tl.store(trace_ptr + _tr_pid * {S} + 5 + _tr_one, _tr_kv)",
        f"    tl.store(trace_ptr + _tr_pid * {S} + 6 + _tr_one, _tr_mt)",
        f"    tl.store(trace_ptr + _tr_pid * {S} + 7 + _tr_one, _tr_nt)",
        "    _tr_t1 = _tr_time()",
        f"    tl.store(trace_ptr + _tr_pid * {S} + 1 + _tr_one, _tr_t1)",
        f"    tl.store(trace_ptr + _tr_pid * {S} + 8 + _tr_one, _tr_t1 - _tr_tt)",
    ]
    new = (
        lines[: i_sig + 1]
        + pro
        + lines[i_sig + 1 : i_loop]
        + stamp("pr")
        + lines[i_loop : i_loop + 1]
        + stamp("mt", 8)
        + ["        _tr_nt += 1"]
        + lines[i_loop + 1 : i_keys]
        + stamp("ix", 8)
        + lines[i_keys:i_dot]
        + stamp("kv", 8)
        + lines[i_dot : i_end + 1]
        + tail
        + lines[i_end + 1 :]
    )
    import importlib.util as _ilu
    import pathlib as _pl

    _path = "/tmp/_qsa_trace_kernel_gen.py"
    _pl.Path(_path).write_text(
        "# auto-generated diagnostic copy of the production QSA kernel\n"
        "import triton\nimport triton.language as tl\n\n\n"
        "@triton.jit\n"
        "def _tr_time():\n"
        "    _o = tl.arange(0, 1)\n"
        "    return tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l,r',\n"
        "                                     [_o], dtype=tl.int64, is_pure=False,\n"
        "                                     pack=1)\n\n\n" + "\n".join(new)
    )
    ns = dict(globals())
    _spec = _ilu.spec_from_file_location("_qsa_trace_kernel_gen", _path)
    _mod = _ilu.module_from_spec(_spec)
    for _k, _v in ns.items():
        if not _k.startswith("__"):
            _mod.__dict__.setdefault(_k, _v)
    _spec.loader.exec_module(_mod)
    _TRACE_KERNEL[0] = _mod._qsa_trace_kernel
    print(
        f"QXTRACE kernel built: {_path} args={len(_TRACE_KERNEL[0].arg_names)} "
        f"trace_ptr_in_sig={'trace_ptr' in _TRACE_KERNEL[0].arg_names}",
        flush=True,
    )
    return _TRACE_KERNEL[0]


def _xp_trace_analyze(tr, t_evt_ms, cal, nq) -> None:
    """Astra's envelope analysis + per-phase attribution from the trace."""
    d = tr.detach().to("cpu").view(-1, _TR_SLOT)
    iv = []
    for r in d.tolist():
        if r[1] > 0:
            iv.append([int(x) for x in r])
    n = len(iv)
    if n == 0:
        print(f"QXTRACE cal={cal} empty (no CTA end stamps)", flush=True)
        return
    tf = min(r[0] for r in iv)
    tlast = max(r[1] for r in iv)
    t_env = tlast - tf
    u = 0
    itv = sorted(iv)
    cs, ce = itv[0][0], itv[0][1]
    for r in itv[1:]:
        if r[0] > ce:
            u += ce - cs
            cs, ce = r[0], r[1]
        elif r[1] > ce:
            ce = r[1]
    u += ce - cs

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    def p95(xs):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * 0.95))]

    ds = [r[1] - r[0] for r in iv]
    pr = [r[3] for r in iv]
    ix = [r[4] for r in iv]
    kv = [r[5] for r in iv]
    mt = [r[6] for r in iv]
    nt = [r[7] for r in iv]
    ep = [r[8] for r in iv]
    kv_s = sum(kv) / 1e9
    kv_conc = (sum(kv) / 1e6) / (t_env / 1e6)
    kv_bw = (sum(nt) * 16384) / kv_s / 1e9 if kv_s > 0 else 0.0
    nb = 40
    prof = []
    for i in range(nb):
        t = tf + (i + 0.5) * t_env / nb
        prof.append(sum(1 for r in iv if r[0] <= t <= r[1]))
    print(
        f"QXTRACE cal={cal} n={n} nq={nq} t_evt={t_evt_ms:.2f}ms "
        f"t_env={t_env / 1e6:.3f}ms U={u / 1e6:.3f}ms "
        f"head_tail={t_evt_ms - t_env / 1e6:.3f}ms env_minus_U={(t_env - u) / 1e6:.3f}ms "
        f"D_us[med/p95/max]={med(ds) / 1e3:.1f}/{p95(ds) / 1e3:.1f}/{max(ds) / 1e3:.1f} "
        f"ntiles={med(nt)} "
        f"PH_us[pr/ix/kv/math/ep]={med(pr) / 1e3:.1f}/{med(ix) / 1e3:.1f}/"
        f"{med(kv) / 1e3:.1f}/{med(mt) / 1e3:.1f}/{med(ep) / 1e3:.1f} "
        f"kv_conc={kv_conc:.1f} kv_bw={kv_bw:.0f}GB/s "
        f"sumkv={kv_s * 1e3:.1f}ms summt={sum(mt) / 1e6:.1f}ms "
        f"sumix={sum(ix) / 1e6:.1f}ms "
        f"nsm={len({r[2] for r in iv})} "
        f"prof={'/'.join(str(x) for x in prof)}",
        flush=True,
    )


def _xp_thermo_analyze(dt_off, meta, ms) -> None:
    """Astra's thermometer report: latency CDF + per-page tail attribution."""
    import array as _arr  # noqa: F401

    def pct(xs, q):
        if not xs:
            return 0.0
        k = min(len(xs) - 1, int(len(xs) * q))
        return xs[k]

    print(f"QXT cal launcher_ms={ms:.3f}", flush=True)
    for r, bname, kg, blk, slot, cnt in meta:
        sl = dt_off[2 * slot: 2 * (slot + cnt)]
        dt = sl[0::2].tolist()
        off = sl[1::2].tolist()
        n = len(dt)
        if n == 0:
            continue
        srt = sorted(dt)
        # per-page (64 KiB) tail attribution
        pages = {}
        for d, o in zip(dt, off):
            pages.setdefault((o * 8) >> 16, []).append(d)
        qs, tail_tot, tail_top, big = [], 0, 0, 0
        thr = 1000
        per_page_slow = sorted(
            ((sum(1 for d in v if d > thr) / len(v), len(v)) for v in pages.values()),
            key=lambda x: -x[0],
        )
        tail_tot = sum(1 for d in dt if d > thr)
        npg = len(per_page_slow)
        k1 = max(1, npg // 100)
        tail_top = sum(int(q * c) for q, c in per_page_slow[:k1])
        big = sum(1 for d in dt if d > 10000)
        ghz = 1.485
        cy = lambda c: c / (ghz * 1000.0)  # cycles -> us
        print(
            f"QXT r={r} buf={bname} KG={kg} BLK={blk} n={n} pages={npg} "
            f"us[p50/p90/p99/p999/p9999/max]={cy(srt[n // 2]):.3f}/"
            f"{cy(pct(srt, 0.90)):.3f}/{cy(pct(srt, 0.99)):.3f}/"
            f"{cy(pct(srt, 0.999)):.3f}/{cy(pct(srt, 0.9999)):.3f}/"
            f"{cy(srt[-1]):.3f} cyc_max={srt[-1]} "
            f"n>10cyc={sum(1 for d in dt if d > 10)} "
            f"n>1us={sum(1 for d in dt if cy(d) > 1)} n>10us={big} "
            f"pg_q1us[p50/p95/max]={pct([q for q, _ in per_page_slow], 0.5):.4f}/"
            f"{pct([q for q, _ in per_page_slow], 0.95):.4f}/"
            f"{per_page_slow[0][0]:.4f} top1pct_share="
            f"{(tail_top / tail_tot if tail_tot else 0.0):.3f}",
            flush=True,
        )


def _xp_thermo_run(k_cache, k16, dev):
    """One shot of the thermometer over both allocations, arms interleaved."""
    # incompressible data: zero pages are served by L2 compression and would
    # measure nothing (validated)
    fresh = torch.randint(-(2 ** 62), 2 ** 62, (8 << 20,), dtype=torch.int64,
                          device=dev)
    # The KV pool is a NON-CONTIGUOUS slice of a larger (possibly sparsely
    # mapped) allocation, so addresses stay inside each block's own contiguous
    # run: block b spans ``b*stride0`` plus its 1584*256-word page body.  This
    # is both always-mapped and the same block/page locality the op reads.
    k16 = k_cache.view(torch.int64)
    s0 = int(k16.stride(0))
    body = int(k16.shape[1]) * int(k16.stride(1))
    _bb = torch.randint(0, int(k16.shape[0]), (_TH_N,), dtype=torch.int64,
                        device=dev)
    _oo = torch.randint(0, body, (_TH_N,), dtype=torch.int64, device=dev)
    pkv = _bb * s0 + _oo
    # n_off must be the PERMUTATION length (it indexes perm), never the buffer
    bufs = [("fresh", fresh, _TH_N), ("kvpool", k16, _TH_N)]
    perms = [torch.randint(0, int(fresh.numel()) - 2, (_TH_N,),
                           dtype=torch.int64, device=dev), pkv]
    slot_total = sum(blk * it for _, blk, it in _TH_ARMS) * _TH_SM * _TH_ROUNDS * 2
    out = torch.zeros(2 * slot_total, dtype=torch.int64, device=dev)
    meta = []
    slot = 0
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for r in range(_TH_ROUNDS):
        for (bname, buf, nb), perm in zip(bufs, perms):
            for kg, blk, it in _TH_ARMS:
                _xp_thermo_kernel[(_TH_SM,)](buf, perm, out, nb, it, slot,
                                             KG=kg, BLK=blk,
                                             num_warps=max(1, blk // 32))
                meta.append((r, bname, kg, blk, slot, blk * it * _TH_SM))
                slot += blk * it * _TH_SM
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1)
    return out[: 2 * slot].to("cpu"), meta, ms


def _xp_probe(*, launch, q, k_cache, v_cache, block_table, logical_indices, out, cal):
    """Time the same compiled kernel under 5 (placement, launch-mode) cells."""
    _XP_LAST[0] = _time.time()
    dev = q.device
    nblk = int(k_cache.shape[0])
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream())
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    out2 = torch.empty_like(out)

    def timed(fn, reps=3):
        ts, hs = [], []
        for _ in range(reps):
            s.synchronize()
            t0 = _time.perf_counter()
            ev0.record(s)
            with torch.cuda.stream(s):
                fn()
            ev1.record(s)
            hs.append((_time.perf_counter() - t0) * 1e3)
            s.synchronize()
            ts.append(ev0.elapsed_time(ev1))
        return ts, hs

    def graph_of(fn):
        g = torch.cuda.CUDAGraph()
        s.synchronize()
        with torch.cuda.stream(s):
            with torch.cuda.graph(g):
                fn()
        return g

    res = {}
    # 0a) allocator-vs-execution discriminator FIRST when armed
    if _os.path.exists(_os.environ.get("QSA_ALLOC_ARM", "/tmp/qsa_alloc_arm")):
        try:
            _alloc_battery(timed, dev)
        except Exception as e:
            _xlog.warning("QXPROBE alloc battery failed: %r", e)
    # 0b) Astra #7 VMM battery FIRST when armed (least-drained poisoned moment)
    if _os.path.exists(_os.environ.get("QSA_MEM_ARM", "/tmp/qsa_mem_arm")):
        try:
            _mem_battery(timed, dev)
        except Exception as e:
            _xlog.warning("QXPROBE vmm battery failed: %r", e)
    # 0) per-CTA %globaltimer/%smid trace: ONE execution, first thing measured
    if _xp_trace_armed():
        try:
            tr = torch.zeros(262144, dtype=torch.int64, device=dev)
            res["TR"], res["TRh"] = timed(
                lambda: launch(k_cache, v_cache, block_table, out2, None, tr),
                reps=1,
            )
            _xp_trace_analyze(tr, res["TR"][0], cal, int(q.shape[0]))
        except Exception as e:
            _xlog.warning("QXPROBE trace arm failed: %r", e)
    # 0a) per-access latency thermometer FIRST: it perturbs the state, so it
    #     must sample the least-drained moment (round index = time axis)
    if _os.path.exists(_os.environ.get("QSA_THERMAL_ARM", "/tmp/qsa_thermal_arm")):
        try:
            _k16 = k_cache.view(torch.int64)
            _t, _m, _ms = _xp_thermo_run(k_cache, _k16, dev)
            _xp_thermo_analyze(_t, _m, _ms)
            del _k16
        except Exception as e:
            _xlog.warning("QXPROBE thermal arm failed: %r", e)
    # 0b) device-wide control: fixed 4096^3 bf16 matmul (~137 GFLOP)
    try:
        gA = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
        gB = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
        gC = torch.empty(4096, 4096, dtype=torch.bfloat16, device=dev)
        res["G4"], res["G4h"] = timed(lambda: torch.mm(gA, gB, out=gC))
    except Exception as e:
        _xlog.warning("QXPROBE gemm arm failed: %r", e)
    # 0c) same-instant memory-service controls: streaming BW + loaded latency
    try:
        if getattr(_xp_probe, "_ctl", None) is None:
            _xp_probe._ctl = (
                torch.empty(32 << 20, dtype=torch.uint8, device=dev),
                torch.empty(32 << 20, dtype=torch.uint8, device=dev),
                torch.randint(0, (4 << 20) - 1, (4 << 20,), dtype=torch.int64,
                              device=dev),
                torch.empty(_LAT_CTA * 64, dtype=torch.int64, device=dev),
            )
        _bwi, _bwo, _latb, _lato = _xp_probe._ctl
        res["BW"], res["BWh"] = timed(lambda: _bwo.copy_(_bwi), reps=2)
        res["LAT"], res["LATh"] = timed(
            lambda: _xp_lat_kernel[(_LAT_CTA,)](
                _latb, _lato, 4 << 20, _LAT_ITERS, BLK=32, num_warps=1
            ),
            reps=1,
        )
        # real KV-pool controls: does THIS allocation still stream / scatter?
        if getattr(_xp_probe, "_pool", None) is None:
            _nb = int(k_cache.shape[0])
            _rh = int(k_cache.shape[1])
            _xp_probe._pool = (
                torch.zeros(16, *k_cache.shape[1:], dtype=k_cache.dtype, device=dev),
                torch.zeros(16384, *k_cache.shape[2:], dtype=k_cache.dtype, device=dev),
                torch.randint(0, _nb, (16384,), device=dev),
                torch.randint(0, _rh, (16384,), device=dev),
            )
        _pi, _ps, _pb, _pr = _xp_probe._pool
        res["KRD"], _ = timed(lambda: _pi.copy_(k_cache[:16]), reps=2)
        res["KSC"], _ = timed(lambda: _ps.copy_(k_cache[_pb, _pr]), reps=2)
        _pb.copy_(torch.randint(0, int(k_cache.shape[0]), (16384,), device=dev))
        _pr.copy_(torch.randint(0, int(k_cache.shape[1]), (16384,), device=dev))
        # scatter controls on a FRESH buffer, same chunk count/size/footprint
        if getattr(_xp_probe, "_fr", None) is None:
            _xp_probe._fr = (
                torch.zeros(32768, 2048, dtype=torch.uint8, device=dev),
                torch.zeros(16384, 2048, dtype=torch.uint8, device=dev),
                torch.randint(0, 32767, (16384,), device=dev),
            )
        _frb, _fro, _fri = _xp_probe._fr
        _sa = torch.arange(16384, device=dev) * 2
        _fro.copy_(_frb[_fri])
        torch.cuda.synchronize()
        res["FSC"], _ = timed(lambda: _fro.copy_(_frb[_fri]), reps=2)
        res["STR"], _ = timed(lambda: _fro.copy_(_frb[_sa]), reps=2)
        res["LAT2"], res["LAT2h"] = timed(
            lambda: _xp_lat_kernel[(_LAT_CTA,)](
                _latb, _lato, 4 << 20, 100, BLK=32, num_warps=1
            ),
            reps=2,
        )
    except Exception as e:
        _xlog.warning("QXPROBE mem-control arms failed: %r", e)
    # 1) original addresses + production selection, isolated eager (first: coldest)
    res["OE"], res["OEh"] = timed(lambda: launch(k_cache, v_cache, block_table, out2))
    # 2) same K/V storage, shifted block table (no copy; writes to scratch out)
    try:
        bt_sh = torch.where(
            block_table >= 0, (block_table + 7) % nblk, block_table
        ).to(torch.int32).contiguous()
        res["SH"], res["SHh"] = timed(
            lambda: launch(k_cache, v_cache, bt_sh, out2)
        )
    except Exception as e:
        _xlog.warning("QXPROBE shifted-table arm failed: %r", e)
    # 3) relocated copy of exactly the referenced blocks + private table
    meta = {}
    try:
        ids_dev = torch.unique(block_table.clamp(min=0))
        ids = [int(x) for x in ids_dev.to("cpu").tolist()]  # host sync, after OE/SH
        nids = len(ids)
        per_block = 1
        for d in k_cache.shape[1:]:
            per_block *= int(d)
        scr_bytes = nids * per_block * k_cache.element_size() * 2
        if scr_bytes > 512 * 2**20:
            raise RuntimeError(f"relocation scratch too large ({scr_bytes/2**20:.0f} MiB)")
        sk = torch.empty(
            (nids, *k_cache.shape[1:]), dtype=k_cache.dtype, device=dev
        )
        sv = torch.empty_like(sk)
        sk.copy_(k_cache.index_select(0, ids_dev))
        sv.copy_(v_cache.index_select(0, ids_dev))
        lut = torch.full((nblk,), -1, dtype=torch.int32)
        lut[ids_dev.to("cpu")] = torch.arange(nids, dtype=torch.int32)
        bt_cpu = block_table.to("cpu")
        bt_rel = torch.where(
            bt_cpu >= 0, lut[bt_cpu.clamp(min=0)], bt_cpu.to(torch.int32)
        ).to(dev).to(torch.int32).contiguous()
        res["RE"], res["REh"] = timed(lambda: launch(sk, sv, bt_rel, out2))
        _bw = res.get("BW")
        _lt = res.get("LAT")
        _lt2 = (res.get("LAT2") or [None])[0]
        _krd = res.get("KRD")
        _ksc = res.get("KSC")
        _kbytes = (16 * k_cache[0].numel() * k_cache.element_size()) / 1e6
        _kbytes_sc = (16384 * int(torch.tensor(k_cache.shape[2:]).prod())
                      * k_cache.element_size()) / 1e6
        meta = {
            "n_ids": nids,
            "scr_mib": round(2 * sk.numel() * sk.element_size() / 2**20, 2),
            "bw_gbs": round(67.1 / _bw[len(_bw) // 2], 1) if _bw else -1,
            "ns_load": round(_lt[0] * 1e6 / _LAT_ITERS, 1) if _lt else -1,
            "ns_load100": round(_lt2 * 1e6 / 100.0, 1) if _lt2 else -1,
            "krd_gbs": round(2 * _kbytes / _krd[1], 1) if _krd else -1,
            "ksc_gbs": round(_kbytes_sc / _ksc[1], 1) if _ksc else -1,
            "fsc_gbs": round(_kbytes_sc / res["FSC"][1], 1) if res.get("FSC") else -1,
            "str_gbs": round(_kbytes_sc / res["STR"][1], 1) if res.get("STR") else -1,
        }
    except Exception as e:
        _xlog.warning("QXPROBE relocation arm failed: %r", e)
        meta["rel"] = "failed"


    # 4) graph-replay arms last (capture failures must not lose the eager cells)
    try:
        g = graph_of(lambda: launch(k_cache, v_cache, block_table, out2))
        res["OG"], res["OGh"] = timed(g.replay)
    except Exception as e:
        _xlog.warning("QXPROBE graph arm failed: %r", e)
    try:
        if "sk" in locals():
            g2 = graph_of(lambda: launch(sk, sv, bt_rel, out2))
            res["RG"], res["RGh"] = timed(g2.replay)
    except Exception as e:
        _xlog.warning("QXPROBE relocated graph failed: %r", e)
    # 5) repeat the first cell last: measures how much probing itself clears
    try:
        res["OE2"], res["OE2h"] = timed(
            lambda: launch(k_cache, v_cache, block_table, out2)
        )
    except Exception as e:
        _xlog.warning("QXPROBE repeat arm failed: %r", e)
    # 6) zero-filled K/V on the relocated table: values vs execution context
    try:
        if "sk" in locals():
            skz = torch.zeros_like(sk)
            svz = torch.zeros_like(sv)
            res["RK"], res["RKh"] = timed(lambda: launch(skz, svz, bt_rel, out2))
    except Exception as e:
        _xlog.warning("QXPROBE zero-value arm failed: %r", e)

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, list):
            return "/".join(f"{x:.2f}" for x in v)
        return f"{v:.2f}"

    try:
        bt_min = int(block_table.min())
        bt_max = int(block_table.max())
        bt_uniq = int(torch.unique(block_table).numel())
        bt0 = [int(x) for x in block_table[0, :8].tolist()]
        cnt = logical_indices[:, -1]
        cnt_max = int(cnt.max())
        cnt_sum = int(cnt.sum())
        ti_max = int(logical_indices[:, :-1].max())
        li0 = [int(x) for x in logical_indices[0, :6].tolist()]
    except Exception:
        bt_min = bt_max = bt_uniq = cnt_max = cnt_sum = ti_max = -1
        bt0 = li0 = []
    _msg = (
        "QXPROBE cal=%d t=%.1f nq=%d nhead=%d kvh=%d sel_w=%d nblk=%d btw=%d "
        "kstr=%s/%s/%s bt_min=%d bt_max=%d bt_uniq=%d bt0=%s cnt_max=%d "
        "cnt_sum=%d ti_max=%d li0=%s | G4=%s BW=%s LAT=%s KRD=%s KSC=%s FSC=%s STR=%s | OE=%s RE=%s SH=%s OG=%s RG=%s "
        "OE2=%s RK=%s TR=%s (ms) | host_ms G4=%s BW=%s LAT=%s OE=%s RE=%s SH=%s OG=%s RG=%s "
        "OE2=%s RK=%s TRh=%s %s"
        % (
            cal, _time.time(), q.shape[0], q.shape[1], k_cache.shape[2],
            logical_indices.shape[1] - 1, k_cache.shape[0], block_table.shape[1],
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            bt_min, bt_max, bt_uniq, bt0, cnt_max, cnt_sum, ti_max, li0,
            fmt(res.get("G4")), fmt(res.get("BW")), fmt(res.get("LAT")),
            fmt(res.get("KRD")), fmt(res.get("KSC")), fmt(res.get("FSC")),
            fmt(res.get("STR")),
            fmt(res.get("OE")), fmt(res.get("RE")),
            fmt(res.get("SH")), fmt(res.get("OG")), fmt(res.get("RG")),
            fmt(res.get("OE2")), fmt(res.get("RK")), fmt(res.get("TR")),
            fmt(res.get("G4h")), fmt(res.get("BWh")), fmt(res.get("LATh")),
            fmt(res.get("OEh")),
            fmt(res.get("REh")), fmt(res.get("SHh")), fmt(res.get("OGh")),
            fmt(res.get("RGh")), fmt(res.get("OE2h")), fmt(res.get("RKh")),
            fmt(res.get("TRh")),
            " ".join(f"{k}={v}" for k, v in meta.items()),
        )
    )
    _xlog.info("%s", _msg)
    print(_msg, flush=True)
    return res


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    *,
    output_gate: torch.Tensor,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 or FP8-e4m3 K/V caches.

    With fp8 caches, k_scale/v_scale are the layer's per-tensor dequant scales
    as host floats (e.g. layer._k_scale_float/_v_scale_float): k_scale is
    pre-multiplied into the softmax scale and v_scale becomes the kernel's
    output scale, so the kernel needs no device scale buffers.

    logical_indices is the PACKED selection buffer: [rows, selection_width + 1]
    with the trailing column holding each row's valid-entry count (written by
    the expand kernel; never a token index). The kernel reads it as the
    tile-loop bound. use_prefill_config only steers the top of the config table; see
    _select_config.
    """
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] < 2:
        raise ValueError(
            "QSA packed indices need selection columns plus the count column"
        )
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype
    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if is_fp8:
        assert k_scale is not None and v_scale is not None
        # Host pre-multiply: fold the K dequant scale into the attention scale
        # and pass V's dequant scale as the kernel's output scale.
        softmax_scale = (head_dim**-0.5) * float(k_scale)
        output_scale = float(v_scale)
    else:
        assert k_cache.dtype == torch.bfloat16
        softmax_scale = head_dim**-0.5
        output_scale = 1.0
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1

    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    assert output_gate.is_contiguous()
    output_gate_view = output_gate.view_as(q)
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    selection_width = logical_indices.shape[1] - 1  # trailing column is the count
    block_n, partial_warps, num_tiles, num_splits = _select_config(
        q.shape[0], k_cache.shape[2], use_prefill_config, selection_width, is_fp8
    )
    if _cg_instr is not None and _cg_instr.sync_ok():
        try:
            _cnt = logical_indices[:, -1]
            _cg_instr.qa(
                "ATTN",
                nq=int(q.shape[0]),
                kvh=int(k_cache.shape[2]),
                sel_w=int(selection_width),
                block_n=int(block_n),
                n_tiles=int(num_tiles),
                n_splits=int(num_splits),
                warps=int(partial_warps),
                prefill=int(bool(use_prefill_config)),
                cnt_max=int(_cnt.max().item()) if _cnt.numel() else -1,
                cnt_sum=int(_cnt.sum().item()) if _cnt.numel() else -1,
                ti_max=int(logical_indices[:, :-1].max().item()) if logical_indices.numel() else -1,
            )
        except Exception:
            pass

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)

    def _xlaunch(kc, vc, bt, o=None, li=None, trp=None):
        oo = out if o is None else o
        lli = logical_indices if li is None else li
        _pos = (
            q,
            kc,
            vc,
            lli,
            bt,
            token_to_req,
            oo if num_splits == 1 else partial_output,
            oo if num_splits == 1 else partial_lse,
            oo,
            softmax_scale,
            output_scale,
            output_gate_view,
            q.stride(0),
            q.stride(1),
            kc.stride(0),
            kc.stride(1),
            kc.stride(2),
            vc.stride(0),
            vc.stride(1),
            vc.stride(2),
            lli.stride(0),
            bt.stride(0),
            oo.stride(0),
            oo.stride(1),
            output_gate_view.stride(0),
            output_gate_view.stride(1),
            q.shape[0],
            kc.shape[0],
            bt.shape[0],
        )
        _kw = dict(
            TOPK=selection_width,
            PAGE_SIZE=kc.shape[1],
            PAGE_TABLE_WIDTH=bt.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=q.shape[2],
            NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            IS_FP8=is_fp8,
        )
        if trp is None:
            _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
                *_pos, **_kw, num_warps=partial_warps, num_stages=2
            )
        else:
            _qx_trace_kernel()[partial_grid](
                *_pos, trp, **_kw, num_warps=partial_warps, num_stages=2
            )

    _qsa_stage = _cg_instr.stage("ATTENTION") if _cg_instr else None
    if _qsa_stage:
        _qsa_stage.__enter__()
    try:
        _xlaunch(k_cache, v_cache, block_table)
    finally:
        if _qsa_stage:
            _qsa_stage.__exit__(None, None, None)

    # ---- diagnostic crossover (frozen inputs; never writes model state) ----
    if _xp_ready(q, use_prefill_config, num_splits):
        _XP_CALL[0] += 1
        try:
            _xp_probe(
                launch=_xlaunch,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                block_table=block_table,
                logical_indices=logical_indices,
                out=out,
                cal=_XP_CALL[0],
            )
        except Exception as _e:
            _xlog.warning("QXPROBE error: %r", _e)
    else:
        try:
            _xp_gate(
                q, k_cache, block_table, logical_indices, use_prefill_config,
                num_splits,
            )
        except Exception as _e:
            _xlog.warning("QXGATE error: %r", _e)

    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        output_gate_view,
        out.stride(0),
        out.stride(1),
        output_gate_view.stride(0),
        output_gate_view.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


def warmup_qsa_sparse_paged_attention(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_query_heads: int,
    selection_width: int,
) -> tuple[tuple[int, int, int], ...]:
    """Compile every production-reachable split-K/merge specialization."""
    head_dim = kv_cache.shape[-1] // 2
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    # An fp8 cache is allocated as uint8 and viewed as e4m3 at attention time.
    is_fp8 = kv_cache.dtype == torch.uint8
    cache_dtype = torch.float8_e4m3fn if is_fp8 else key_cache.dtype
    num_kv_heads = key_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    block_m = triton.next_power_of_2(group_size)

    # Every config the dispatch can pick for this group size.
    profiles = {
        _select_config(
            num_rows, num_kv_heads, use_prefill_config, selection_width, is_fp8
        )
        for num_rows in range(1, 8193)
        for use_prefill_config in (False, True)
    }

    # Scalars constant per deployment get their real values (their divisibility
    # specialization is wanted); the batch-varying ones are do_not_specialize'd
    # on the kernels, so any value here compiles the only variant.
    num_rows = 16
    num_requests = 16
    q_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    k_cache_ptr = TritonWarmupTensor(
        cache_dtype,
        shape=tuple(key_cache.shape),
        strides=tuple(key_cache.stride()),
    )
    v_cache_ptr = TritonWarmupTensor(
        cache_dtype,
        shape=tuple(value_cache.shape),
        strides=tuple(value_cache.stride()),
    )
    # +1: the packed buffer's trailing count column.
    indices_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1))
    block_table_ptr = TritonWarmupTensor(
        block_table.dtype,
        shape=tuple(block_table.shape),
        strides=tuple(block_table.stride()),
    )
    token_to_req_ptr = TritonWarmupTensor(torch.int32)
    output_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    # The output gate is mandatory at runtime; warm the gated specialization.
    output_gate_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    head_stride = head_dim
    row_stride = num_query_heads * head_dim
    num_cache_blocks = triton_scalar_specialization_rep(kv_cache.shape[0])

    warmed = []
    for block_n, warps, num_tiles, num_splits in sorted(profiles):
        if num_splits == 1:
            partial_output_ptr = output_ptr
            partial_lse_ptr = output_ptr
        else:
            partial_output_ptr = TritonWarmupTensor(
                torch.float32,
                shape=(num_splits, num_rows, num_query_heads, head_dim),
            )
            partial_lse_ptr = TritonWarmupTensor(
                torch.float32, shape=(num_splits, num_rows, num_query_heads)
            )
        _qsa_sparse_paged_gqa_splitk_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            indices_ptr,
            block_table_ptr,
            token_to_req_ptr,
            partial_output_ptr,
            partial_lse_ptr,
            output_ptr,
            1.0,
            1.0,
            output_gate_ptr,
            row_stride,
            head_stride,
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            selection_width + 1,
            block_table.stride(0),
            row_stride,
            head_stride,
            row_stride,
            head_stride,
            num_rows,
            num_cache_blocks,
            num_requests,
            TOPK=selection_width,
            PAGE_SIZE=key_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            IS_FP8=is_fp8,
            num_warps=warps,
            num_stages=2,
            grid=(num_rows, num_kv_heads, num_splits),
        )
        if num_splits > 1:
            _qsa_merge_splitk_kernel.warmup(
                partial_output_ptr,
                partial_lse_ptr,
                output_ptr,
                output_gate_ptr,
                row_stride,
                head_stride,
                row_stride,
                head_stride,
                num_rows,
                HEAD_DIM=head_dim,
                NUM_QUERY_HEADS=num_query_heads,
                NUM_SPLITS=num_splits,
                BLOCK_SPLITS=triton.next_power_of_2(num_splits),
                num_warps=2,
                num_stages=1,
                grid=(num_rows, num_query_heads),
            )
        warmed.append((block_n, num_splits, warps))
    return tuple(warmed)


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""
    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,  # this step's raw key rows [rows, 1, head_size]
    raw_positions: torch.Tensor,  # this step's positions [rows, 1, 3] int64
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the compressor-state ring and raw token rows."""
    if not raw_keys.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA compression requires Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        # The ring is wider than one group so speculative rows cannot alias
        # onto the committed keys of the group still being collected.
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError(
            "QSA compressor-state cache does not match the compression layout"
        )
    if (
        compressor_state_block_table.ndim != 2
        or compressor_state_block_table.shape[1] < 1
    ):
        raise ValueError(
            "QSA compressor-state block table must contain one block per request"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (
        not all(compressor_state_cache.shape)
        or not all(compressor_state_block_table.shape)
    ):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "qsa_compress_groups_with_ratio",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
    "warmup_qsa_sparse_paged_attention",
]
