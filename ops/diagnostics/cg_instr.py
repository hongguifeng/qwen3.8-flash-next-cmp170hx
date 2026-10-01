"""CUPTI-free GPU-time attribution for vLLM's breakable CUDA graphs.

Why: this host (WSL2 + custom sm_80) has no working CUPTI
(`CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`), so torch.profiler / nsys / ncu are all
unusable.  CUDA events do work, so we bake `external=True` event records into the
captured graphs (their record nodes run on every replay) and harvest the elapsed
times one step later -- this gives per-module GPU time *inside* graph replay.

Also times each `BreakableCUDAGraphCapture.segments[i]` invocation (graph replays
and eager-break callables) with normal events + host wall clock.

Mounted at /opt/vllm/src/vllm/compilation/cg_instr.py and installed from the
instrumented ple_ssd.py when CG_INSTR=1.  Purely additive: every path is guarded,
no synchronisation, no `.item()` in the hot path, and if the pool runs dry it
degrades to no instrumentation instead of failing.
"""

from __future__ import annotations

import collections
import os
import threading
import time
import traceback

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

_ON = os.environ.get("CG_INSTR", "0").strip().lower() not in ("", "0", "false", "no")
_MAX_PAIRS = int(os.environ.get("CG_INSTR_MAX_PAIRS", "24576"))
_STAGE_POOL = int(os.environ.get("CG_INSTR_STAGE_POOL", "8192"))
_LOG_EVERY = float(os.environ.get("CG_INSTR_LOG_EVERY", "2.0"))
_VERBOSE = os.environ.get("CG_INSTR_VERBOSE", "1").strip().lower() not in ("", "0", "false", "no")

# module class -> short tag.  Deliberately no module whose forward spans an
# eager break (e.g. Qwen4ExpPLELayer: its start_prefetch/_finalize_prefetch are
# add_eager sites, so an event pair around it would straddle two graph segments).
_WL = {
    "Qwen4ExpDecoderLayer": "ALL",
    "QwenGatedDeltaNetAttention": "GDN",
    "Qwen3NextAttention": "ATN3",
    "Qwen4ExpQSAAttention": "ATNQ",
    "Qwen4ExpSparseMoeBlock": "MOE",
    "Qwen3NextMLP": "MLP",
    # QSA sparse indexer (top-k candidate selection) -- the prime suspect
    # for a per-step, token-count-independent fixed cost.
    "QSAIndexer": "IDX",
}
_LAYER_CLS = "Qwen4ExpDecoderLayer"

_LOCK = threading.Lock()
_ACC = collections.defaultdict(lambda: [0, 0.0, 0.0])      # label -> [n, sum_ms, max_ms]
_CLS = collections.defaultdict(lambda: [0, 0.0, 0.0])      # tag   -> [n, sum_ms, max_ms]
_LAST_LAYERS: dict[int, float] = {}                        # layer -> inclusive ms (last round)
_ROUNDS: list = []
_ROLL: list = []          # rolling window of finalized rounds (len<=12)
_STATS = collections.Counter()
_SEG_HOST: list[float] = []
_SEG_GPU: list[float] = []
_LAST_TOK = -1
_STAGE_HOST: dict = collections.defaultdict(float)   # stage label -> replay host ms

_modpool: list = []          # pre-created external event pairs
_stagepool: list = []        # dedicated pool for QSA stage pairs
_freepairs: list = []        # free normal event pairs (segment timing)
_ensured = False
_STAGE_SEEN = set()
_SEEN_CAPS: dict = {}        # id(capture) -> [desc, nlayers, nseg, replays]
_QA: dict = {}               # tag -> {n, key: [n, min, max, sum]}  (windowed)


def sync_ok() -> bool:
    """True when a device->host readback is safe (no CUDA stream capture active)."""
    return bool(_ON) and not _CAPT_ON[0]


def qa(tag: str, **kv) -> None:
    """Accumulate one launch/state record for `tag`; the logger prints them
    windowed as `QSALOG tag n=.. k=avg[min..max]`.  Used to fingerprint the
    *work* a kernel is asked to do (grid, splits, candidate counts, physical
    block layout) so clean and poisoned runs can be compared without CUPTI.
    """
    if not _ON:
        return
    try:
        d = _QA.setdefault(tag, {})
        for k, v in kv.items():
            try:
                v = float(v)
            except Exception:
                continue
            e = d.get(k)
            if e is None:
                d[k] = [1, v, v, v]
            else:
                e[0] += 1
                if v < e[1]:
                    e[1] = v
                if v > e[2]:
                    e[2] = v
                e[3] += v
    except Exception:
        pass


def stage(name: str):
    """Record one named QSA stage.

    Two distinct lifetimes:

    * capture-time, inside a *graph* segment -> the event records become graph
      nodes, so the captured graph re-records them on every replay and the pair
      yields a replay-time GPU duration.  These pairs stay owned by the capture
      object (`_cg_stagepairs`) and are never recycled.
    * replay-time, inside an *eager break* callable (the QSA path runs through
      `@eager_break_during_capture`) -> a fresh pair is taken per replay, appended
      to the round's `epairs` and returned to `_stagepool` at finalize.

    A dedicated pool is used so that stage events can never be starved by the
    module hooks' pool.  Purely additive: on an empty pool this degrades to a
    no-op instead of failing.
    """
    class _Stage:
        __slots__ = ("name", "pair", "cap", "label", "t0")
        def __init__(self):
            self.name = name
            self.pair = None
            self.cap = None
            self.label = None
            self.t0 = 0.0
        def __enter__(self):
            target = None
            cap = None
            seg = -1
            if _REPLAY_ON[0]:
                target = _REPLAY_ON[2]
                cap = _REPLAY_ON[1]
                seg = _REPLAY_ON[3]
                self.label = "QSAE/%s#s%d" % (self.name, seg)
            elif _CAPT_ON[0]:
                cap = _CAPT_ON[1]
                target = getattr(cap, "_cg_stagepairs", None)
                seg = len(getattr(cap, "_cg_bounds", ())) - 1
                self.label = "QSA/%s#s%d" % (self.name, seg)
            if target is None or torch is None:
                self.label = None
                return self
            if not _stagepool:
                _STATS["stage_pool_exhausted"] += 1
                self.label = None
                return self
            try:
                self.cap = cap
                self.pair = _stagepool.pop()
                self.t0 = time.perf_counter()
                self.pair[0].record()
            except Exception:
                self.pair = None
                self.label = None
            return self
        def __exit__(self, *exc):
            pair, self.pair = self.pair, None   # idempotent: a double exit is a no-op
            if pair is not None and self.cap is not None and self.label is not None:
                try:
                    pair[1].record()
                    if _REPLAY_ON[0]:
                        target = _REPLAY_ON[2]
                    else:
                        target = getattr(self.cap, "_cg_stagepairs", None)
                    if target is not None:
                        target.append((self.label, pair[0], pair[1]))
                        if _REPLAY_ON[0]:
                            _STAGE_HOST[self.label] += (time.perf_counter() - self.t0) * 1e3
                        if self.label not in _STAGE_SEEN:
                            _STAGE_SEEN.add(self.label)
                            _out("CGSTAT stage_live %s" % (self.label,))
                    else:
                        _stagepool.append(pair)
                except Exception:
                    _stagepool.append(pair)
            return False
    return _Stage()


def _out(*a) -> None:
    try:
        print(*a, flush=True)
    except Exception:
        pass


def _new_pair(external: bool = False):
    if external:
        return (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
    return (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))


def _ensure_pool() -> None:
    """Create the external-event pool *outside* any capture (called from the
    wrapper's _capture, before `with capture:`)."""
    global _ensured
    if _ensured or not _ON or torch is None:
        return
    _ensured = True
    try:
        t0 = time.perf_counter()
        _modpool[:] = [_new_pair(True) for _ in range(_MAX_PAIRS)]
        _out("CGSTAT pool_created pairs=%d ms=%.1f" % (_MAX_PAIRS, (time.perf_counter() - t0) * 1e3))
        t1 = time.perf_counter()
        _stagepool[:] = [_new_pair(True) for _ in range(_STAGE_POOL)]
        _out(
            "CGSTAT stage_pool_created pairs=%d ms=%.1f"
            % (_STAGE_POOL, (time.perf_counter() - t1) * 1e3)
        )
    except Exception as e:  # pragma: no cover
        _out("CGSTAT pool_failed %r" % (e,))


# --------------------------------------------------------------------------
# module hooks (fire only while a capture segment is open)
# --------------------------------------------------------------------------

def _mod_pre(module, args):
    if not _ON or not _CAPT_ON[0]:
        return None
    try:
        cls = type(module).__name__
        if cls not in _WL:
            return None
        cap = _CAPT_ON[1]
        if len(_modpool) == 0:
            _STATS["pool_exhausted"] += 1
            return None
        pre, post = _modpool.pop()
        if cls == _LAYER_CLS:
            cap._cg_depth += 1
            cap._cg_layer = getattr(module, "layer_idx", -1)
            if cap._cg_tok < 0:
                for a0 in args or ():
                    if isinstance(a0, torch.Tensor) and a0.dim() >= 1:
                        cap._cg_tok = int(a0.shape[0])
                        break
            label = "L%s/ALL" % cap._cg_layer
        else:
            label = "L%s/%s" % (cap._cg_layer, _WL[cls])
        pre.record()
        cap._cg_stack.append((label, pre, post))
    except Exception:
        pass
    return None


def _mod_post(module, args, output):
    if not _ON or not _CAPT_ON[0]:
        return None
    cls = type(module).__name__
    if cls not in _WL:
        return None
    try:
        cap = _CAPT_ON[1]
        stack = cap._cg_stack
        if not stack:
            return None
        label, pre, post = stack.pop()
        post.record()
        cap._cg_pairs.append((label, pre, post))
        if cls == _LAYER_CLS:
            cap._cg_depth -= 1
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# capture replayed: time segments, harvest the previous round
# --------------------------------------------------------------------------

_CAPT_ON = [False, None]      # [capturing?, active capture object]
_REPLAY_ON = [False, None, None, -1]  # [active?, capture, per-replay QSA pairs, seg idx]


def _begin_segment(self):
    _orig_begin(self)
    try:
        if not hasattr(self, "_cg_pairs"):
            self._cg_pairs = []
            self._cg_stagepairs = []
            self._cg_stack = []
            self._cg_bounds = []
            self._cg_depth = 0
            self._cg_layer = -1
            self._cg_tok = -1
            self._cg_desc = "?"
            self._cg_warm = True
            self._cg_cap_count = 0
        self._cg_bounds.append(len(self._cg_pairs))
        _CAPT_ON[0] = True
        _CAPT_ON[1] = self
    except Exception:
        pass


def _end_segment(self):
    try:
        _CAPT_ON[0] = False
        _CAPT_ON[1] = None
    except Exception:
        pass
    _orig_end(self)


def _wrap_capture(self, entry, args, kwargs):
    _ensure_pool()
    out = _orig_capture(self, entry, args, kwargs)
    try:
        cap = getattr(entry, "capture", None)
        if cap is not None:
            cap._cg_desc = repr(getattr(entry, "batch_descriptor", None))[:70]
            cap._cg_cap_count += 1
            _STATS["captures"] += 1
            if cap._cg_cap_count <= 40 or cap._cg_cap_count % 20 == 0:
                _log_capture_map(cap)
    except Exception:
        pass
    return out


def _log_capture_map(cap) -> None:
    try:
        segs = []
        prev = 0
        bounds = list(cap._cg_bounds) + [len(cap._cg_pairs)]
        for i, b in enumerate(bounds):
            seg = cap._cg_pairs[prev:b]
            prev = b
            cnt = collections.Counter(lb.split("/")[-1] for lb, _, _ in seg)
            segs.append("s%d[%s]" % (i, " ".join("%s:%d" % kv for kv in cnt.most_common(3))))
        _out(
            "CGSTAT capture #%d tok=%s layers=%d desc=%s pairs=%d segs=%s"
            % (
                cap._cg_cap_count,
                cap._cg_tok,
                len({lb.split("/")[0] for lb, _, _ in cap._cg_pairs}),
                cap._cg_desc,
                len(cap._cg_pairs),
                " ".join(segs),
            )
        )
    except Exception:
        pass


def _n_layers(cap) -> int:
    """How many distinct `Qwen4ExpDecoderLayer`s this capture covers (28 for the
    target model, 1 for the MTP drafter) -- identifies which replays are shown."""
    try:
        idx = set()
        for label, _a, _b in list(getattr(cap, "_cg_pairs", ())) + list(
            getattr(cap, "_cg_stagepairs", ())
        ):
            if label.endswith("/ALL"):
                idx.add(int(label[1:-4]))
        return len(idx)
    except Exception:
        return -1


def _release_extra(pairs):
    for item in pairs or ():
        try:
            _stagepool.append((item[1], item[2]))
        except Exception:
            pass


def _finalize(rd, force: bool) -> bool:
    pairs = list(rd["pairs"]) + list(rd.get("epairs") or ())
    segp = rd["segp"]
    if rd.get("warm"):
        _release_seg(segp)
        _release_extra(rd.get("epairs"))
        _STATS["warmups"] += 1
        return True
    # ---- artifact-free primary metric: per-segment GPU/host time ----
    seg_g: list[float] = []
    for e0, e1 in segp or ():
        try:
            seg_g.append(e0.elapsed_time(e1))
        except Exception:
            seg_g.append(float("nan"))
    seg_h: list[float] = list(rd["host"])
    if pairs:
        try:
            ready = all(p.query() for _, _, p in pairs)
        except Exception:
            ready = False
        if not ready:
            if not force:
                return False
            _STATS["invalid_rounds"] += 1
            _release_seg(segp)
            _release_extra(rd.get("epairs"))
            return True
    per_label: dict[str, float] = {}
    bad = 0
    for item in pairs:
        try:
            label, pre, post = item[0], item[1], item[2]
            ms = pre.elapsed_time(post)
        except Exception:
            bad += 1
            continue
        per_label[label] = per_label.get(label, 0.0) + ms
    _STATS["bad_pairs"] += bad
    with _LOCK:
        per_cls: dict[str, float] = {}
        for label, ms in per_label.items():
            a = _ACC[label]
            a[0] += 1
            a[1] += ms
            if ms > a[2]:
                a[2] = ms
            t = label.split("/")[1]
            per_cls[t] = per_cls.get(t, 0.0) + ms
        for t, ms in per_cls.items():
            c = _CLS[t]
            c[0] += 1
            c[1] += ms
            if ms > c[2]:
                c[2] = ms
        for label, ms in per_label.items():
            if label.endswith("/ALL"):
                try:
                    _LAST_LAYERS[int(label[1:-4])] = ms
                except Exception:
                    pass
        _STATS["rounds"] += 1
        if not pairs:
            _STATS["rounds_no_pairs"] += 1
        # segment summary for this round (artifact-free: a segment cannot
        # straddle a graph break, unlike a module forward)
        try:
            order = sorted(range(len(seg_g)), key=lambda i: -(seg_g[i] if seg_g[i] == seg_g[i] else -1))
            top = []
            for i in order[:5]:
                g = seg_g[i] if seg_g[i] == seg_g[i] else -1.0
                lbls = _seg_labels(rd["cap"], i)
                top.append((i, g, seg_h[i] if i < len(seg_h) else 0.0, lbls))
            _ROLL.append(
                {
                    "tok": rd["tok"],
                    "nseg": len(seg_g),
                    "sum_g": sum(x for x in seg_g if x == x),
                    "sum_h": sum(seg_h),
                    "top": top,
                    "cls": dict(per_cls),
                    "nl": rd.get("nl", -1),
                    "desc": rd.get("desc", "?"),
                }
            )
        except Exception:
            pass
        if len(_ROLL) > 12:
            del _ROLL[:-12]
    _SEG_HOST[:] = seg_h
    _SEG_GPU[:] = seg_g
    if _VERBOSE:
        try:
            rd_ = _ROLL[-1]
            _out(
                "CGSTEP nl=%s nseg=%s desc=%s sum_gpu=%.2f sum_host=%.2f cls=%s top5=%s"
                % (
                    rd_["nl"],
                    rd_["nseg"],
                    rd_["desc"],
                    rd_["sum_g"],
                    rd_["sum_h"],
                    " ".join("%s=%.1f" % kv for kv in sorted(rd_["cls"].items())),
                    " ".join("s%d:%.1f/%.2f[%s]" % t for t in rd_["top"]),
                )
            )
        except Exception:
            pass
    global _LAST_TOK
    _LAST_TOK = rd["tok"]
    _release_seg(segp)
    _release_extra(rd.get("epairs"))
    return True


def _seg_labels(cap, idx: int) -> str:
    """Module labels whose callbacks were captured inside segment `idx`."""
    try:
        bounds = list(cap._cg_bounds) + [len(cap._cg_pairs)]
        if idx + 1 >= len(bounds):
            return "?"
        seg = cap._cg_pairs[bounds[idx]:bounds[idx + 1]]
        cnt = collections.Counter(lb.split("/")[-1] for lb, _, _ in seg)
        return " ".join("%s:%d" % kv for kv in cnt.most_common(2)) or "-"
    except Exception:
        return "?"



def _release_seg(segp) -> None:
    for p in segp or ():
        _freepairs.append(p)


def _harvest(cap) -> None:
    i = 0
    while i < len(_ROUNDS):
        rd = _ROUNDS[i]
        newest = i == len(_ROUNDS) - 1
        must = rd["cap"] is cap          # its events are about to be overwritten
        if must or not newest:
            if _finalize(rd, force=must):
                _ROUNDS.pop(i)
                _STATS["harvested"] += 1
                continue
        i += 1
    if len(_ROUNDS) > 12:                 # safety valve
        for rd in _ROUNDS[:-4]:
            _finalize(rd, force=True)
        del _ROUNDS[:-4]


def _replay(self):
    if not _ON:
        return _orig_replay(self)
    try:
        _harvest(self)
        nseg = len(self.segments)
        replay_stagepairs = []
        _REPLAY_ON[0] = True
        _REPLAY_ON[1] = self
        _REPLAY_ON[2] = replay_stagepairs
        segp = [_freepairs.pop() if _freepairs else _new_pair(False) for _ in range(nseg)]
        host = []
        for i, r in enumerate(self.segments):
            e0, e1 = segp[i]
            _REPLAY_ON[3] = i
            e0.record()
            h0 = time.perf_counter()
            r()
            host.append((time.perf_counter() - h0) * 1e3)
            e1.record()
        _REPLAY_ON[0] = False
        _REPLAY_ON[1] = None
        _REPLAY_ON[2] = None
        _REPLAY_ON[3] = -1
        n_replayed = getattr(self, "_cg_replayed", 0)
        self._cg_replayed = n_replayed + 1
        _ROUNDS.append(
            {
                "cap": self,
                "pairs": getattr(self, "_cg_pairs", []) + getattr(self, "_cg_stagepairs", []),
                "epairs": replay_stagepairs,
                "segp": segp,
                "host": host,
                "tok": getattr(self, "_cg_tok", -1),
                "desc": repr(getattr(self, "_cg_desc", "?"))[:44],
                "nl": _n_layers(self),
                # First replay only reproduces the capture-time timestamps.
                "warm": n_replayed == 0,
            }
        )
        _STATS["replays"] += 1
        try:
            key = id(self)
            rec = _SEEN_CAPS.get(key)
            if rec is None:
                desc = repr(getattr(self, "_cg_desc", "?"))[:44]
                nl = _n_layers(self)
                _SEEN_CAPS[key] = [desc, nl, nseg, 1]
                _out(
                    "CGREP new#%d desc=%s nl=%d nseg=%d pairs=%d"
                    % (len(_SEEN_CAPS), desc, nl, nseg, len(getattr(self, "_cg_pairs", [])))
                )
            else:
                rec[3] += 1
        except Exception:
            pass
        if _STATS["replays"] == 1:
            _out("CGSTAT first_replay segs=%d pairs=%d" % (nseg, len(getattr(self, "_cg_pairs", []))))
    except Exception:
        _STATS["replay_errors"] += 1
        if _STATS["replay_errors"] < 4:
            _out("CGSTAT replay_error\n" + traceback.format_exc())
    return None


def _logger() -> None:
    while True:
        time.sleep(_LOG_EVERY)
        try:
            with _LOCK:
                cls_txt = " ".join(
                    "%s=%.2f/%.1f" % (k, v[1] / max(1, v[0]), v[2])
                    for k, v in sorted(_CLS.items())
                )
                lay = " ".join(
                    "%d:%.2f" % (k, _LAST_LAYERS[k]) for k in sorted(_LAST_LAYERS)
                )
                st = " ".join("%s=%d" % (k, v) for k, v in sorted(_STATS.items()))
            _out(
                "CGSTAT tok=%s seg_host=%s seg_gpu=%s | stepcls(avg/max) %s | lastL %s | %s"
                % (
                    _LAST_TOK,
                    " ".join("%.2f" % x for x in _SEG_HOST[:12]),
                    " ".join("%.2f" % x for x in _SEG_GPU[:12]),
                    cls_txt,
                    lay,
                    st,
                )
            )
            if _ROLL:
                r = _ROLL[-1]
                _out(
                    "CGWIN n=%d avg_sum_gpu=%.2f avg_sum_host=%.2f desc=%s nl=%s avg_cls=%s"
                    % (
                        len(_ROLL),
                        sum(x["sum_g"] for x in _ROLL) / len(_ROLL),
                        sum(x["sum_h"] for x in _ROLL) / len(_ROLL),
                        r.get("desc", "?"),
                        r.get("nl", -1),
                        " ".join(
                            "%s=%.1f"
                            % (k, sum(x["cls"].get(k, 0.0) for x in _ROLL) / len(_ROLL))
                            for k in sorted({k2 for x in _ROLL for k2 in x["cls"]})
                        ),
                    )
                )
            if _STAGE_HOST:
                _out(
                    "CGSTAGE(host_ms) %s"
                    % " ".join(
                        "%s=%.2f" % (k, v)
                        for k, v in sorted(_STAGE_HOST.items(), key=lambda kv: -kv[1])[:12]
                    )
                )
            if _QA:
                for tag, d in sorted(_QA.items()):
                    n = max([e[0] for e in d.values()] or [0])
                    parts = []
                    for k, e in sorted(d.items()):
                        av = e[3] / max(1, e[0])
                        parts.append(
                            ("%s=%.4g" % (k, av))
                            if e[1] == e[2]
                            else ("%s=%.4g[%.4g..%.4g]" % (k, av, e[1], e[2]))
                        )
                    _out("QSALOG %s calls=%d %s" % (tag, n, " ".join(parts)))
                _QA.clear()
        except Exception:
            pass


def install() -> None:
    global _orig_replay, _orig_capture, _orig_begin, _orig_end
    if not _ON:
        _out("CGSTAT disabled (CG_INSTR unset)")
        return
    if torch is None:
        _out("CGSTAT install_failed: torch import failed")
        return
    from vllm.compilation import breakable_cudagraph as bc

    C = bc.BreakableCUDAGraphCapture
    W = bc.BreakableCUDAGraphWrapper
    _orig_replay = C.replay
    _orig_begin = C._begin_segment
    _orig_end = C._end_segment
    _orig_capture = W._capture
    C.replay = _replay
    C._begin_segment = _begin_segment
    C._end_segment = _end_segment
    W._capture = _wrap_capture
    try:
        torch.nn.modules.module.register_module_forward_pre_hook(_mod_pre)
        torch.nn.modules.module.register_module_forward_hook(_mod_post)
    except Exception as e:
        _out("CGSTAT hook_install_failed %r" % (e,))
    threading.Thread(target=_logger, name="cg-stats", daemon=True).start()
    _out("CGSTAT installed whitelist=%s max_pairs=%d" % (",".join(_WL), _MAX_PAIRS))
