#!/usr/bin/env bash
# 跑一次标准体检，并把结果**永久存档**（每次运行都追加一行历史）。
#
# 用法:
#   bin/bench.sh                 # 2048/8192 预填 + 解码（约 1.5 分钟）
#   bin/bench.sh --model unc     # 体检另一个档位（默认 main；--model list 看有哪些）
#   bin/bench.sh --full          # 追加 131072 长上下文 + 事后“中毒”检查（约 3 分钟）
#   bin/bench.sh --reps 3        # 预填每个长度重复次数（默认 2）
#   bin/bench.sh --prefill       # 只测预填
#   bin/bench.sh --decode        # 只测解码
#   bin/bench.sh --tag mtp3      # 给这次记录打标签（非 main 档位会自动写成 mtp3@unc，避免混档）
#   bin/bench.sh --no-save       # 不写存档
#   bin/bench.sh --params        # 只打印生效参数（含当前引擎端口）
#   bin/bench.sh --help
#
# 存档位置:
#   ops/measurements/perf-history.csv     每次运行一行（追加，永久保留）
#   ops/measurements/bench-<时间戳>.log   本次原始输出
# 判读基线（见 docs/RESULTS-WSL2.md）:
#   prefill 2048 ≈ 0.64~0.73 s；8192 ≈ 2.0~2.4 s；131072 ≈ 54~56 s
#   decode 步时 ≈ 15.2~15.7 ms（MTP=1）/ 17.2~17.6 ms（MTP=2）
#   decode tok/s = (1 + 接受的 draft 数) / 步时 ⇒ MTP=2 约 118~131
#
# ⚠️ 解码必须在“充分预热 + 宿主低压”下测：脚本会在解码前自动预热 2560 token，
#    并在开始时检查 vmmemWSL 是否已缩回（>32 GB 会提示先跑 bin/drop_host_cache.sh）。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# 探针（ops/bench/*.py）不硬编码端口：它们优先读 BASE_URL/QWEN_PORT（见 ops/bench/_cfg.py）
export BASE_URL QWEN_PORT QWEN_SERVED_NAME MODEL_NAME

REPS=2; FULL=0; DO_PREFILL=1; DO_DECODE=1; SAVE=1; TAG=manual; MODEL_ARG=""; PARAMS_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL=1 ;;
        --reps) REPS="${2:-2}"; shift ;;
        --prefill) DO_DECODE=0 ;;
        --decode) DO_PREFILL=0 ;;
        --tag) TAG="${2:-manual}"; shift ;;
        --no-save) SAVE=0 ;;
        --model) MODEL_ARG="${2:-}"; shift ;;
        --params) PARAMS_ONLY=1 ;;          # 不直接输出：要先让 --model 生效
        -h|--help) help_exit "$0" ;;
        *) echo "未知参数: $1（--help 看用法）" >&2; exit 2 ;;
    esac
    shift
done

# ---- 选档位（必须在探活/探针之前）------
case "${MODEL_ARG:-}" in
    '') ;;
    list|ls) variant_list; exit 0 ;;
    *) set_variant "$MODEL_ARG" || exit 2 ;;
esac
if [ "$PARAMS_ONLY" = 1 ]; then print_params; exit 0; fi
# 非默认档位：tag 自动带 @档位，否则两支模型的数据在历史表里分不出来
[ "$VARIANT" != main ] && TAG="$TAG@$VARIANT"

[ "$(health_code)" = "200" ] || { c_err "档位 $VARIANT 的服务未就绪（$BASE_URL/health ≠ 200），先 bin/start.sh --model $VARIANT"; exit 1; }

OUTDIR="$ROOT/ops/measurements"; HIST="$OUTDIR/perf-history.csv"
mkdir -p "$OUTDIR"
TS="$(date '+%Y-%m-%dT%H:%M:%S')"
MTP_NOW="$(tr '\0' ' ' < "/proc/$(engine_pid)/cmdline" 2>/dev/null | grep -oP '"num_speculative_tokens":\K[0-9]+' || true)"
SUFFIX=""; [ "$VARIANT" != main ] && SUFFIX="-$VARIANT"
RAW="$OUTDIR/bench${SUFFIX}-$(date '+%Y%m%d-%H%M%S').log"
exec > >(tee -a "$RAW") 2>&1

echo "# bench $TS  tag=$TAG  variant=$VARIANT  gpu=${CUDA_VISIBLE_DEVICES}  model=$MODEL_NAME"
echo "# 接受率 $("$ROOT/bin/status.sh" --short --model "$VARIANT" | grep -oP 'acc=\S+')  接口=$BASE_URL"
echo "# 宿主: $(windows_host_mem)   guest: $(free -m | sed -n 2p | awk '{print "used="$3"MB free="$4"MB"}')"
host_ws="$(windows_host_mem | grep -oP 'vmmem=\K[0-9.]+' || echo 0)"
if [ "$DO_DECODE" = 1 ] && awk -v w="$host_ws" 'BEGIN{exit !(w>32)}'; then
    echo "$(c_warn "⚠️ vmmemWSL=${host_ws}GB > 32GB：宿主内存高压会让解码步时虚高 25~40%，建议先跑 bin/drop_host_cache.sh")"
fi

if [ "$DO_PREFILL" = 1 ]; then
    hr; echo "① 新鲜 prefill（全新 token id，剔除编译/缓存干扰）"; hr
    if [ "$FULL" = 1 ]; then
        python3 "$ROOT/ops/bench/warmup.py" --reps "$REPS" 2048 8192 131072
        echo; echo "② 长上下文后置检查（全新 2048；若 >1.5s 说明“中毒”未被治愈）"
        python3 "$ROOT/ops/bench/warmup.py" --reps 2 2048 | tail -3
    else
        python3 "$ROOT/ops/bench/warmup.py" --reps "$REPS" 2048 8192
    fi
fi

if [ "$DO_DECODE" = 1 ]; then
    hr; echo "③ 解码前预热 2560 token（否则步时虚高）"; hr
    python3 - "$MODEL_NAME" "$BASE_URL" <<'PY'
import json,sys,time,urllib.request
M=sys.argv[2]; t0=time.time(); tot=0
for _ in range(10):
    body=json.dumps({"model":sys.argv[1] if len(sys.argv)>1 else "Qwen3.8-Flash-Next",
        "prompt":"请详细说明 vLLM 中 PagedAttention 的工作原理。","max_tokens":256,
        "temperature":0.6,"ignore_eos":True}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(M+"/v1/completions",body,
        {"Content-Type":"application/json"}),timeout=600)); tot+=r["usage"]["completion_tokens"]
print(f"  预热 {tot} token / {time.time()-t0:.1f}s")
PY
    hr; echo "④ 解码速度 / MTP 接受率（流式，剔除 TTFT）"; hr
    python3 "$ROOT/ops/bench/dec_bench.py"
fi

hr
if [ "$SAVE" = 1 ]; then
    python3 - "$RAW" "$HIST" "$TS" "$TAG" "${MTP_NOW:-?}" <<'PY'
import re, sys, os, statistics as st
raw, hist, ts, tag, mtp = sys.argv[1:6]
t = open(raw, encoding="utf-8", errors="ignore").read()
def med(vals):
    return round(st.median(vals), 4) if vals else ""
pf = {}
for m in re.finditer(r'^\s+(\d+) tok\s+best of \d+\s+([\d.]+) s\s+(\d+) tok/s', t, re.M):
    pf[int(m.group(1))] = float(m.group(2))
tps, steps, accs, toks = [], [], [], []
for m in re.finditer(r'max_tokens|completion=(\d+) 接受率=([\d.]+)%', t):
    if m.group(1):
        toks.append(int(m.group(1))); accs.append(float(m.group(2)))
for m in re.finditer(r'纯解码=[\d.]+s ⇒ ([\d.]+) tok/s', t):
    tps.append(float(m.group(1)))
for m in re.finditer(r'步间 p50=([\d.]+)', t):
    steps.append(float(m.group(1)))
host = {}
hm = re.search(r'vmmem=([\d.]+)GB', t); hf = re.search(r'free=([\d.]+)GB', t)
if hm: host["vmmem_gb"] = float(hm.group(1))
if hf: host["free_gb"] = float(hf.group(1))
row = [ts, tag, mtp,
       pf.get(2048, ""), pf.get(8192, ""), pf.get(131072, ""),
       med([v for v in tps]), med(steps), med(accs), med(toks),
       host.get("free_gb", ""), host.get("vmmem_gb", ""), os.path.basename(raw)]
hdr = ["timestamp", "tag", "mtp", "prefill2048_s", "prefill8192_s", "prefill131072_s",
       "decode_tok_s_med", "step_p50_ms_med", "accept_pct_med", "completion_tokens_med",
       "host_free_gb", "vmmem_gb", "raw_log"]
new = not os.path.exists(hist)
with open(hist, "a", encoding="utf-8") as f:
    if new: f.write(",".join(hdr) + "\n")
    f.write(",".join(str(x) for x in row) + "\n")
print("已存档：")
print("  " + ",".join(str(x) for x in row))
print(f"  历史表: {hist}\n  原始输出: {raw}")
PY
else
    echo "（--no-save：本次未存档）"
fi
hr
echo "对照基线: docs/RESULTS-WSL2.md   失败排查: bin/logs.sh -e"
