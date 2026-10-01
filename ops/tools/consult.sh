#!/usr/bin/env bash
# 和 gpt-6-astra 会诊一条 brief（单向调用，回复落盘）。
# 何时该问、brief 怎么写、结论怎么回收：见仓库根目录 AGENTS.md 第 2 节。
#
# 用法:
#   ops/tools/consult.sh <brief.md> [编号]          # 后台（默认，15~25 分钟），立即返回
#   ops/tools/consult.sh <brief.md> 10 --wait       # 前台等待 + 进度 + 结果汇报
#   ops/tools/consult.sh --list                     # 历史会诊一览
#   ops/tools/consult.sh --check <编号>             # 查某次是否已完成（rc/字节/首行）
#
# 产物（都在 ops/consultations/）:
#   brief-<编号>.md   提问的归档副本          reply-<编号>.md   回复正文
#   reply-<编号>.err  stderr                  reply-<编号>.rc    完成标记（内容=退出码）
#
# 环境变量: CONSULT_PROVIDER(lmstudio) CONSULT_MODEL(gpt-6-astra)
#           CONSULT_THINKING(low|medium|high|xhigh|max，默认 high；minimal/off 会被端点 400 拒绝)
#           CONSULT_TIMEOUT(默认 2400)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/ops/consultations"
PROVIDER="${CONSULT_PROVIDER:-lmstudio}"
MODEL="${CONSULT_MODEL:-gpt-6-astra}"
THINKING="${CONSULT_THINKING:-high}"
TIMEOUT="${CONSULT_TIMEOUT:-2400}"

GUARD='You are a read-only reviewer for a vLLM debugging project on Windows 11 + WSL2 + a single NVIDIA CMP 170HX (SM80, 64 GiB).
Hard constraints: read-only inspection only. Do NOT restart, stop, or reconfigure the running service or the host; do NOT run GPU-heavy benchmarks. Prefer the cheapest decisive experiment, and give a judgement matrix (which outcome supports which hypothesis, which refutes it). Answer in English, concretely and quantitatively, ranked by expected value per unit cost.'

report() { # $1 = 编号
    local n="$1"
    local rc="$OUT/reply-$n.rc"
    local rp="$OUT/reply-$n.md"
    echo "  退出码   : $(cat "$rc" 2>/dev/null || echo '（仍在运行）')"
    echo "  回复     : $rp  ($(wc -c < "$rp" 2>/dev/null || echo 0) 字节)"
    [ -s "$rp" ] && { echo "  首行     : $(head -1 "$rp")"; }
    if [ "$(cat "$rc" 2>/dev/null)" != "0" ]; then
        echo "  ⚠️ 非零退出，看错误日志：$OUT/reply-$n.err"; tail -3 "$OUT/reply-$n.err" 2>/dev/null | sed 's/^/    /'
    fi
    echo "  下一步   : read $rp  →  结论按 AGENTS.md §2.5 写进 ops/OPS.md"
}

case "${1:-}" in
--list|-l)
    hr=$(printf '%.0s-' $(seq 1 70))
    echo "$hr"; echo "历史会诊：$OUT"; echo "$hr"
    for b in "$OUT"/brief-*.md; do
        [ -e "$b" ] || continue
        n="$(basename "$b" .md | sed 's/brief-//')"; r="$OUT/reply-$n.md"
        printf '  #%-5s brief %6s B' "$n" "$(stat -c%s "$b")"
        if [ -f "$r" ]; then printf '   reply %6s B   %s\n' "$(stat -c%s "$r")" "$(head -c 95 "$r" | tr '\n' ' ')"; else printf '   (无回复)\n'; fi
    done
    echo "$hr"; exit 0 ;;
--check|-c)
    n="${2:?用法: $0 --check <编号>}"; report "$n"; exit 0 ;;
esac

BRIEF="${1:-}"
[ -n "$BRIEF" ] && [ -f "$BRIEF" ] || { echo "用法: $0 <brief.md> [编号] [--wait]   （--list / --check <编号>）" >&2; exit 2; }
N="${2:-}"; WAIT="${3:-}"
case "$N" in --*|"") WAIT="${N:-}"; N="$(date +%Y%m%d-%H%M%S)" ;; esac

mkdir -p "$OUT"
REPLY="$OUT/reply-$N.md"; ERR="$OUT/reply-$N.err"; RC="$OUT/reply-$N.rc"
rm -f "$RC"; : > "$REPLY"   # 预创建，便于用字节数监控进度（也避免 wc 重定向报错）
cp -f "$BRIEF" "$OUT/brief-$N.md"   # 归档 brief（/tmp 会被清掉）

echo "会诊 #$N"
echo "  provider/model : $PROVIDER / $MODEL   thinking=$THINKING"
echo "  brief          : $BRIEF ($(wc -c < "$BRIEF") 字节) → 归档 ops/consultations/brief-$N.md"
echo "  预计            : 15~25 分钟（timeout ${TIMEOUT}s）"

# 外层包一层 bash，好在 pi 退出后把真实退出码写成 .rc（完成标记）
nohup bash -c '
  timeout "$1" pi --print --provider "$2" --model "$3" --thinking "$4" --approve \
      --append-system-prompt "$5" "$6" >"$7" 2>"$8"
  echo $? > "$9"
' _ "$TIMEOUT" "$PROVIDER" "$MODEL" "$THINKING" "$GUARD" "$(cat "$BRIEF")" \
  "$REPLY" "$ERR" "$RC" </dev/null >/dev/null 2>&1 &
PID=$!
echo "  后台 pid       : $PID"

if [ "$WAIT" = "--wait" ]; then
    t0=$(date +%s)
    while [ ! -f "$RC" ]; do
        printf '\r  等待中… %ss，已收 %s 字节   ' "$(( $(date +%s) - t0 ))" "$(wc -c < "$REPLY" 2>/dev/null || echo 0)"
        sleep 20
    done
    printf '\r%*s\r' 40 ''
    report "$N"
else
    echo "  回复           : $REPLY  ← 已完成时生成 $RC"
    echo "  等待/查询      : $0 --check $N     （或直接 wc -c $REPLY；日志尾 bin 见 $ERR）"
fi
