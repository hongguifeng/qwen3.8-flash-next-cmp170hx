#!/usr/bin/env bash
# ablation with the in-context gather SENTINEL (Astra #9 §6): restart with overrides, then
#   clean 3x2048 -> clean-state sentinel -> >=96K trigger -> post 3x2048 -> poisoned-state sentinel
#
#   ./probe_ablation.sh <tag> [VAR=VAL ...]
#
# The sentinel is _mem_battery() from qsa_ops_instr.py (armed by touching /tmp/qsa_mem_arm in the
# container; it must be DISARMED while triggering).  Clean-state V0 ~0.17 ms, poisoned-state ~39.7 ms.
set -u
tag="$1"; shift
cd /home/hong/code/qwen3.8-flash-next-cmp170hx/ops/diagnostics
log=/tmp/pabl_${tag}.log
PY=~/dlvenv/bin/python
arm()   { docker exec hong-pc touch /tmp/qsa_mem_arm; }
disarm(){ docker exec hong-pc rm -f /tmp/qsa_mem_arm; }
probe() { # probe <label> <request-length>
  local lbl="$1" n="$2" nlines
  nlines=$(docker logs hong-pc 2>&1 | wc -l)
  arm
  $PY warmup.py "$n" >>"$log" 2>&1
  disarm
  docker logs hong-pc 2>&1 | tail -n +$((nlines + 1)) | grep -E 'QXPROBE mem (cell|part2|DONE)' \
    | sed "s/^/[$lbl] /" | tee -a "$log"
}

running=$(curl -s --max-time 5 localhost:9393/metrics 2>/dev/null | awk '/^vllm:num_requests_running/{print $2; exit}')
if [ -n "${running:-}" ] && [ "${running}" != "0" ] && [ "${running}" != "0.0" ]; then
  echo "REFUSING: engine busy (num_requests_running=$running)"; exit 2
fi

{ echo "### probe-ablation '$tag'  env overrides: $*"; date; } | tee "$log"
old=$(docker ps -q -f name=hong-pc || true)
env PORT=9393 STREAM_LOGS=0 READY_TIMEOUT=1800 QSA_OPS_PATCH=/home/hong/code/qwen3.8-flash-next-cmp170hx/ops/diagnostics/qsa_ops_instr.py "$@" \
    ./run_container.sh >>"$log" 2>&1 &
start=$(date +%s)
while :; do
  new=$(docker ps -q -f name=hong-pc || true)
  if [ -n "$new" ] && [ "$new" != "$old" ] && [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 localhost:9393/health)" = 200 ]; then break; fi
  sleep 10
  [ $(( $(date +%s) - start )) -gt 2400 ] && { echo TIMEOUT | tee -a "$log"; exit 3; }
done
echo "health=200 on new container ${new:0:12} after $(( $(date +%s) - start ))s" | tee -a "$log"

echo "[$(date +%H:%M:%S)] phase: compile warmup" >>"$log"
$PY warmup.py 2048 8192 >>"$log" 2>&1
echo "[$(date +%H:%M:%S)] phase: clean 3x2048" >>"$log"
$PY warmup.py --reps 3 2048 >/tmp/pabl_${tag}_pre.txt 2>&1; cat /tmp/pabl_${tag}_pre.txt >>"$log"
echo "[$(date +%H:%M:%S)] phase: CLEAN-state sentinel" >>"$log"
probe clean 2048
echo "[$(date +%H:%M:%S)] phase: >=96K trigger" >>"$log"
$PY warmup.py 98304 >/tmp/pabl_${tag}_trig.txt 2>&1; cat /tmp/pabl_${tag}_trig.txt >>"$log"
echo "[$(date +%H:%M:%S)] phase: post 3x2048" >>"$log"
$PY warmup.py --reps 3 2048 >/tmp/pabl_${tag}_post.txt 2>&1; cat /tmp/pabl_${tag}_post.txt >>"$log"
echo "[$(date +%H:%M:%S)] phase: POISONED-state sentinel" >>"$log"
probe poisoned 2048

med() { awk '$1=="fresh"{print $6}' "$1" | sort -n | awk '{a[NR]=$1} END{if(NR)printf "%s", a[int((NR+1)/2)]}'; }
c=$(med /tmp/pabl_${tag}_pre.txt); a=$(med /tmp/pabl_${tag}_post.txt)
{
  echo "----------------------------------------------------------------"
  echo "RESULT tag=$tag (sentinel-assisted)"
  awk -v c="$c" -v a="$a" 'BEGIN{ printf "prefill 2048: clean %s s -> post %s s = %.2fx\n", c, a, (c+0>0?a/c:0) }'
  echo "sentinel V0: clean -> poisoned  (<=0.5 ms = context NOT poisoned; >=10 ms = poisoned)"
} | tee -a "$log"
