#!/usr/bin/env bash
# Allocator-vs-execution discriminator run: instrumented engine, default config.
#   clean 3x2048 -> CLEAN-state alloc battery -> >=96K trigger -> post 3x2048 -> POISONED-state alloc battery
set -u
cd /home/hong/code/qwen3.8-flash-next-cmp170hx/ops/diagnostics
log=/tmp/allocprobe.log
PY=~/dlvenv/bin/python
arm()   { docker exec hong-pc touch /tmp/qsa_probe_arm /tmp/qsa_alloc_arm; }
disarm(){ docker exec hong-pc rm -f /tmp/qsa_probe_arm /tmp/qsa_alloc_arm; }
probe() { # probe <label>
  local lbl="$1" nlines
  nlines=$(docker logs hong-pc 2>&1 | wc -l)
  arm
  $PY warmup.py 2048 >>"$log" 2>&1
  disarm
  docker logs hong-pc 2>&1 | tail -n +$((nlines + 1)) | grep -E 'QXPROBE alloc' | sed "s/^/[$lbl] /" | tee -a "$log"
}
running=$(curl -s --max-time 5 localhost:9393/metrics 2>/dev/null | awk '/^vllm:num_requests_running/{print $2; exit}')
if [ -n "${running:-}" ] && [ "${running}" != "0" ] && [ "${running}" != "0.0" ]; then
  echo "REFUSING: engine busy (num_requests_running=$running)"; exit 2
fi
{ echo "### allocator discriminator run"; date; } | tee "$log"
old=$(docker ps -q -f name=hong-pc || true)
env PORT=9393 STREAM_LOGS=0 READY_TIMEOUT=1800 QSA_OPS_PATCH=/home/hong/code/qwen3.8-flash-next-cmp170hx/ops/diagnostics/qsa_ops_instr.py \
    ./run_container.sh >>"$log" 2>&1 &
start=$(date +%s)
while :; do
  new=$(docker ps -q -f name=hong-pc || true)
  if [ -n "$new" ] && [ "$new" != "$old" ] && [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 localhost:9393/health)" = 200 ]; then break; fi
  sleep 10
  [ $(( $(date +%s) - start )) -gt 2400 ] && { echo TIMEOUT | tee -a "$log"; exit 3; }
done
echo "health=200 on ${new:0:12} after $(( $(date +%s) - start ))s" | tee -a "$log"
echo "[$(date +%H:%M:%S)] compile warmup" >>"$log"
$PY warmup.py 2048 8192 >>"$log" 2>&1
echo "[$(date +%H:%M:%S)] clean 3x2048" >>"$log"
$PY warmup.py --reps 3 2048 >/tmp/allocprobe_pre.txt 2>&1; cat /tmp/allocprobe_pre.txt >>"$log"
echo "[$(date +%H:%M:%S)] CLEAN-state alloc battery" >>"$log"
probe clean
echo "[$(date +%H:%M:%S)] >=96K trigger" >>"$log"
$PY warmup.py 98304 >/tmp/allocprobe_trig.txt 2>&1; cat /tmp/allocprobe_trig.txt >>"$log"
echo "[$(date +%H:%M:%S)] post 3x2048" >>"$log"
$PY warmup.py --reps 3 2048 >/tmp/allocprobe_post.txt 2>&1; cat /tmp/allocprobe_post.txt >>"$log"
echo "[$(date +%H:%M:%S)] POISONED-state alloc battery" >>"$log"
probe poisoned
med() { awk '$1=="fresh"{print $6}' "$1" | sort -n | awk '{a[NR]=$1} END{if(NR)printf "%s", a[int((NR+1)/2)]}'; }
c=$(med /tmp/allocprobe_pre.txt); a=$(med /tmp/allocprobe_post.txt)
{ echo "RESULT: prefill 2048 clean $c s -> post $a s"; } | tee -a "$log"
