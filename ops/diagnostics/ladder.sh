#!/usr/bin/env bash
# Long-context ladder: restart with ENV overrides, then step the prompt length up to the full
# context, using a fresh 3x2048 prefill as the probe after every step.
#
#   ./ladder.sh <tag> [VAR=VAL ...]
#
# A healthy engine keeps the 2048 probe at ~0.65-0.70 s at EVERY length; a regression shows up
# as 1.5-2.6 s (or worse) and as a slower long prefill (tok/s).
set -u
cd /home/hong/code/qwen3.8-flash-next-cmp170hx/ops/diagnostics || exit 1
tag="$1"; shift
log=/tmp/ladder_${tag}.log
PY=~/dlvenv/bin/python
NS=/mnt/c/Windows/System32/nvidia-smi.exe
LENGTHS="${LADDER_LENGTHS:-131072 196608 262143}"

busy=$(curl -s --max-time 5 localhost:9393/metrics 2>/dev/null | awk '/^vllm:num_requests_running/{print $2; exit}')
if [ -n "${busy:-}" ] && [ "$busy" != "0" ] && [ "$busy" != "0.0" ]; then
  echo "REFUSING: engine busy (num_requests_running=$busy)"; exit 2
fi

{ echo "### ladder '$tag'  env: $*  lengths: $LENGTHS"; date; } | tee "$log"
old=$(docker ps -q -f name=hong-pc || true)
env PORT=9393 STREAM_LOGS=0 READY_TIMEOUT=1800 "$@" ./run_container.sh >>"$log" 2>&1 &
start=$(date +%s)
while :; do
  new=$(docker ps -q -f name=hong-pc || true)
  if [ -n "$new" ] && [ "$new" != "$old" ] && [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 localhost:9393/health)" = 200 ]; then break; fi
  sleep 10
  [ $(( $(date +%s) - start )) -gt 2400 ] && { echo TIMEOUT | tee -a "$log"; exit 3; }
done
echo "health=200 on ${new:0:12} after $(( $(date +%s) - start ))s" | tee -a "$log"

probe() {  # prints "median (v1 v2 v3)"
  $PY warmup.py --reps 3 2048 2>&1 | awk '$1=="fresh" && $3=="2048"{v[++n]=$6} END{
      m=""; for(i=1;i<=n;i++) for(j=i+1;j<=n;j++) if(v[j]<v[i]){t=v[i];v[i]=v[j];v[j]=t}
      printf "%.3f  (%s)\n", v[int((n+1)/2)], v[1]" "v[2]" "v[3] }'
}

echo "[$(date +%H:%M:%S)] compile warmup" >>"$log"
$PY warmup.py 2048 8192 >>"$log" 2>&1
echo "baseline 2048 = $(probe)" | tee -a "$log"
for L in $LENGTHS; do
  pre=$(probe)
  echo "--- length $L : pre=$pre  free=$(timeout 20 $NS --query-gpu=memory.free --format=csv,noheader | head -1)" | tee -a "$log"
  trig=$($PY warmup.py "$L" 2>&1 | awk '$1=="fresh"{print $4" s ("$5" "$6")"}')
  post=$(probe)
  echo "    trigger $L = $trig" | tee -a "$log"
  echo "    post 2048 = $post   free=$(timeout 20 $NS --query-gpu=memory.free --format=csv,noheader | head -1)" | tee -a "$log"
done
echo "=== heal thread activity ===" | tee -a "$log"
docker logs hong-pc 2>&1 | grep -c 'QXHEAL fired' | sed 's/^/QXHEAL fires: /' | tee -a "$log"
docker logs hong-pc 2>&1 | grep 'QXHEAL fired' | tail -3 | tee -a "$log"
curl -s --max-time 5 localhost:9393/metrics | awk '/^vllm:num_requests_running/{print "running="$2}' | tee -a "$log"
