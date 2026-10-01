#!/usr/bin/env bash
# Astra#9 stage-2 ablation runner:  restart with ENV=V overrides, then screen the >=96K poison.
#
#   ./ablation.sh <tag> [VAR=VAL ...]
#
# Screening protocol (per consultation #9 §6):
#   compile warmup -> 3x fresh 2048 (clean median) -> 1x fresh 98304 trigger -> 3x fresh 2048 (post median)
#   verdict: post/clean <= 1.2  => candidate survives;  >= 2.0 => poison still happens (feature not the trigger)
# Everything is logged to /tmp/abl_<tag>.log ; per-phase raw output in /tmp/abl_<tag>_{pre,trig,post}.txt
set -u
tag="$1"; shift
cd /home/hong/code/qwen3.8-flash-next-cmp170hx/ops/diagnostics
log=/tmp/abl_${tag}.log
pre=/tmp/abl_${tag}_pre.txt; trig=/tmp/abl_${tag}_trig.txt; post=/tmp/abl_${tag}_post.txt

running=$(curl -s --max-time 5 localhost:9393/metrics 2>/dev/null | awk '/^vllm:num_requests_running/{print $2; exit}')
if [ "${running:-0}" != "0" ] && [ "${running:-0}" != "0.0" ] && [ -n "${running:-}" ]; then
  echo "REFUSING: engine busy (num_requests_running=$running)"; exit 2
fi

{
  echo "### ablation '$tag'  env overrides: $*"
  date
} | tee "$log"

old=$(docker ps -q -f name=hong-pc || true)
env PORT=9393 STREAM_LOGS=0 READY_TIMEOUT=1800 "$@" ./run_container.sh >>"$log" 2>&1 &
start=$(date +%s)
# wait for the NEW container (different id than the one that was running) to appear and answer health
while :; do
  new=$(docker ps -q -f name=hong-pc || true)
  if [ -n "$new" ] && [ "$new" != "$old" ] && [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 localhost:9393/health)" = 200 ]; then break; fi
  sleep 10
  if [ $(( $(date +%s) - start )) -gt 2400 ]; then echo "TIMEOUT waiting for new container" | tee -a "$log"; exit 3; fi
done
echo "health=200 on new container ${new:0:12} after $(( $(date +%s) - start ))s" | tee -a "$log"

PY=~/dlvenv/bin/python
echo "[$(date +%H:%M:%S)] phase: compile warmup" >>"$log"
$PY warmup.py 2048 8192 >>"$log" 2>&1                 # triton/jit compile warmup (not measured)
echo "[$(date +%H:%M:%S)] phase: clean baseline 3x2048" >>"$log"
$PY warmup.py --reps 3 2048 >"$pre"  2>&1; cat "$pre"  >>"$log"
echo "[$(date +%H:%M:%S)] phase: >=96K trigger" >>"$log"
$PY warmup.py 98304         >"$trig" 2>&1; cat "$trig" >>"$log"
echo "[$(date +%H:%M:%S)] phase: post-trigger 3x2048" >>"$log"
$PY warmup.py --reps 3 2048 >"$post" 2>&1; cat "$post" >>"$log"

median() { awk '$1=="fresh" && $3=="2048"{print $6}' "$1" | sort -n | awk '{a[NR]=$1} END{ if(NR==0) print "NA"; else print a[int((NR+1)/2)] }'; }
clean=$(median "$pre"); after=$(median "$post")
trigs=$(awk '$1=="fresh"{print $3" tok " $6" s"}' "$trig")
{
  echo "----------------------------------------------------------------"
  echo "RESULT tag=$tag  trigger: $trigs"
  echo "clean 2048 median = $clean s"
  echo "post  2048 median = $after s"
  awk -v c="$clean" -v a="$after" 'BEGIN{ if(c>0) printf "RATIO post/clean = %.2fx   verdict: %s\n", a/c, (a/c<=1.2 ? "SURVIVED (feature is implicated)" : (a/c>=2.0 ? "POISONED (feature is NOT the trigger)" : "AMBIGUOUS")) }'
  echo "container: $(docker ps --format '{{.ID}} {{.Names}}' | grep hong-pc)"
} | tee -a "$log"
