#!/usr/bin/env bash
# Downtime window for minrepro4: stop engine -> run the high-occupancy+churn reproducer (WSL2 arm)
# -> restart the engine with the default config and verify it is healthy.
set -u
cd /home/hong/code/qwen3.8-flash-next-cmp170hx/ops/diagnostics
log=/tmp/minrepro4_run.log
NS=/mnt/c/Windows/System32/nvidia-smi.exe
running=$(curl -s --max-time 5 localhost:9393/metrics 2>/dev/null | awk '/^vllm:num_requests_running/{print $2; exit}')
if [ -n "${running:-}" ] && [ "${running}" != "0" ] && [ "${running}" != "0.0" ]; then
  echo "REFUSING: engine busy (num_requests_running=$running)"; exit 2
fi
{
  echo "=== [$(date +%H:%M:%S)] 停引擎腾显存 ==="
  docker rm -f hong-pc
  sleep 8
  timeout 60 $NS --query-gpu=memory.used,memory.free --format=csv,noheader
  echo "=== [$(date +%H:%M:%S)] WSL 臂：高占用 + 分配 churn ==="
  timeout 2400 docker run --rm --device nvidia.com/gpu=all -v /home/hong/code/qwen3.8-flash-next-cmp170hx/ops/diagnostics/minrepro4.py:/m.py:ro \
      18gogogo/170hx1-qwen38nextf:sm80 /opt/vllm/.venv/bin/python /m.py \
      --ballast-gb 58 --churn 0,2000,20000
  echo "[$(date +%H:%M:%S)] WSL 臂退出码 $?"
  echo "=== [$(date +%H:%M:%S)] 恢复引擎（默认配置）==="
  env PORT=9393 STREAM_LOGS=0 READY_TIMEOUT=1800 ./run_container.sh
  echo "=== [$(date +%H:%M:%S)] 服务核验 ==="
  curl -s -o /dev/null -w 'health=%{http_code}\n' localhost:9393/health
  ~/dlvenv/bin/python warmup.py 2048 2>&1 | tail -2
  curl -s --max-time 5 localhost:9393/metrics | awk '/^vllm:num_requests_running/{print "running="$2}'
  timeout 60 $NS --query-gpu=memory.used,memory.free --format=csv,noheader
} >"$log" 2>&1
