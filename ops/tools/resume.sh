#!/usr/bin/env bash
# Resume after a reboot: restart the checkpoint download and the container starter.
set -euo pipefail

export http_proxy="${http_proxy:-http://127.0.0.1:7897}"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP}"
DEST="$MODEL_DIR"

echo "state: $DEST/.fdl-state.json"
python3 - "$DEST" <<'PY'
import json, os, sys
d = sys.argv[1]
s = json.load(open(os.path.join(d, ".fdl-state.json")))
CH = 16 * 1024 * 1024
total = 0
for name, idxs in s.items():
    part = os.path.join(d, name + ".part")
    size = os.path.getsize(part) if os.path.exists(part) else 0
    total += sum(min(CH, size - i * CH) for i in set(idxs))
print(f"resumable bytes: {total / 2**30:.2f} GiB")
PY

if ! screen -ls | grep -q '[.]hfdl[[:space:]]'; then
    screen -dmS hfdl -L -Logfile "$HOME/vllm/fdl.log" \
        bash -c "exec $HOME/dlvenv/bin/python $HOME/vllm/fdl.py"
    echo "started screen 'hfdl' (download)"
else
    echo "screen 'hfdl' already running"
fi

if ! screen -ls | grep -q '[.]qwen-start[[:space:]]'; then
    screen -dmS qwen-start bash -c \
        "bash $HOME/vllm/run_container.sh > $HOME/vllm/start.log 2>&1"
    echo "started screen 'qwen-start' (waits for download, then runs the container)"
else
    echo "screen 'qwen-start' already running"
fi

echo "monitor:  tail -f $HOME/vllm/fdl.log"
echo "screens:  screen -r hfdl | screen -r qwen-start"
