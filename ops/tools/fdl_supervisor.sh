#!/usr/bin/env bash
# Keep the resumable downloader alive across stalls/crashes.
export http_proxy="${http_proxy:-http://127.0.0.1:7897}"
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
while true; do
    "$HOME/dlvenv/bin/python" "$HOME/vllm/fdl.py" || echo "downloader exited ($?), restarting" 
    if [[ -f "$HOME/vllm/download-complete" ]]; then break; fi
    sleep 5
done
echo "supervisor: download complete"
