#!/usr/bin/env bash
# Entrypoint wrapper used only with PROFILE=1 (run_container.sh).
#
# The image's entrypoint hard-codes the `vllm serve` flag list, and vLLM only
# mounts /start_profile + /stop_profile when started with --profiler-config, so
# patch the entrypoint in place (writable layer, idempotent) and then run it.
set -euo pipefail

if ! grep -q 'enable_profiler' /opt/entrypoint.sh; then
    cat > /tmp/_patch_ep.py <<'PY'
p = "/opt/entrypoint.sh"
s = open(p).read()
ins = '''prof=()
if [[ -f /prof/enable_profiler ]]; then
    echo "[entrypoint] torch profiler enabled (marker /prof/enable_profiler)"
    prof=(--profiler-config "{\\"profiler\\":\\"torch\\",\\"torch_profiler_dir\\":\\"/prof\\",\\"ignore_frontend\\":true}")
fi

exec "$VLLM_VENV/bin/vllm" serve '''
assert s.count('exec "$VLLM_VENV/bin/vllm" serve ') == 1, "anchor 1 not found"
s = s.replace('exec "$VLLM_VENV/bin/vllm" serve ', ins)
assert s.count('    "${graph[@]}" "${spec[@]}"\n') == 1, "anchor 2 not found"
s = s.replace('    "${graph[@]}" "${spec[@]}"\n', '    "${graph[@]}" "${spec[@]}" "${prof[@]}"\n')
open(p, "w").write(s)
print("[enable_profiler_entrypoint] patched /opt/entrypoint.sh")
PY
    /opt/vllm/.venv/bin/python /tmp/_patch_ep.py
    rm -f /tmp/_patch_ep.py
fi

exec /opt/entrypoint.sh "$@"
