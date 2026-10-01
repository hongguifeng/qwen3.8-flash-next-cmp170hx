import os, time
from huggingface_hub import snapshot_download
tgt = os.path.expanduser("~/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP")
snapshot_download(
    repo_id="klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP",
    revision="ce0e0b94083895bd836b916f29bf105c40a8162a",
    local_dir=tgt, max_workers=8,
)
print("DONE", tgt)
