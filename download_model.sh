#!/usr/bin/env bash
# Runs on the node: download a public HF model into the shared cache (no token needed).
. "$HOME/.venv/bin/activate"
python - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    print("downloading", repo, flush=True)
    print(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "*.jinja", "*.model", "*.tiktoken"]), flush=True)
print("DOWNLOAD_DONE", flush=True)
PY
