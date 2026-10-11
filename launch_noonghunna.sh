#!/usr/bin/env bash
# Replicates noonghunna's #33713 server (comment of 2026-09-15) on v0.5.19:
#   Qwen3.8-27B AutoRound INT4, fp8_e4m3 KV, native MTP n=4,
#   --enable-hierarchical-cache --hicache-size 30 --hicache-host-memory-mode cache
#   --enable-metrics --enable-cache-report, TP=2 on 2x RTX 3090.
# Differences on this node (1x H200, TP=1):
#   * pool sizes are pinned to the reporter's per-rank COUNTS
#     (device KV 375,755 tok, device mamba 55 slots); the host side is sized
#     by ratio instead of --hicache-size 30, see HICACHE_ARGS below.
#   * launched with the debug-logging patch applied to ~/sglang-0519.
VENV="${VENV:-$HOME/.venv519}"
REPO="${REPO:-$HOME/sglang-0519}"
LOG="${LOG:-$HOME/server_noonghunna.log}"
MODEL="${MODEL:-$HOME/.cache/huggingface/hub/models--Frozenlock--Qwen3.8-27B-int4-AutoRound/snapshots/b4c61732c4f2d8af323d75ba5702b5c7f3361539}"
# --hicache-size splits host memory by device-pool bytes; at TP=1 with the MTP
# intermediate state cache that gave 749K host KV tokens / 233 host Mamba states.
# --hicache-ratio 2.5 gives 939K tokens / 137 states, matching the reporter's
# per-rank state count (10.24 GB / ~75 MB per TP=2 state = ~135).
HICACHE_ARGS="${HICACHE_ARGS:---hicache-ratio 2.5}"
. "$VENV/bin/activate"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
cd "$REPO"
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port 30000 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-hierarchical-cache $HICACHE_ARGS --hicache-host-memory-mode cache \
  --enable-metrics --enable-cache-report \
  --max-total-tokens 375755 --max-mamba-cache-size 55 \
  "$@" 2>&1 | tee "$LOG"
