# Reproduction material for sgl-project/sglang#33713

Scripts used to replicate the report of 2026-09-15 on one H200 (TP=1), on v0.5.19 and on main.

- `launch_noonghunna.sh`: server launch with the reporter's device pools pinned (`--max-total-tokens 375755 --max-mamba-cache-size 55`), MTP, fp8 KV, HiCache. `HICACHE_ARGS` selects the host size (`--hicache-ratio 2.5` reproduces the miss, `--hicache-ratio 6` is the passing control).
- `probe_noonghunna.py`: one 32.7K-token sentinel chat prompt, two-turn noise conversations until ~700K prompt tokens, the sentinel again; prints `/metrics` deltas and, when the debug patch is applied, host Mamba occupancy and tree events from the server log.
- `patches/v0519_debug_logging.patch`: log lines in the Python tree core, the Mamba component and the host Mamba pool (`[TREE]`, `[MAMBA]`, `[HOSTMAMBA]`), for `git apply` on the v0.5.19 tag.
- `patches/head_debug_logging.patch`: only the host Mamba pool occupancy line, for main (Rust tree core).
- `download_model.sh`: fetches the public checkpoints.

Usage:

```
git apply patches/v0519_debug_logging.patch          # in a v0.5.19 checkout
HICACHE_ARGS="--hicache-ratio 2.5" bash launch_noonghunna.sh
python probe_noonghunna.py --model-path <snapshot dir> --server-log ~/server_noonghunna.log
```
