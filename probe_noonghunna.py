"""Replicate noonghunna's #33713 reproduction (comment of 2026-09-15) as closely
as the reporter described it:

  1. one salted sentinel chat prompt that templates to 32,727 tokens
  2. N distinct noise conversations (2 turns each) sized to exceed the device
     KV pool without saturating the host KV pool
  3. the sentinel again

Between steps it prints what the caches are doing, read from /metrics and from
the debug lines that the v0519_debug_logging.patch adds to the server log, so
we can see whether the device pool actually overflowed, whether the sentinel
was backed up, and when (if ever) its host Mamba state was evicted.
"""
import argparse
import json
import random
import re
import time

import requests
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:30000")
p.add_argument("--model-path", required=True, help="local snapshot dir, for the tokenizer")
p.add_argument("--server-log", required=True, help="server log to scan for [TREE]/[HOSTMAMBA] lines")
p.add_argument("--sentinel-tokens", type=int, default=32727)
p.add_argument("--noise-total-tokens", type=int, default=693_000, help="prompt tokens of noise to submit")
p.add_argument("--noise-turn-tokens", type=int, default=4096)
p.add_argument("--turns", type=int, default=2)
p.add_argument("--max-new-tokens", type=int, default=64)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--settle", type=float, default=5.0, help="seconds to wait for async backups")
p.add_argument("--summary", default="probe_summary.json")
args = p.parse_args()

rng = random.Random(args.seed)
tok = AutoTokenizer.from_pretrained(args.model_path)
WORDS = [f"w{i}" for i in range(5000)]


def filler(n_words, r):
    return " ".join(r.choice(WORDS) for _ in range(n_words))


def templated_len(messages):
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):  # transformers v5 returns a BatchEncoding
        ids = ids["input_ids"]
    return len(ids)


def templated_text(messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_user_content(target_tokens, salt, r):
    """Binary-search the word count so the templated single-turn prompt is exactly target_tokens."""
    lo, hi = 1, target_tokens * 2
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        rr = random.Random(r.random())
        content = f"{salt} " + filler(mid, rr)
        n = templated_len([{"role": "user", "content": content}])
        if n < target_tokens:
            lo = mid + 1
        elif n > target_tokens:
            hi = mid - 1
        else:
            return content
        if best is None or abs(n - target_tokens) < abs(best[0] - target_tokens):
            best = (n, content)
    return best[1]


# ---------------------------------------------------------------- metrics ----
METRIC_RE = re.compile(r"^(sglang:[a-z_]+)(\{[^}]*\})? ([0-9.e+-]+)$")


def metrics():
    out = {}
    for line in requests.get(f"{args.url}/metrics", timeout=30).text.splitlines():
        m = METRIC_RE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1).replace("sglang:", ""), m.group(2) or "", float(m.group(3))
        if "bucket" in name or "duration" in name or "_bytes" in name:
            continue
        pool = re.search(r'pool="(\w+)"', labels)
        src = re.search(r'cache_source="(\w+)"', labels)
        reason = re.search(r'reason="(\w+)"', labels)
        key = name + (f"[{pool.group(1)}]" if pool else "") + (f"[{src.group(1)}]" if src else "") + (f"[{reason.group(1)}]" if reason else "")
        out[key] = out.get(key, 0) + val
    return out


def server_log_state():
    """Occupancy of the host Mamba pool and counts of tree events, from the debug patch."""
    st = {"host_mamba_used": None, "host_mamba_size": None, "events": {}, "sentinel_lines": []}
    try:
        with open(args.server_log, errors="replace") as f:
            for line in f:
                if "[HOSTMAMBA]" in line:
                    m = re.search(r"used=(\d+)/(\d+)", line)
                    if m:
                        st["host_mamba_used"], st["host_mamba_size"] = int(m.group(1)), int(m.group(2))
                elif "[TREE]" in line or "[MAMBA]" in line:
                    ev = re.search(r"\[(TREE|MAMBA)\] (\S+)", line)
                    if ev:
                        k = f"{ev.group(1)}:{ev.group(2)}"
                        st["events"][k] = st["events"].get(k, 0) + 1
                    if "SENTINEL" in line or re.search(r"depth=(\d+)", line) and int(re.search(r"depth=(\d+)", line).group(1)) >= 20000:
                        st["sentinel_lines"].append(line.rstrip()[-220:])
    except FileNotFoundError:
        pass
    return st


def status(tag, m0, m1):
    d = {k: m1[k] - m0.get(k, 0) for k in m1 if m1[k] - m0.get(k, 0)}
    pick = lambda k: d.get(k, 0)
    st = server_log_state()
    print(
        f"  [{tag}] Δevicted={pick('evicted_tokens_total'):.0f} "
        f"Δbackup kv/mamba={pick('hicache_backup_tokens_total[kv]'):.0f}/{pick('hicache_backup_tokens_total[mamba]'):.0f} "
        f"Δload_back kv/mamba={pick('load_back_tokens_total[kv]'):.0f}/{pick('load_back_tokens_total[mamba]'):.0f} "
        f"Δdropped={sum(v for k, v in d.items() if k.startswith('hicache_dropped')):.0f} | "
        f"hostKV used/total={m1.get('hicache_host_used_tokens', 0):.0f}/{m1.get('hicache_host_total_tokens', 0):.0f} "
        f"devMamba evictable/avail={m1.get('mamba_evictable_tokens', 0):.0f}/{m1.get('mamba_available_tokens', 0):.0f} "
        f"hostMamba used/size={st['host_mamba_used']}/{st['host_mamba_size']} "
        f"events={st['events']}",
        flush=True,
    )
    return st


# ---------------------------------------------------------------- requests ----
def gen(text, tag):
    t0 = time.time()
    r = requests.post(
        f"{args.url}/generate",
        json={"text": text, "sampling_params": {"max_new_tokens": args.max_new_tokens, "temperature": 0}},
        timeout=1200,
    )
    r.raise_for_status()
    j = r.json()
    mi = j["meta_info"]
    print(
        f"{tag:<14} prompt={mi['prompt_tokens']:>6} cached={mi.get('cached_tokens', 0):>6} "
        f"details={mi.get('cached_tokens_details')} e2e={time.time() - t0:6.2f}s",
        flush=True,
    )
    return j["text"], mi


summary = {"args": vars(args), "phases": {}}

# ---- phase 1: sentinel -------------------------------------------------------
sent_messages = [{"role": "user", "content": build_user_content(args.sentinel_tokens, "SENTINEL-salt-7f3a9c", rng)}]
sent_text = templated_text(sent_messages)
print(f"== phase 1: sentinel ({templated_len(sent_messages)} templated tokens)", flush=True)
m0 = metrics()
_, mi = gen(sent_text, "sentinel[cold]")
time.sleep(args.settle)
m1 = metrics()
summary["phases"]["sentinel_cold"] = {"meta": mi, "state": status("after sentinel", m0, m1)}

# ---- phase 2: noise ----------------------------------------------------------
print(f"== phase 2: noise conversations, {args.turns} turns x ~{args.noise_turn_tokens} tokens, until {args.noise_total_tokens} prompt tokens submitted", flush=True)
submitted = 0
conv_i = 0
m_prev = m1
while submitted < args.noise_total_tokens:
    r = random.Random(args.seed * 100_003 + conv_i)
    messages = []
    for turn in range(args.turns):
        messages.append({"role": "user", "content": build_user_content(args.noise_turn_tokens, f"noise-{conv_i}-{turn}", r)})
        reply, mi = gen(templated_text(messages), f"noise[{conv_i}].{turn}")
        submitted += mi["prompt_tokens"]
        messages.append({"role": "assistant", "content": reply})
    conv_i += 1
    if conv_i % 5 == 0:
        m_now = metrics()
        status(f"noise {conv_i} convs, {submitted} tok", m_prev, m_now)
        m_prev = m_now
time.sleep(args.settle)
m2 = metrics()
print(f"-- noise done: {conv_i} conversations, {submitted} prompt tokens submitted", flush=True)
summary["phases"]["noise"] = {"conversations": conv_i, "submitted": submitted, "state": status("after noise", m1, m2)}

# ---- phase 3: sentinel again -------------------------------------------------
print("== phase 3: sentinel return", flush=True)
_, mi = gen(sent_text, "sentinel[ret]")
time.sleep(args.settle)
m3 = metrics()
st = status("after return", m2, m3)
summary["phases"]["sentinel_return"] = {"meta": mi, "state": st}

d = mi.get("cached_tokens_details") or {}
dev, host = d.get("device", 0), d.get("host", 0)
print(f"\nRESULT: device={dev} host={host} cached={mi.get('cached_tokens')} (sentinel={args.sentinel_tokens})")
print("BUG REPRODUCED (no device/host hit on return)" if dev == 0 and host == 0 else "no bug: sentinel served from cache")
print("\n-- server-log lines about the sentinel (depth>=20000):")
for line in st["sentinel_lines"][-40:]:
    print("  ", line)
with open(args.summary, "w") as f:
    json.dump(summary, f, indent=1, default=str)
