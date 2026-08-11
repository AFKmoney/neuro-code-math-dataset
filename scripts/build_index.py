#!/usr/bin/env python3
"""Build an index.jsonl listing every file with its entry count and size."""
import os, json

DATA = "/home/z/my-project/neuro-code-math/data"
out_path = os.path.join(DATA, "index.jsonl")

entries = []
total_entries = 0
total_bytes = 0
total_tokens = 0

for fn in sorted(os.listdir(DATA)):
    if not fn.endswith(".jsonl") or fn == "index.jsonl":
        continue
    path = os.path.join(DATA, fn)
    size = os.path.getsize(path)
    n = 0
    chars = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            n += 1
            e = json.loads(line)
            for m in e["messages"]:
                chars += len(m["content"])
    tokens = chars // 4
    domain, subtopic = fn.replace(".jsonl", "").split("__", 1)
    entries.append({
        "file": fn,
        "domain": domain,
        "subtopic": subtopic,
        "entries": n,
        "bytes": size,
        "approx_tokens": tokens,
    })
    total_entries += n
    total_bytes += size
    total_tokens += tokens

# Sort by domain order then subtopic
order = {"applied_neuroscience": 0, "rust": 1, "python": 2, "math": 3}
entries.sort(key=lambda e: (order.get(e["domain"], 99), e["subtopic"]))

with open(out_path, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"Wrote {out_path}")
print(f"Total files: {len(entries)}")
print(f"Total entries: {total_entries:,}")
print(f"Total bytes: {total_bytes:,}")
print(f"Approx total tokens: {total_tokens:,}")

# Per-domain summary
print("\n--- Per-domain summary ---")
by_domain = {}
for e in entries:
    d = e["domain"]
    if d not in by_domain:
        by_domain[d] = {"files": 0, "entries": 0, "tokens": 0, "bytes": 0}
    by_domain[d]["files"] += 1
    by_domain[d]["entries"] += e["entries"]
    by_domain[d]["tokens"] += e["approx_tokens"]
    by_domain[d]["bytes"] += e["bytes"]

for d in ["applied_neuroscience", "rust", "python", "math"]:
    s = by_domain[d]
    print(f"{d:22s}  files={s['files']:3d}  entries={s['entries']:>9,}  tokens={s['tokens']:>12,}  size={s['bytes']/1024/1024:>8.1f} MB")
