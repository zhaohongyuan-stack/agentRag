"""终端检索 — 改下面 JSON，然后 python query.py"""

# ═══════════ 在这改 ═══════════
Q = {
  "query": "寿险合同负债评估 折现率曲线 基础利率曲线 综合溢价",
  "strategy": "hybrid",
  "top_k": 5
}


# ═══════════════════════════════

import json, sys, urllib.request

sys.stdout.reconfigure(encoding="utf-8")

data = json.dumps(Q, ensure_ascii=False).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/v1/search",
    data=data,
    headers={"Content-Type": "application/json; charset=utf-8"},
)
with urllib.request.urlopen(req) as resp:
    hits = json.loads(resp.read().decode("utf-8"))

print(json.dumps(hits, ensure_ascii=False, indent=2))
