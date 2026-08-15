"""终端检索客户端 — 改下面 Q，然后 python query.py

用法：
  1. 先启动服务（另开终端）:
       python -m retrieval_service.server
  2. 改下面的 Q（query / strategy / top_k / target_industries）
  3. 运行 python query.py 查询
"""

# ═══════════ 在这改 ═══════════
Q = {
  "query": "不良贷款余额",           # ← 只写指标名（表格行匹配用）
  "strategy": "table",              # ← 查表用 table 策略
  "top_k": 10,
  # 行业分区过滤（可选）: ["银行业"] | ["保险业"] | ["其他"] | ["银行业","保险业"]
  # 不填 = 自动从 filters 推断，无法推断则搜全部三个分区
  "target_industries": ["银行业"],
  # 表格限定（查表必须）: table_name 限定表，doc_name 限定回退搜索范围
  "filters": {
    "table_name": "商业银行主要指标分机构类情况表(季度)(2023年)",
    "doc_name": "130_2023年商业银行主要指标分机构类情况表（季度）_商业银行主要指标分机构类情况表(季度)(2023年).xlsx",
  },
}



# ═══════════════════════════════

import json, sys, urllib.request

sys.stdout.reconfigure(encoding="utf-8")

data = json.dumps(Q, ensure_ascii=False).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8001/api/v1/search",
    data=data,
    headers={"Content-Type": "application/json; charset=utf-8"},
)
with urllib.request.urlopen(req) as resp:
    hits = json.loads(resp.read().decode("utf-8"))

for h in hits:
    print(f"[{h.get('rank')}] {h.get('chunk_type')} | 《{h.get('doc_name', '')}》 | score={h.get('score', 0):.4f}")
    print(f"    出处: {h.get('citation', '')[:120]}")
    print(f"    内容: {h.get('content', '')[:200].replace(chr(10), ' ')}")
    print()
