"""
检索调用示例
"""

import json
from retrieval_service import Retriever, RetrievalAPI

# 加载数据（启动时一次）
api = RetrievalAPI()
api.load("regulatory_docs/")
r = Retriever("regulatory_docs/").load()

# ── 接口一：统一 Retrieval API → 返回 RetrievalHit ──
hits = r.search("核心一级资本合格标准", top_k=5, lightweight=False)

for h in hits:
    print(json.dumps(h, ensure_ascii=False))

# ── 接口二：文档与 chunk 查询 API → 返回 JSON ──
# 查 chunk
chunk = api.get_chunk("001_s1_summary")
print(json.dumps(chunk, ensure_ascii=False))

# 查文档
doc = api.get_document("001")
print(json.dumps(doc, ensure_ascii=False))

# 列出文档（只展示前 3 个）
for d in api.list_documents()[:3]:
    print(json.dumps(d, ensure_ascii=False))

# 多字段组合查 chunk
results = api.search_chunks(doc_id="001", chunk_type="sheet_summary", limit=5)
for c in results:
    print(json.dumps(c, ensure_ascii=False))
