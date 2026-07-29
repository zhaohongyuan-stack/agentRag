"""
检索调用示例
"""

from retrieval_service import Retriever

# 1. 加载数据（启动时一次）
r = Retriever("regulatory_docs/").load()

# 2. 检索 — 选一个通道，改 strategy 即可
results = r.search("核心一级资本合格标准", top_k=5)
# results = r.search("不良贷款率", strategy="bm25", top_k=10)      # BM25 关键词
# results = r.search("银行业总资产", strategy="dense", top_k=5)    # 语义向量
# results = r.search("第十二条", strategy="exact", exact_mode="exact")  # 精确匹配
# results = r.search("KM1", strategy="table", filters={"table_name": "KM1"})  # 表格
# results = r.search("", strategy="metadata", filters={"chunk_type": "clause"})  # 元数据过滤

# 3. 每条结果是 dict，字段: chunk_id, content, score, doc_name 等
for h in results:
    print(h["doc_name"], h["score"])
