# 检索系统实现文档

## 架构总览

```
RetrievalAPI (search_request)
    │
    ├─ Phase 0: MetadataRetriever  元数据前置过滤（SQL WHERE）
    ├─ Phase 1: 策略路由（7选1）
    │   ├─ BM25     LexicalRetriever   jieba分词 + 倒排索引
    │   ├─ DENSE    DenseRetriever     BGE向量 + 内积相似度
    │   ├─ HYBRID   BM25 + Dense → RRF融合 → 可选Cross-Encoder精排
    │   ├─ EXACT    ExactRetriever     FTS5全文/LIKE/正则
    │   ├─ METADATA MetadataRetriever  SQL WHERE多字段组合
    │   ├─ TABLE    TableRetriever     表格行列检索 + BM25/Dense补充
    │   └─ RELATION RelationRetriever  文档/Chunk关系查询
    ├─ Phase 2: _ensure_diversity()  table ↔ cell_fact 类型互补
    └─ Phase 3: NeighborhoodRetriever 邻域上下文扩展（父/子/前/后/同级）
```

## 数据层

**ChunkStore** — 三层缓存，检索器不再持有 content 全文：

| 层 | 内容 | 命中方式 |
|----|------|---------|
| `_meta_map`（常驻） | chunk_id → 元信息（不含 content） | O(1) 纯内存 |
| `_lru`（500条） | chunk_id → content | 热点自动缓存 |
| SQLite DB（回源） | chunks.content 列 | 兜底查询 |

**RetrievalDB** — SQLite 双表（`documents` + `chunks`），12 个索引覆盖 parent/prev/next 链、chunk_type、table_name 等。FTS5 全文索引通过 `_tokenize_cjk()` 在 CJK 字符间插空格实现按字分词。

## 各检索器核心技术

### LexicalRetriever（BM25）
- **分词**：jieba + 40+ 金融领域自定义词典（"核心一级资本"等），回退逐字分词
- **索引**：倒排索引 `{term: {doc_idx: tf}}`，查询时只算包含 query term 的文档
- **公式**：标准 BM25（`k1=1.5, b=0.75`）

### DenseRetriever（向量语义）
- **双模式**：本地 `bge-small-zh-v1.5` (512维) / API `bge-large-zh-v1.5` (1024维)
- **API 模式**：硅基流动 Embedding API，自动重试、分批（32条/批）、空串替换、超长截断
- **相似度**：向量矩阵 @ query向量 → 余弦相似度（内积，向量已归一化）

### ExactRetriever（精确匹配）
- 四种模式：`exact`(TRIM比较) / `contains`(FTS5 MATCH→LIKE降级) / `regex`(LIKE粗筛+Python精排) / `prefix`(逐行startswith)
- FTS5 的 `unicode61` 默认不按字分词 → `_tokenize_cjk()` 在中文字符间插空格解决

### MetadataRetriever（元数据过滤）
- **DB模式**（生产）：SQL WHERE 驱动，支持 eq/in/contains/regex/prefix/suffix/gt/lt 7 种操作符
- **降级模式**（测试）：内存展平过滤
- 正则走 LIKE 粗筛 + Python `re` 精排两步

### TableRetriever（表格检索）
- 解析 Markdown 管道符表格（`| col |` 格式），建立 `{表名: {headers, rows}}` 索引
- 支持按行列取值、按关键词模糊搜行（全行/指定列）、全表转字典列表喂LLM

### NeighborhoodRetriever（邻域图）
- 基于 SQLite 三条链：`parent_chunk_id` / `prev_chunk_id` / `next_chunk_id`
- `get_context()` 一次返回 chunk + parent + children + siblings + prev + next + doc

## RRF 融合（HYBRID 策略）

```
RRF_score(d) = Σ 1/(k + rank_i(d))    k=60
```

BM25 取 `bm25_k` 条 + Dense 取 `vector_k` 条 → RRF 融合排序 → 可选 Cross-Encoder 精排（硅基流动 `bge-reranker-v2-m3` API）→ 取 `top_k`。

## TABLE 策略特有逻辑

1. **表名解析**（四层降级）：精确 → 标准化（全角→半角括号、月份去前导零）→ 双向子串 → 最长公共子串+年份冲突检查
2. **双路补充**：find_rows() 失败/不足时，BM25+Dense RRF 融合补充 narrative chunk
3. **Dense 的必要性**：纯 BM25 因文档长度惩罚（`b=0.75`）会降低 "2023年_一季度" 类 cell_fact 得分，Dense 不受长度影响
4. **同级展开**：命中非B列的 cell_fact 时，自动补入同行的B列（一季度）数据。新版按 `parent_chunk_id` 缓存 siblings 并批量预取 content

## 缓存系统

**指纹机制**：JSONL 文件名+大小+修改时间 → SHA256[:16] → `manifest_{hash}.json`。JSONL 缺失时从已有 manifest 恢复。

```
.cache/
├── manifest_{hash}.json
├── chunk_store_meta_{hash}.pkl   # 元信息（~5MB）
├── bm25_{hash}.pkl               # 倒排索引
├── dense_meta_{text_hash}.pkl    # chunk_ids + API配置
├── embeddings_{model}_{text_hash}.npy  # 向量矩阵（大）
├── table/metadata/exact_{hash}.pkl
└── retrieval.db                  # SQLite 持久化
```

`CACHE_VERSION` 递增时自动清理旧缓存，下次全量重建。

## 轻量模式（disk版新增）

`load(source, lightweight=True)` 跳过 Dense 向量，内存和启动时间大幅减少。适用只需 BM25 + 精确 + 结构化的场景。

## 类型互补 `_ensure_diversity()`（disk版新增）

确保结果中 table 和 cell_fact 兼有：
- 缺 cell_fact → 从邻域图递归拉子节点（深度≤2）
- 缺 table → 从邻域图拉父 table 置顶

## LLM 格式化

`format_for_llm(results, max_chars=3000)` → 每条编号 `[1][2]...`，附 emoji 图标(cell_fact→📈, table→📊, clause→📜)，标注出处和适用机构范围。LLM 可引用编号作答。

## 模块清单

| 模块 | 职责 |
|------|------|
| `retrieval_api.py` | 统一入口，策略路由，RRF融合 |
| `chunk.py` | Chunk数据结构，JSONL加载，metadata展平 |
| `chunk_store.py` | 分层缓存（元信息常驻 + LRU content + DB回源） |
| `retrieval_db.py` | SQLite持久化，FTS5全文索引，关系查询，元数据过滤 |
| `retrieval_request.py` | RetrievalRequest/Hit 强类型数据结构 |
| `lexical_retriever.py` | BM25 + jieba + 倒排索引 |
| `dense_retriever.py` | BGE向量（本地/API双模式） |
| `exact_retriever.py` | FTS5精确/子串/正则匹配 |
| `metadata_retriever.py` | SQL WHERE元数据过滤（DB/降级双模式） |
| `relation_retriever.py` | 文档/Chunk关系查询 |
| `neighborhood_retriever.py` | 邻域图查询（父子/前后/同级） |
| `table_retriever.py` | Markdown表格解析 + 行列检索 |
| `siliconflow_client.py` | 硅基流动API（Embedding + Rerank），自动重试 |
