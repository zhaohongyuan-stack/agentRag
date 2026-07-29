# Retrieval API 接口文档

> 启动方式：`python -m retrieval_service.server`  
> 默认地址：`http://127.0.0.1:8000`  
> 交互式文档：启动后访问 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 接口一：POST `/api/v1/search`

统一检索入口，返回 `RetrievalHit` 列表。

**请求体：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | 是 | - | 检索查询文本 |
| `strategy` | string | 否 | `hybrid` | 检索策略：`hybrid` / `bm25` / `dense` / `exact` / `metadata` / `table` |
| `top_k` | int | 否 | `10` | 返回结果条数 |
| `filters` | object | 否 | `{}` | 元数据过滤条件 |
| `expand_context` | bool | 否 | `false` | 是否附带邻域上下文 |

**请求示例：**

```json
{
  "query": "核心一级资本",
  "strategy": "hybrid",
  "top_k": 5
}
```

**响应字段：**

每条命中包含 `chunk_id`、`chunk_type`、`doc_id`、`doc_name`、`citation`、`content`、`score`、`matched_by`、`scores_detail`、`metadata` 等。

---

## 接口二：GET `/api/v1/chunks/{chunk_id}`

按 `chunk_id` 查单条 chunk。

**路径参数：**

| 参数 | 说明 |
|------|------|
| `chunk_id` | chunk 的唯一标识，如 `001_s1_summary` |

**响应示例：**

```json
{
  "chunk_id": "001_s1_summary",
  "chunk_type": "sheet_summary",
  "doc_id": "001",
  "content": "...",
  "hierarchy_path": "...",
  "metadata": { ... }
}
```

---

## 接口三：GET `/api/v1/documents/{doc_id}`

按 `doc_id` 查文档元信息。

**路径参数：**

| 参数 | 说明 |
|------|------|
| `doc_id` | 文档编号，如 `001` |

**响应示例：**

```json
{
  "doc_id": "001",
  "doc_name": "001_2026年银行业总资产、总负债（月度）_...xls",
  "parser_type": "excel",
  "applicable_scope": "全部",
  "parse_timestamp": "2026-07-22T00:37:02.347863"
}
```

---

## 接口四：GET `/api/v1/documents`

列出文档。

**查询参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `limit` | int | `20` | 返回条数（1~200） |

---

## 接口五：POST `/api/v1/chunks/search`

多字段组合查 chunk，所有条件 AND 关联。

**请求体：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `doc_id` | string | 否 | 文档编号 |
| `chunk_type` | string | 否 | chunk 类型 |
| `table_name` | string | 否 | 表格编码名 |
| `clause_number` | string | 否 | 条款编号 |
| `chapter_number` | string | 否 | 章节编号 |
| `limit` | int | 否 | 返回条数（默认 20） |

**请求示例：**

```json
{
  "doc_id": "001",
  "chunk_type": "sheet_summary",
  "limit": 10
}
```
