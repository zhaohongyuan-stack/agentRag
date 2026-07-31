# 知识库数据目录

本目录存放检索服务所需的 chunk 数据文件（JSONL 格式）。

## 数据不随仓库分发

出于版权和体积考虑，`*.jsonl` 数据文件被 `.gitignore` 忽略，不会提交到仓库。拉取项目后此目录为空，需要自行获取或生成数据。

## 放置方式

将 `*_chunks.jsonl` 文件放入本目录：

```
regulatory_docs/
├── README.md                          ← 本说明文件
├── 001_银行业资产负债月度_chunks.jsonl   ← 自行放入
├── 002_资本管理办法_chunks.jsonl         ← 自行放入
└── ...
```

检索服务启动时会递归扫描本目录下所有 `*_chunks.jsonl` 和 `*_chunks.json` 文件。

## 数据获取途径

- **从数据源下载**：如有独立数据仓库或网盘链接，下载后放入本目录。
- **自行解析生成**：使用 `data_pipeline/` 下的解析器（docx / pdf / excel）将原始文档转换为标准 chunk JSONL。

## chunk 格式要求

每行一个 JSON 对象，核心字段：

| 字段 | 说明 |
|------|------|
| `chunk_id` | 全局唯一标识 |
| `chunk_type` | 片段类型（clause / table / table_row / glossary 等） |
| `content` | 检索用文本 |
| `hierarchy_path` | 层级路径 |
| `parent_chunk_id` | 父片段 ID（可为 null） |
| `metadata` | 元数据对象，需含 `doc_id`、`doc_name` 等 |

详细格式示例见项目根目录 `USAGE.md` 的"加入新的 Chunk"章节。

## 缓存说明

服务首次启动后会在本目录下生成 `.cache/` 子目录，存放构建好的索引（BM25、向量、FTS5、SQLite 等）。`.cache/` 已被 `.gitignore` 忽略，不会提交。修改或新增数据文件后，服务会自动检测文件指纹变化并重建索引。
