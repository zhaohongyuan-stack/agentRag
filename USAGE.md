# ACE-RAG 使用指南

本指南面向刚拉取项目的开发者，覆盖从环境准备、服务启动、联调测试到扩充知识库的完整流程。

## 项目架构速览

ACE-RAG（Adaptive Compiled Evidence RAG）是一个监管知识库智能问答系统，将用户问题"编译"成可优化的检索执行计划，通过状态机驱动的 Agent 编排多通道检索、证据组装和声明级验证，输出带来源引用的回答。

系统分为两个平台，通过 HTTP 通信：

| 平台 | 目录 | 端口 | 职责 |
|------|------|------|------|
| A组 检索服务 | `knowledge_platform/retrieval/` | 8000 | 7 大检索器、RRF 融合、SQLite 存储 |
| B组 Agent | `agent_platform/` | 8002 | 查询理解、编译、路由、编排、证据、生成、验证 |

### 实际技术栈

> 注意：`README.md` 中的技术栈描述已过时，实际采用轻量化方案，不依赖 PostgreSQL / Elasticsearch / Milvus。

| 组件 | 选型 | 说明 |
|------|------|------|
| 关系数据库 | SQLite | 文档与 chunk 持久化，FTS5 全文检索 |
| 向量存储 | 内存 + `.npy` 文件 | 小规模场景无需外部向量库 |
| 嵌入服务 | SiliconFlow API | `BAAI/bge-large-zh-v1.5`，1024 维 |
| 重排序服务 | SiliconFlow API | `BAAI/bge-reranker-v2-m3` |
| LLM | DeepSeek API | 查询理解、回答生成 |
| 会话/缓存 | 内存（Redis 可选） | 未配置 Redis 时自动降级 |
| 分词 | jieba | BM25 关键词检索的中文分词 |

## 环境准备

### 1. Python 环境

项目要求 Python 3.11 及以上。建议使用虚拟环境隔离依赖：

```bash
# 创建虚拟环境
python -m venv .venv

# 激活（Windows PowerShell）
.venv\Scripts\Activate.ps1

# 激活（Linux / macOS）
source .venv/bin/activate
```

### 2. 安装依赖

```bash
# 安装核心依赖 + 开发工具 + LLM 客户端
pip install -e ".[dev,llm]"

# 如需 Redis 会话持久化（可选），追加 redis 额外依赖
pip install -e ".[dev,llm,redis]"
```

核心依赖包括：FastAPI、uvicorn、sentence-transformers、numpy、modelscope、jieba、pyyaml。开发依赖包括：pytest、ruff、mypy、httpx、jsonschema。

### 3. 配置环境变量

复制模板并填写实际的 API Key：

```bash
cp .env.example .env
```

`.env` 中需要配置的关键项：

| 变量 | 用途 | 是否必填 |
|------|------|----------|
| `LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL` | DeepSeek LLM 接口 | 必填（否则降级 Mock） |
| `USE_EMBED_API=true` | 启用 SiliconFlow 嵌入 API | 必填 |
| `SILICONFLOW_EMBED_API_KEY` | 嵌入 API Key | 必填 |
| `SILICONFLOW_EMBED_MODEL` | 嵌入模型名 | 必填 |
| `USE_RERANKER=true` | 启用重排序 | 推荐 |
| `SILICONFLOW_RERANK_API_KEY` | 重排序 API Key | 推荐填 |
| `SILICONFLOW_RERANK_MODEL` | 重排序模型名 | 推荐填 |
| `REDIS_*` | Redis 连接（可选） | 不填则内存模式 |

`.env` 文件已被 `.gitignore` 忽略，不会提交到仓库，API Key 是安全的。

### 4. 准备知识库数据

检索服务启动时需要加载数据。数据文件放置位置：

```
knowledge_platform/retrieval/regulatory_docs/
```

> 数据文件不随仓库分发。`.gitignore` 忽略了所有 `*.jsonl` 文件（数据可能有版权或体积较大），拉取仓库后该目录只有 `.gitkeep` 占位文件。数据需另行获取，详见下方"数据获取与放置"。

#### 数据获取与放置

数据以 JSONL 格式组织，每行一个 JSON 对象代表一个 chunk。获取方式有两种：

**方式一：从数据源下载**

如果项目有独立的数据仓库或网盘链接，将 `*_chunks.jsonl` 文件下载后放入上述目录。文件命名约定 `{编号}_{描述}_chunks.jsonl`，例如：

```
knowledge_platform/retrieval/regulatory_docs/
├── 001_银行业资产负债月度_chunks.jsonl
├── 002_资本管理办法_chunks.jsonl
└── 003_商业银行法_chunks.jsonl
```

**方式二：自行解析生成**

使用 `data_pipeline/` 下的解析器将原始文档（docx / pdf / excel）转换为标准 chunk JSONL。解析器产出符合项目 chunk 约定的 JSONL 文件后，放入同一目录即可。chunk 的核心字段格式见"加入新的 Chunk"章节。

目录下有 `regulatory_docs/README.md` 记录了数据放置规范，可供参考。

## 启动服务

### 一键启动（推荐）

```bash
python scripts/start_servers.py
```

该脚本会依次启动两个服务，等待健康检查通过后输出访问地址：

```
A组检索服务: http://127.0.0.1:8000/docs
B组Agent:   http://127.0.0.1:8002/docs
```

按 `Ctrl+C` 停止全部服务。脚本支持的参数：

```bash
python scripts/start_servers.py --retrieval   # 仅启动检索服务
python scripts/start_servers.py --agent       # 仅启动 Agent 服务
python scripts/start_servers.py --check       # 检查服务运行状态
```

### 单独启动

```bash
# A组检索服务（端口 8000）
cd knowledge_platform/retrieval
python -m retrieval_service.server

# B组 Agent 服务（端口 8002）
AGENT_PORT=8002 python -m agent_platform.server
```

> Windows 下设置端口需用 `$env:AGENT_PORT=8002`。

### 验证服务

两个服务都提供 `/health` 健康检查和 `/docs` 交互式 API 文档。首次启动检索服务时，如果缓存不存在，会全量构建索引（加载数据 → BM25 → 向量 → FTS5 → 元数据 → 关系 → 表格 → 重排序），构建完成后会持久化到 `.cache/` 目录，后续启动直接命中缓存（约 0.2 秒）。

## 联调测试

项目提供四个层次的测试，从单元测试到真实端到端联调。

### 1. 单元测试

```bash
# 运行全部测试（约 1100 个用例）
pytest

# 仅运行单元测试
pytest agent_platform/tests/unit knowledge_platform/tests/unit

# 运行指定测试文件
pytest agent_platform/tests/unit/test_chunk_store.py -v
```

测试配置在 `pyproject.toml` 中，`testpaths` 指向 `agent_platform/tests` 和 `knowledge_platform/tests`。单元测试使用 Mock，不需要真实服务或 API。

### 2. 工作流 / 集成测试

```bash
pytest agent_platform/tests/workflow
```

这些测试验证多阶段端到端流程（Phase2 到 Phase5），同样基于 Mock 数据。

### 3. 交互式联调（推荐）

交互式联调是验证整个系统最直观的方式。有两种模式：

**模式 A：真实联调（需要检索服务 + LLM）**

先启动检索服务，并确保 `.env` 中配置了 DeepSeek API Key：

```bash
# 步骤 1：启动检索服务（等待 "服务就绪" 提示）
python scripts/start_servers.py --retrieval

# 步骤 2：另开终端，进入交互式联调
python scripts/real_integration_test.py
```

进入交互模式后，输入问题即可获得完整的端到端响应，包括：

- 带引用来源的回答（DeepSeek LLM 生成）
- 链路追踪树（Span 树，显示每个阶段的耗时）
- 指标采集（查询延迟、检索调用数、证据充分性评分、预算消耗）
- 结构化执行日志（分层打印，如"进入路由层 XXX"）
- 状态轨迹（Agent 状态机流转过程）

交互命令：输入问题直接提问；输入 `simple` 切换简化模式（只显示回答）；输入 `full` 切换完整模式；输入 `quit` 退出。

```bash
# 单次提问（不进入交互）
python scripts/real_integration_test.py "银行业总资产是多少"

# 多轮对话模式（保持会话，测试指代消解和记忆）
python scripts/real_integration_test.py --multi

# 查看推荐测试问题
python scripts/real_integration_test.py --list
```

**模式 B：Mock 联调（无需启动任何服务）**

无需启动检索服务，也无需配置 API Key，使用内置 Mock 数据快速验证 Agent 全流程：

```bash
python scripts/interactive_test.py
python scripts/interactive_test.py "核心一级资本充足率最低要求是多少"
python scripts/interactive_test.py --multi   # 多轮对话
```

Mock 模式适合快速验证 Agent 的路由、编排、证据组装和回答生成逻辑是否正常，不依赖外部服务。

### 5. 检索服务接口验证

检索服务启动后，可直接调用其 API 验证检索能力：

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 统一检索
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "银行业总资产", "strategy": "hybrid", "top_k": 10}'

# 查看 chunk 详情
curl http://127.0.0.1:8000/api/v1/chunks/001_s1_table_5

# 列出文档
curl http://127.0.0.1:8000/api/v1/documents?limit=20
```

`strategy` 支持 `hybrid`（默认，BM25 + Dense + RRF 融合）、`lexical`、`dense`、`exact`、`metadata` 等。

### 6. 代码质量检查

```bash
make lint       # ruff 静态检查
make format     # ruff 格式化
make typecheck  # mypy 类型检查
```

## 加入新的 Chunk

知识库以 JSONL 文件组织，每行一个 JSON 对象代表一个 chunk。

### 1. 数据格式

每个 chunk 的核心字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunk_id` | string | 全局唯一标识，建议 `{doc_id}_{序号}_{类型}` |
| `chunk_type` | string | 片段类型，如 `clause`、`table`、`table_row`、`glossary` 等 |
| `content` | string | 检索用文本（已含上下文前插，直接用于 BM25 / 向量） |
| `hierarchy_path` | string | 层级路径，如 `文档名 / 章节 / 表格` |
| `parent_chunk_id` | string\|null | 父片段 ID，用于关系查询 |
| `metadata` | object | 元数据，含 `doc_id`、`doc_name`、`keywords` 等 |

一个最简的条款 chunk 示例：

```json
{"chunk_id": "002_c1", "chunk_type": "clause", "hierarchy_path": "资本管理办法 / 第一章 总则 / 第三条", "content": "资本管理办法第一章第三条：商业银行核心一级资本充足率不得低于5%。", "content_raw": "商业银行核心一级资本充足率不得低于5%。", "parent_chunk_id": null, "metadata": {"doc_id": "002", "doc_name": "资本管理办法", "parser_type": "docx", "applicable_scope": "全部", "normative_level": "mandatory", "keywords": ["核心一级资本充足率", "5%"], "chapter_number": "1", "clause_number": "3"}}
```

一个表格数据 chunk 示例（来自 Excel 解析）：

```json
{"chunk_id": "001_s1_row_7", "chunk_type": "table_row", "hierarchy_path": "银行业资产负债月度 / 1. 银行业金融机构 / 第7行", "content": "银行业资产负债月度 / 1. 银行业金融机构。第7行：总资产=4806061.6912。", "parent_chunk_id": "001_s1_table_5", "metadata": {"doc_id": "001", "doc_name": "001_银行业资产负债月度.xls", "parser_type": "excel", "table_name": "1. 银行业金融机构", "metric_name": "总资产", "unit": "亿元", "keywords": ["总资产"], "numeric_conditions": [{"field": "2026年_1月", "value": "4806061.6912"}]}}
```

### 2. 添加文件

将 `*_chunks.jsonl` 文件放入数据目录：

```
knowledge_platform/retrieval/regulatory_docs/
```

文件命名约定为 `{编号}_{描述}_chunks.jsonl`，如 `002_资本管理办法_chunks.jsonl`。加载器会递归扫描目录下所有 `*_chunks.jsonl` 和 `*_chunks.json` 文件。

### 3. 缓存自动失效机制

检索服务通过数据指纹判断缓存是否有效：对数据目录下所有 JSONL 文件计算 `文件名 + 文件大小 + 修改时间` 的哈希。只要新增、修改或删除任意 JSONL 文件，指纹就会变化，下次启动时自动触发全量重建索引。

因此加入新 chunk 后：

- **重启检索服务**即可，无需手动清理缓存。
- 如果想强制重建，删除 `regulatory_docs/.cache/` 目录后重启。

首次全量构建会调用 SiliconFlow 嵌入 API 为所有 chunk 生成向量（1024 维），并写入 SQLite、FTS5 索引、BM25 索引和 `.npy` 向量文件。构建完成后这些产物持久化到 `.cache/`，后续启动直接加载。

### 4. 验证新数据

```bash
# 重启检索服务后，检查文档数量是否增加
curl http://127.0.0.1:8000/api/v1/documents

# 用新 chunk 的关键词检索验证
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "新加入内容的关键词", "top_k": 5}'
```

## 提交到 GitHub 前的检查

### 已正确忽略的内容

`.gitignore` 已配置忽略以下内容，不会误提交：

- `.env`（含真实 API Key）
- `*.jsonl`（数据文件，另行获取）
- `**/.cache/`（检索索引缓存，含 `.pkl`、`.npy`、`retrieval.db`）
- `docx/`（原始设计文档）
- `__pycache__/`、`.venv/`、`.pytest_cache/` 等

### 数据文件策略

本项目采用"数据另行获取"策略：`*.jsonl` 数据文件不随仓库分发。拉取项目后，参考 `knowledge_platform/retrieval/regulatory_docs/README.md` 了解数据放置规范，将数据文件放入该目录后即可启动服务。

### 提交前自检清单

```bash
# 确认 .env 不会被提交
git status .env

# 确认缓存目录不会被提交
git status knowledge_platform/retrieval/regulatory_docs/.cache/

# 运行全量测试确保通过
pytest -q

# 检查代码风格
ruff check .
```

## 常见问题

**Q: 启动检索服务时报 `SILICONFLOW_EMBED_API_KEY` 未配置？**
A: 检查 `.env` 中 `USE_EMBED_API=true` 且填写了 `SILICONFLOW_EMBED_API_KEY`。检索服务启动时会从项目根目录的 `.env` 加载配置。

**Q: Agent 服务回答是 Mock 内容？**
A: 说明 `LLM_API_KEY` 未配置或 DeepSeek API 不可达，系统自动降级为 Mock 模式。检查 `.env` 中的 LLM 配置和网络连通性。

**Q: 首次启动很慢？**
A: 首次需要全量构建索引并调用嵌入 API 生成向量，耗时取决于 chunk 数量。构建完成后会缓存，后续启动约 0.2 秒。

**Q: 修改了 chunk 数据但检索结果没变？**
A: 数据指纹基于文件大小和修改时间，如果只改了内容未改大小可能不会触发重建。删除 `.cache/` 目录后重启即可强制重建。

**Q: 如何切换回本地嵌入模型（不使用 API）？**
A: 在 `.env` 中设置 `USE_EMBED_API=false`，首次运行会通过 ModelScope 下载 `BAAI/bge-small-zh-v1.5` 模型到本地缓存（需网络，建议开启代理）。
