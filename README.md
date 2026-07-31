# ACE-RAG: Adaptive Compiled Evidence RAG

自适应编译式证据检索架构 — 监管知识库智能问答系统

## 项目概述

ACE-RAG 将用户问题"编译"成可优化、可缓存、可解释的检索执行计划，通过状态机驱动的 Agent 编排多通道检索、证据组装和声明级验证，最终输出带来源引用的回答。

## 分组架构

| 组别 | 模块 | 职责 |
|------|------|------|
| A组 | `data_pipeline/`, `knowledge_platform/` | 文件解析、索引构建、检索服务、上下文扩展 |
| B组 | `agent_platform/` | 查询理解、编译、路由、编排、证据、生成、验证 |
| 共享 | `contracts/`, `evaluation/`, `runtime/` | 对接契约、评测体系、运行基础设施 |

## 目录结构

```
ace-rag/
├── docs/                 # 架构文档、API文档、ADR、运维手册
├── contracts/            # 双方共同冻结的对接契约（Schema、枚举、样例）
├── data_pipeline/        # A组: 文件解析与标准化
├── knowledge_platform/   # A组: 存储、索引、检索服务
├── agent_platform/       # B组: Agent 核心（查询理解、编译、编排、证据、验证）
├── runtime/              # 共享: Redis、队列、缓存、可观测性
├── evaluation/           # 共享: 数据集、指标、实验、报告
├── services/             # API 服务层
├── deployment/           # 部署配置
└── scripts/              # 工具脚本
```

## 开发阶段

- **阶段0**: 契约冻结（QuerySpec、RetrievalRequest/Hit、EvidenceBundle）
- **阶段1**: 解析标准化 + 流程设计（Mock 驱动）
- **阶段2**: 检索基线（BM25 + Dense + 基础 Evidence）
- **阶段3**: 结构与表格检索（声明槽位覆盖）
- **阶段4**: 复杂 Agent（Query Compiler + DAG + 推测式检索）
- **阶段5**: 验证与生产化（多轮状态 + 故障恢复 + 可观测性）

## 技术栈

- Python 3.11+
- SQLite（FTS5 全文检索 + 关系存储，替代 PostgreSQL）
- 内存向量 + `.npy` 持久化（替代 Milvus，适合小规模知识库）
- SiliconFlow API（嵌入 + 重排序）
- DeepSeek API（LLM 回答生成）
- Redis（可选，未配置时自动降级为内存模式）

## 快速开始

详细的使用、联调和扩库说明见 [USAGE.md](USAGE.md)。

```bash
# 安装依赖
pip install -e ".[dev,llm]"

# 配置环境变量
cp .env.example .env   # 填写 SiliconFlow / DeepSeek API Key

# 一键启动 A组检索服务 + B组 Agent 服务
python scripts/start_servers.py

# 运行测试
make test

# 真实联调测试
python scripts/real_integration_test.py "银行业总资产是多少"
```

## 文档

- 架构设计: `docs/architecture/`
- API 文档: `docs/api/`
- 架构决策记录: `docs/adr/`
- 运维手册: `docs/runbooks/`
- 原始设计文档: `docx/rag架构/`
