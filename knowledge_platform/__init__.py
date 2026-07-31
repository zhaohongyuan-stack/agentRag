"""
knowledge_platform — 法规知识库平台

分层架构：
  retrieval/   — 检索服务（7 路检索引擎 + API）
  indexes/     — 索引层（向量 / 词汇 / 元数据 / 关系 / 表格）
  ingestion/   — 入库层（去重 / 导入 / 向量化 / 索引构建 / 回滚）
  repositories — 数据仓库层（文档 / Chunk / 关系 / 版本）
  services/    — 领域服务层
  fusion/      — 融合层（去重 / 精排 / RRF）
  context/     — 上下文扩展层
"""
