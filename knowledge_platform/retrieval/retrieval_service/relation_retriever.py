"""
Relation 检索器 — 基于关系数据库的文档与 Chunk 结构化查询

职责：
  文档级：按 ID 查文档、列出全部文档、按名称查版本、获取最新版本
  Chunk 级：按 ID 查 chunk、多字段组合过滤查询、按文档取全部 chunk ID

依赖：
  retrieval_db.RetrievalDB（SQLite 存储层）

使用方式：
    from retrieval_db import RetrievalDB
    from relation_retriever import RelationRetriever

    db = RetrievalDB("retrieval.db").open()
    retriever = RelationRetriever(db)

    doc = retriever.get_document("400")
    versions = retriever.get_document_versions("资本管理办法")
    chunks = retriever.search_chunks(chunk_type="clause", clause_number="12")
"""

from typing import List, Optional, Dict, Any

from .retrieval_db import RetrievalDB


class RelationRetriever:
    """
    关系检索器 — 文档与 Chunk 的结构化查询。

    所有方法都委托给 RetrievalDB，本类是语义词义层面的封装。
    """

    def __init__(self, db: RetrievalDB):
        self._db = db

    # ============================================================
    # 文档操作
    # ============================================================
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """按 doc_id 获取文档摘要信息"""
        return self._db.get_document(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有已加载的文档"""
        return self._db.list_documents()

    def get_document_versions(self, doc_name: str) -> List[Dict[str, Any]]:
        """
        获取同一文档名的所有解析版本（按时间降序）。

        用途：法规文件更新后可能有多次解析，此方法返回所有版本。
        """
        return self._db.get_document_versions(doc_name)

    def get_latest_version(self, doc_name: str) -> Optional[Dict[str, Any]]:
        """获取文档的最新解析版本"""
        return self._db.get_latest_version(doc_name)

    # ============================================================
    # Chunk 操作
    # ============================================================
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """按 chunk_id 获取单条 chunk 完整信息"""
        return self._db.get_chunk(chunk_id)

    def get_chunk_ids_by_doc(self, doc_id: str) -> List[str]:
        """获取某文档下所有 chunk 的 ID 列表"""
        return self._db.get_chunk_ids_by_doc(doc_id)

    def count_chunks(self, doc_id: Optional[str] = None) -> int:
        """统计 chunk 总数，可按文档过滤"""
        return self._db.count_chunks(doc_id)

    # ============================================================
    # 多字段过滤查询
    # ============================================================
    def search_chunks(self,
                      doc_id: Optional[str] = None,
                      chunk_type: Optional[str] = None,
                      table_name: Optional[str] = None,
                      clause_number: Optional[str] = None,
                      chapter_number: Optional[str] = None,
                      glossary_term: Optional[str] = None,
                      applicable_scope: Optional[str] = None,
                      normative_level: Optional[str] = None,
                      parent_chunk_id: Optional[str] = None,
                      limit: int = 100,
                      offset: int = 0) -> List[Dict[str, Any]]:
        """
        多字段 AND 组合过滤查询。

        所有参数可选，未传表示不过滤该字段。
        glossary_term 使用 LIKE %term% 模糊匹配，其余使用 = 等值匹配。

        示例：
          search_chunks(doc_id="400", chunk_type="clause",
                        clause_number="12", applicable_scope="全部", limit=50)
        """
        return self._db.search_chunks(
            doc_id=doc_id,
            chunk_type=chunk_type,
            table_name=table_name,
            clause_number=clause_number,
            chapter_number=chapter_number,
            glossary_term=glossary_term,
            applicable_scope=applicable_scope,
            normative_level=normative_level,
            parent_chunk_id=parent_chunk_id,
            limit=limit,
            offset=offset,
        )
