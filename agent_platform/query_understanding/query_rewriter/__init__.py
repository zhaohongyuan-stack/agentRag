"""查询改写器模块 — 指代消解 + 同义词扩展"""

from .reference_resolver import (
    ReferenceResolver,
    ResolutionResult,
    SessionContext,
)
from .rewriter import QueryRewriter, RewrittenQuery
from .synonym_dict import SynonymDict

__all__ = [
    "QueryRewriter",
    "RewrittenQuery",
    "SynonymDict",
    "ReferenceResolver",
    "ResolutionResult",
    "SessionContext",
]
