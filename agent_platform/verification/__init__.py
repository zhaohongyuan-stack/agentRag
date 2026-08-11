"""
声明级验证模块 — M5.1

对生成的回答进行声明级验证，确保每个声明都有证据支持且数值、版本、
引用、范围均正确。

验证体系:
  1. NumericValidator: 数值、日期、比例、单位校验
  2. VersionValidator: 版本有效性校验（查询时间点）
  3. CitationValidator: 引用存在性和引用内容校验
  4. ScopeValidator: 适用主体和范围校验
  5. AnswerValidator: 综合验证 — 声明-证据对齐

验证流程:
  AnswerValidator.validate_answer(answer, bundle)
    → 拆分回答为声明列表
    → 逐声明查找支持证据
    → 对每个声明执行数值/版本/引用/范围校验
    → 汇总结果，决定 retry / refuse / accept

核心导出:
  - NumericValidator / ValidationResult
  - VersionValidator
  - CitationValidator
  - ScopeValidator
  - AnswerValidator / AnswerValidation / ClaimValidation
"""

from .answer_validator.validator import (
    AnswerValidation,
    AnswerValidator,
    ClaimValidation,
)
from .citation_validator.validator import CitationValidator
from .numeric_validator.validator import NumericValidator, ValidationResult
from .scope_validator.validator import ScopeValidator
from .version_validator.validator import VersionValidator

__all__ = [
    "NumericValidator",
    "ValidationResult",
    "VersionValidator",
    "CitationValidator",
    "ScopeValidator",
    "AnswerValidator",
    "AnswerValidation",
    "ClaimValidation",
]
