"""
领域上下文感知消歧器

在歧义检测之前增加上下文消歧层，根据查询上下文自动判断多义术语的具体含义。
只有上下文确实无法消歧时才触发澄清。

核心逻辑：
  1. 领域检测 — 从查询中识别监管领域（insurance / banking / securities）
  2. 术语-领域消歧 — 为每个多义术语定义领域特定含义
  3. 自动消歧 — 领域明确 + 术语在该领域有映射 → 不标记歧义
  4. 保留兜底 — 领域不明确或术语无领域映射 → 保留原澄清逻辑
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class DomainContextResolver:
    """领域上下文感知消歧器"""

    # 领域检测信号表
    DOMAIN_SIGNALS: Dict[str, Dict[str, Any]] = {
        "insurance": {
            # 文档名/共现词信号（+2分/命中）：保险领域强信号
            "doc_patterns": [
                r"保险", r"意外险", r"人身险", r"财产险", r"寿险",
                r"健康险", r"赔付", r"保单", r"保费", r"再保险",
            ],
            # 共现词信号（+1分/命中）：保险领域辅助信号
            "co_terms": [
                "保险", "保费", "理赔", "赔付", "保单", "承保",
                "自留毛保费", "计提", "准备金", "意外险",
                "人身险", "财产险", "寿险", "健康险",
                "未到期责任准备金", "未决赔款准备金", "寿险责任准备金",
            ],
        },
        "banking": {
            "doc_patterns": [
                r"商业银行", r"资本管理", r"流动性", r"杠杆率",
                r"存款", r"贷款", r"银行", r"资本充足",
            ],
            "co_terms": [
                "银行", "存款", "贷款", "资本充足率", "杠杆率",
                "拨备覆盖率", "拨贷比", "不良贷款", "不良贷款率",
                "风险加权资产", "流动性覆盖率", "净稳定资金比例",
                "存贷比", "核心一级资本", "一级资本", "总资本",
                "储备资本", "逆周期资本", "附加资本",
                "系统重要性银行", "存款准备金率",
            ],
        },
        "securities": {
            "doc_patterns": [
                r"证券", r"股票", r"债券", r"基金", r"期货",
            ],
            "co_terms": [
                "证券", "股票", "债券", "基金", "期货",
                "交易", "清算", "结算", "交易所",
            ],
        },
    }

    # 术语-领域消歧映射表
    TERM_DOMAIN_MAP: Dict[str, Dict[str, Dict[str, str]]] = {
        "准备金": {
            "insurance": {
                "resolved_meaning": "保险准备金",
                "explanation": "在保险业务语境中指保险准备金（特别准备金、未到期责任准备金等）",
            },
            "banking": {
                "resolved_meaning": "存款准备金/贷款损失准备金",
                "explanation": "在银行业务语境中指存款准备金或贷款损失准备金",
            },
        },
        "资本": {
            "banking": {
                "resolved_meaning": "银行资本",
                "explanation": "在银行监管语境中指银行各级资本（核心一级/一级/总资本）",
            },
            "insurance": {
                "resolved_meaning": "保险实际资本",
                "explanation": "在保险监管语境中指保险偿付能力资本（实际资本/最低资本）",
            },
        },
        "拨备": {
            "banking": {
                "resolved_meaning": "贷款损失准备",
                "explanation": "在银行业务语境中指贷款损失准备（一般准备/专项准备/特种准备）",
            },
        },
        "流动性": {
            "banking": {
                "resolved_meaning": "银行流动性指标",
                "explanation": "在银行监管语境中指流动性覆盖率、流动性比例、净稳定资金比例等",
            },
        },
        "风险": {
            "banking": {
                "resolved_meaning": "银行风险",
                "explanation": "在银行监管语境中指信用风险、市场风险、操作风险等",
            },
        },
    }

    # 领域判定最低阈值：总分 >= 此值才认定领域明确
    DOMAIN_THRESHOLD = 2

    def resolve(
        self,
        term: str,
        query: str,
        entities: Optional[List[Any]] = None,
    ) -> Tuple[bool, str, str]:
        """
        尝试基于上下文消歧多义术语

        Args:
            term: 多义术语（如"准备金"）
            query: 用户原始查询
            entities: 已抽取的实体列表

        Returns:
            (resolved, meaning, explanation)
            - resolved=True 时 meaning 为消歧后含义
            - resolved=False 时上下文不足以消歧
        """
        domain = self._detect_domain(query, entities or [])

        if domain and term in self.TERM_DOMAIN_MAP:
            domain_map = self.TERM_DOMAIN_MAP[term]
            if domain in domain_map:
                entry = domain_map[domain]
                print(
                    f"  [DomainContextResolver] 术语 '{term}' 领域消歧 → "
                    f"domain={domain}, meaning='{entry['resolved_meaning']}'"
                )
                return True, entry["resolved_meaning"], entry["explanation"]

        print(
            f"  [DomainContextResolver] 术语 '{term}' 无法消歧 "
            f"(domain={domain or '未判定'})"
        )
        return False, "", ""

    def _detect_domain(
        self,
        query: str,
        entities: List[Any],
    ) -> Optional[str]:
        """
        从查询中识别监管领域

        基于文档名模式和共现词加权评分，取最高分领域。

        Args:
            query: 用户原始查询
            entities: 已抽取的实体列表

        Returns:
            领域标识符（"insurance"/"banking"/"securities"），或 None（无法判定）
        """
        scores: Dict[str, int] = {}

        for domain, signals in self.DOMAIN_SIGNALS.items():
            score = 0

            # 文档名/共现词模式匹配（+2分/命中）
            for pattern in signals["doc_patterns"]:
                if re.search(pattern, query):
                    score += 2

            # 共现词匹配（+1分/命中）
            for term in signals["co_terms"]:
                if term in query:
                    score += 1

            # 从实体值中追加匹配
            for entity in entities:
                entity_value = getattr(entity, "value", "") if hasattr(entity, "value") else ""
                if not entity_value:
                    if isinstance(entity, dict):
                        entity_value = entity.get("value", "")
                if not entity_value:
                    continue
                for term in signals["co_terms"]:
                    if term in entity_value:
                        score += 1

            scores[domain] = score

        # 取最高分领域
        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        if best_score >= self.DOMAIN_THRESHOLD:
            # 检查是否有并列最高分（多领域并列时保守返回 None）
            tied_domains = [d for d, s in scores.items() if s == best_score and s > 0]
            if len(tied_domains) > 1:
                print(
                    f"  [DomainContextResolver] 领域并列: {tied_domains} "
                    f"(scores={scores})，保守返回 None"
                )
                return None

            print(
                f"  [DomainContextResolver] 领域判定: {best_domain} "
                f"(scores={scores})"
            )
            return best_domain

        print(
            f"  [DomainContextResolver] 领域不明确 "
            f"(scores={scores}, threshold={self.DOMAIN_THRESHOLD})"
        )
        return None
