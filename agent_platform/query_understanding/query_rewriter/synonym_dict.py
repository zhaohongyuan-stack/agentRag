"""
监管领域同义词词典

维护银行业监管术语及其同义词/缩写的双向映射关系。
支持双向查询：既可以从中文全称查到英文缩写，也可以从英文缩写查到中文全称。

示例:
  "资本充足率" → ["CAR", "Capital Adequacy Ratio"]
  "核心一级资本充足率" → ["CET1", "Core Tier 1 Capital Adequacy Ratio"]
  "系统重要性银行" → ["SIB", "Systemically Important Bank"]

用法:
    synonym_dict = SynonymDict()
    syns = synonym_dict.get_synonyms("资本充足率")  # ["CAR", "Capital Adequacy Ratio"]
    expanded = synonym_dict.expand_query("CAR最低多少")  # "CAR最低多少 资本充足率 Capital Adequacy Ratio"
"""

from typing import Dict, List, Set


class SynonymDict:
    """
    监管领域同义词词典

    维护术语与其同义词/缩写的双向映射，支持查询扩展。
    每个术语组以「主术语」（通常为中文全称）为键，同义词列表为值。
    内部构建反向索引，使得任意同义词均可反查到主术语及其全部别名。
    """

    # ============================================================
    # 默认监管术语同义词表
    # 主术语（中文全称） → [同义词/缩写列表]
    # ============================================================
    DEFAULT_SYNONYMS: Dict[str, List[str]] = {
        # ── 资本充足率系列 ──
        "资本充足率": ["CAR", "Capital Adequacy Ratio"],
        "核心一级资本充足率": ["CET1", "Core Tier 1 Capital Adequacy Ratio", "核心一级资本充足比率"],
        "一级资本充足率": ["T1CAR", "Tier 1 Capital Adequacy Ratio"],
        # ── 系统重要性银行 ──
        "系统重要性银行": ["SIB", "Systemically Important Bank"],
        "国内系统重要性银行": ["D-SIB", "Domestic Systemically Important Bank"],
        "全球系统重要性银行": ["G-SIB", "Global Systemically Important Bank", "G-SIFI"],
        # ── 杠杆与流动性指标 ──
        "杠杆率": ["Leverage Ratio"],
        "流动性覆盖率": ["LCR", "Liquidity Coverage Ratio"],
        "净稳定资金比例": ["NSFR", "Net Stable Funding Ratio"],
        "流动性比例": ["Liquidity Ratio"],
        "存贷比": ["LDR", "Loan-to-Deposit Ratio"],
        # ── 货币政策工具 ──
        "存款准备金率": ["RRR", "Reserve Requirement Ratio", "存准率"],
        # ── 资产质量指标 ──
        "不良贷款率": ["NPL Ratio", "NPL", "Non-Performing Loan Ratio"],
        "拨备覆盖率": ["PCR", "Provision Coverage Ratio"],
        "拨贷比": ["LLR", "Loan Loss Provision Ratio", "贷款拨备率"],
        # ── 盈利能力指标 ──
        "资产利润率": ["ROA", "Return on Assets"],
        "资本利润率": ["ROE", "Return on Equity"],
        # ── 资本构成 ──
        "核心一级资本": ["CET1 Capital", "Core Tier 1 Capital"],
        "一级资本": ["Tier 1 Capital", "T1 Capital"],
        "总资本": ["Total Capital"],
        "风险加权资产": ["RWA", "Risk-Weighted Assets"],
        # ── 资本缓冲 ──
        "储备资本": ["Capital Conservation Buffer", "CCB", "资本储备缓冲"],
        "逆周期资本": ["Countercyclical Capital Buffer", "CCyB", "逆周期资本缓冲"],
        "附加资本": ["Additional Capital", "Surcharge"],
        "系统重要性银行附加资本": ["SIB Surcharge"],
        # ── 贷款损失准备 ──
        "贷款损失准备": ["Loan Loss Provisions", "LLP"],
        # ── 法规与协议 ──
        "商业银行资本管理办法": ["资本新规", "Capital Rules"],
        "巴塞尔协议": ["Basel Accord", "Basel Framework"],
        "巴塞尔协议III": ["Basel III", "Basel 3"],
        # ── 监管机构 ──
        "金融监督管理总局": ["NFRA", "国家金融监督管理总局"],
        "银保监会": ["CBIRC"],
        "人民银行": ["PBOC", "中国人民银行", "央行"],
        # ── 风险管理 ──
        "压力测试": ["Stress Testing", "Stress Test"],
        "内部评级法": ["IRB", "Internal Ratings-Based Approach"],
        "标准法": ["Standardized Approach", "SA"],
    }

    def __init__(self, synonyms: Dict[str, List[str]] = None):
        """
        初始化同义词词典

        Args:
            synonyms: 自定义同义词表，为 None 时使用默认 DEFAULT_SYNONYMS
        """
        self._synonyms: Dict[str, List[str]] = (
            dict(synonyms) if synonyms else dict(self.DEFAULT_SYNONYMS)
        )
        # 反向索引: 任意术语（主术语或同义词） → 主术语
        self._reverse_index: Dict[str, str] = {}
        # 所有已知术语集合（主术语 + 同义词）
        self._all_terms: Set[str] = set()
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """重建反向索引和全术语集合"""
        self._reverse_index.clear()
        self._all_terms.clear()
        for main_term, syns in self._synonyms.items():
            self._all_terms.add(main_term)
            self._reverse_index[main_term] = main_term
            for syn in syns:
                self._all_terms.add(syn)
                self._reverse_index[syn] = main_term

    def get_synonyms(self, term: str) -> List[str]:
        """
        获取术语的所有同义词/别名

        返回该术语所属同义词组中除输入术语本身以外的所有别名（含主术语）。

        Args:
            term: 查询术语（可以是主术语或任意同义词）

        Returns:
            同义词列表，不包含输入术语本身。未知术语返回空列表。

        示例:
            >>> d = SynonymDict()
            >>> d.get_synonyms("CAR")
            ['资本充足率', 'Capital Adequacy Ratio']
            >>> d.get_synonyms("资本充足率")
            ['CAR', 'Capital Adequacy Ratio']
        """
        main_term = self._reverse_index.get(term)
        if main_term is None:
            return []
        all_aliases = [main_term] + self._synonyms.get(main_term, [])
        return [a for a in all_aliases if a != term]

    def expand_query(self, query: str) -> str:
        """
        扩展查询：在原始查询后追加已知术语的同义词

        扫描查询文本中包含的所有已知监管术语，将其同义词/缩写追加到查询末尾，
        以增强检索召回率。子术语重叠时保留最长匹配（如「核心一级资本充足率」
        包含「资本充足率」，仅扩展前者）。

        Args:
            query: 用户查询文本

        Returns:
            扩展后的查询文本。若无已知术语则原样返回。

        示例:
            >>> d = SynonymDict()
            >>> d.expand_query("CAR最低多少")
            'CAR最低多少 资本充足率 Capital Adequacy Ratio'
            >>> d.expand_query("第43条内容")
            '第43条内容'
        """
        if not query or not query.strip():
            return query or ""

        found_terms = self.find_terms(query)
        if not found_terms:
            return query

        # 收集需要追加的同义词
        additions: List[str] = []
        seen: Set[str] = set()

        # 将查询中已出现的术语标记为已见，避免重复追加
        for term in self._all_terms:
            if term in query:
                seen.add(term)

        for term in found_terms:
            for syn in self.get_synonyms(term):
                if syn not in seen:
                    additions.append(syn)
                    seen.add(syn)

        if not additions:
            return query

        return f"{query} {' '.join(additions)}"

    def find_terms(self, text: str) -> List[str]:
        """
        查找文本中包含的所有已知监管术语

        采用最长匹配策略：如果某术语是另一个更长已匹配术语的子串，
        则不单独列出（如「资本充足率」是「核心一级资本充足率」的子串）。

        Args:
            text: 待查找的文本

        Returns:
            匹配到的术语列表，按在文本中出现的顺序排列
        """
        if not text:
            return []

        found: List[str] = []
        for term in self._all_terms:
            if term in text:
                found.append(term)

        if not found:
            return []

        # 按长度降序排列，优先保留长术语
        found.sort(key=len, reverse=True)

        # 过滤掉作为已选术语子串的短术语
        result: List[str] = []
        for term in found:
            if not any(term in longer for longer in result):
                result.append(term)

        # 按在文本中的出现顺序排序
        result.sort(key=lambda t: text.find(t))
        return result

    def add_term(self, term: str, synonyms: List[str]) -> None:
        """
        添加或更新术语及其同义词

        Args:
            term: 主术语（通常为中文全称）
            synonyms: 同义词/缩写列表

        示例:
            >>> d = SynonymDict()
            >>> d.add_term("大额风险暴露", ["Large Exposure", "LE"])
            >>> d.get_synonyms("LE")
            ['大额风险暴露', 'Large Exposure']
        """
        self._synonyms[term] = list(synonyms)
        self._rebuild_index()

    @property
    def all_terms(self) -> Set[str]:
        """返回所有已知术语（主术语 + 同义词）的集合"""
        return set(self._all_terms)
