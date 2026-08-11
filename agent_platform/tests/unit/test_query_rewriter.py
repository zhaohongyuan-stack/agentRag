"""
查询改写器单元测试

测试用例:
  - 同义词扩展: "CAR最低多少" → 扩展后包含 "资本充足率"
  - 指代消解: "这个比例适用吗" + 上下文提到 "核心一级资本充足率" → 消解成功
  - 无需改写: "第43条内容" → 原样返回
  - 歧义指代: "那个文件" + 上下文有两个文档 → 标记歧义
  - 通道查询生成: lexical / dense / exact 三个通道均存在
  - 空查询处理: 不崩溃，返回空结果
"""

import pytest

from agent_platform.query_understanding.query_rewriter import (
    QueryRewriter,
    RewrittenQuery,
    ReferenceResolver,
    ResolutionResult,
    SessionContext,
    SynonymDict,
)


class TestSynonymDict:
    """同义词词典测试"""

    def setup_method(self):
        self.dict = SynonymDict()

    def test_get_synonyms_chinese(self):
        """中文术语 → 英文缩写"""
        syns = self.dict.get_synonyms("资本充足率")
        assert "CAR" in syns
        assert "Capital Adequacy Ratio" in syns
        # 不应包含自身
        assert "资本充足率" not in syns

    def test_get_synonyms_english(self):
        """英文缩写 → 中文术语（双向查询）"""
        syns = self.dict.get_synonyms("CAR")
        assert "资本充足率" in syns
        assert "Capital Adequacy Ratio" in syns
        assert "CAR" not in syns

    def test_get_synonyms_cet1(self):
        """核心一级资本充足率"""
        syns = self.dict.get_synonyms("核心一级资本充足率")
        assert "CET1" in syns

    def test_get_synonyms_unknown(self):
        """未知术语返回空列表"""
        syns = self.dict.get_synonyms("不存在的术语XYZ")
        assert syns == []

    def test_expand_query_with_abbreviation(self):
        """同义词扩展: "CAR最低多少" → 包含 "资本充足率" """
        expanded = self.dict.expand_query("CAR最低多少")
        assert "资本充足率" in expanded
        assert "CAR" in expanded  # 原文保留
        assert "Capital Adequacy Ratio" in expanded

    def test_expand_query_with_chinese(self):
        """中文术语扩展: "资本充足率最低多少" → 包含 "CAR" """
        expanded = self.dict.expand_query("资本充足率最低多少")
        assert "CAR" in expanded
        assert "Capital Adequacy Ratio" in expanded

    def test_expand_query_no_synonyms(self):
        """无已知术语时不扩展"""
        expanded = self.dict.expand_query("第43条内容")
        assert expanded == "第43条内容"

    def test_expand_query_longest_match(self):
        """最长匹配: "核心一级资本充足率" 不重复扩展 "资本充足率" """
        expanded = self.dict.expand_query("核心一级资本充足率最低多少")
        assert "CET1" in expanded
        assert "核心一级资本充足率" in expanded
        # "资本充足率" 的同义词不应单独追加（已被更长术语覆盖）
        # 但 "CAR" 不应出现，因为 "资本充足率" 是子串被跳过
        assert "CAR" not in expanded

    def test_expand_query_empty(self):
        """空查询"""
        assert self.dict.expand_query("") == ""
        assert self.dict.expand_query(None) is None or self.dict.expand_query("") == ""

    def test_add_term(self):
        """添加新术语"""
        self.dict.add_term("大额风险暴露", ["Large Exposure", "LE"])
        syns = self.dict.get_synonyms("LE")
        assert "大额风险暴露" in syns
        assert "Large Exposure" in syns

    def test_find_terms(self):
        """查找文本中的已知术语"""
        terms = self.dict.find_terms("CAR和杠杆率的关系")
        assert "CAR" in terms
        assert "杠杆率" in terms


class TestReferenceResolver:
    """指代消解器测试"""

    def setup_method(self):
        self.resolver = ReferenceResolver()

    def test_resolve_metric_reference(self):
        """指代消解: "这个比例" → 具体指标"""
        ctx = SessionContext(mentioned_metrics=["核心一级资本充足率"])
        result = self.resolver.resolve_detailed("这个比例适用吗", ctx)
        assert result.was_resolved is True
        assert result.ambiguity_flagged is False
        assert "核心一级资本充足率" in result.resolved_query
        assert result.resolved_entity == "核心一级资本充足率"

    def test_resolve_metric_reference_simple(self):
        """resolve 方法返回字符串"""
        ctx = SessionContext(mentioned_metrics=["核心一级资本充足率"])
        resolved = self.resolver.resolve("这个比例适用吗", ctx)
        assert "核心一级资本充足率" in resolved
        assert "这个比例" not in resolved

    def test_resolve_doc_reference(self):
        """文档指代: "那个文件" → 具体文档"""
        ctx = SessionContext(mentioned_docs=["商业银行资本管理办法"])
        result = self.resolver.resolve_detailed("那个文件讲了什么", ctx)
        assert result.was_resolved is True
        assert "商业银行资本管理办法" in result.resolved_query

    def test_resolve_topic_reference(self):
        """主题指代: "之前提到的" → 具体实体"""
        ctx = SessionContext(
            mentioned_metrics=["杠杆率"],
            mentioned_docs=["商业银行资本管理办法"],
        )
        result = self.resolver.resolve_detailed("之前提到的内容是什么", ctx)
        # 主题指代有多个候选（指标+文档），应标记歧义
        assert result.ambiguity_flagged is True

    def test_resolve_topic_single_candidate(self):
        """主题指代: 只有一个候选时消解成功"""
        ctx = SessionContext(mentioned_metrics=["杠杆率"])
        result = self.resolver.resolve_detailed("之前提到的内容是什么", ctx)
        assert result.was_resolved is True
        assert "杠杆率" in result.resolved_query

    def test_resolve_ambiguous_docs(self):
        """歧义指代: 两个文档 → 标记歧义"""
        ctx = SessionContext(
            mentioned_docs=["商业银行资本管理办法", "商业银行流动性风险管理办法"]
        )
        result = self.resolver.resolve_detailed("那个文件讲了什么", ctx)
        assert result.ambiguity_flagged is True
        assert result.was_resolved is False
        assert "商业银行资本管理办法" in result.ambiguity_reason
        assert "商业银行流动性风险管理办法" in result.ambiguity_reason

    def test_resolve_ambiguous_metrics(self):
        """歧义指代: 两个指标 → 标记歧义"""
        ctx = SessionContext(
            mentioned_metrics=["核心一级资本充足率", "杠杆率"]
        )
        result = self.resolver.resolve_detailed("这个比例是多少", ctx)
        assert result.ambiguity_flagged is True

    def test_resolve_no_candidate(self):
        """无候选实体 → 标记歧义"""
        ctx = SessionContext(mentioned_metrics=[])
        result = self.resolver.resolve_detailed("这个比例是多少", ctx)
        assert result.ambiguity_flagged is True
        assert result.was_resolved is False

    def test_resolve_no_reference(self):
        """无指代词 → 原样返回"""
        ctx = SessionContext(mentioned_metrics=["核心一级资本充足率"])
        result = self.resolver.resolve_detailed("核心一级资本充足率最低多少", ctx)
        assert result.was_resolved is False
        assert result.ambiguity_flagged is False
        assert result.resolved_query == "核心一级资本充足率最低多少"

    def test_resolve_clause_reference(self):
        """条款指代: "那条" → 具体条款号"""
        ctx = SessionContext(
            previous_queries=["《商业银行资本管理办法》第43条的内容是什么"]
        )
        result = self.resolver.resolve_detailed("那条是怎么规定的", ctx)
        assert result.was_resolved is True
        assert "第43条" in result.resolved_query

    def test_resolve_with_entities(self):
        """从 previous_entities 中提取候选"""
        ctx = SessionContext(
            previous_entities=[
                {"entity_type": "metric_name", "value": "流动性覆盖率"},
            ]
        )
        result = self.resolver.resolve_detailed("这个指标是多少", ctx)
        assert result.was_resolved is True
        assert "流动性覆盖率" in result.resolved_query

    def test_resolve_empty_query(self):
        """空查询"""
        ctx = SessionContext(mentioned_metrics=["资本充足率"])
        result = self.resolver.resolve_detailed("", ctx)
        assert result.resolved_query == ""
        assert result.was_resolved is False

    def test_resolve_none_context(self):
        """无上下文"""
        result = self.resolver.resolve_detailed("这个比例是多少", None)
        assert result.was_resolved is False
        assert result.ambiguity_flagged is False

    def test_session_context_to_dict(self):
        """SessionContext 序列化"""
        ctx = SessionContext(
            previous_queries=["之前的查询"],
            mentioned_metrics=["资本充足率"],
            mentioned_docs=["商业银行资本管理办法"],
        )
        d = ctx.to_dict()
        assert d["previous_queries"] == ["之前的查询"]
        assert d["mentioned_metrics"] == ["资本充足率"]
        assert d["mentioned_docs"] == ["商业银行资本管理办法"]

    def test_session_context_from_dict(self):
        """SessionContext 反序列化"""
        d = {
            "previous_queries": ["q1"],
            "mentioned_metrics": ["m1"],
            "mentioned_docs": ["d1"],
        }
        ctx = SessionContext.from_dict(d)
        assert ctx.previous_queries == ["q1"]
        assert ctx.mentioned_metrics == ["m1"]
        assert ctx.mentioned_docs == ["d1"]


class TestQueryRewriter:
    """查询改写器测试"""

    def setup_method(self):
        self.rewriter = QueryRewriter()

    def test_synonym_expansion(self):
        """同义词扩展: "CAR最低多少" → 扩展后包含 "资本充足率" """
        result = self.rewriter.rewrite("CAR最低多少")
        assert "资本充足率" in result.channel_queries["dense"]
        assert result.ambiguity_flagged is False

    def test_coreference_resolution(self):
        """指代消解: "这个比例适用吗" + 上下文 → 消解成功"""
        ctx = SessionContext(mentioned_metrics=["核心一级资本充足率"])
        result = self.rewriter.rewrite("这个比例适用吗", session_context=ctx)
        assert "核心一级资本充足率" in result.contextualized_query
        assert "这个比例" not in result.contextualized_query
        assert result.ambiguity_flagged is False

    def test_no_rewrite_needed(self):
        """无需改写: "第43条内容" → 原样返回"""
        result = self.rewriter.rewrite("第43条内容")
        assert result.contextualized_query == "第43条内容"
        assert result.original_query == "第43条内容"
        assert result.ambiguity_flagged is False
        # 原始查询应在 rewrites 列表中
        assert "第43条内容" in result.rewrites

    def test_ambiguous_reference(self):
        """歧义指代: "那个文件" + 两个文档 → 标记歧义"""
        ctx = SessionContext(
            mentioned_docs=["商业银行资本管理办法", "商业银行流动性风险管理办法"]
        )
        result = self.rewriter.rewrite("那个文件讲了什么", session_context=ctx)
        assert result.ambiguity_flagged is True
        assert result.ambiguity_reason != ""
        # 歧义时 contextualized_query 应保持原始
        assert "那个文件" in result.contextualized_query

    def test_channel_queries_generation(self):
        """通道查询生成: 所有通道均存在"""
        result = self.rewriter.rewrite("资本充足率最低多少")
        assert "lexical" in result.channel_queries
        assert "dense" in result.channel_queries
        assert "exact" in result.channel_queries
        # lexical 通道应包含原始查询内容
        assert "资本充足率" in result.channel_queries["lexical"]
        # dense 通道应包含同义词扩展
        assert "CAR" in result.channel_queries["dense"]
        # exact 通道应包含关键术语
        assert "资本充足率" in result.channel_queries["exact"]

    def test_empty_query(self):
        """空查询处理"""
        result = self.rewriter.rewrite("")
        assert result.original_query == ""
        assert result.contextualized_query == ""
        assert result.channel_queries == {}
        assert result.rewrites == []
        assert result.ambiguity_flagged is False

    def test_none_query(self):
        """None 查询处理"""
        result = self.rewriter.rewrite(None)
        assert result.original_query == ""
        assert result.contextualized_query == ""

    def test_whitespace_query(self):
        """空白查询处理"""
        result = self.rewriter.rewrite("   ")
        assert result.original_query == "   "
        assert result.contextualized_query == "   "
        assert result.channel_queries == {}

    def test_to_dict(self):
        """to_dict 序列化"""
        result = self.rewriter.rewrite("资本充足率最低多少")
        d = result.to_dict()
        assert "original_query" in d
        assert "contextualized_query" in d
        assert "channel_queries" in d
        assert "rewrites" in d
        assert "ambiguity_flagged" in d
        assert "ambiguity_reason" in d
        assert isinstance(d["channel_queries"], dict)
        assert isinstance(d["rewrites"], list)

    def test_rewrites_contain_original(self):
        """rewrites 列表始终包含原始查询"""
        result = self.rewriter.rewrite("核心一级资本充足率最低多少")
        assert "核心一级资本充足率最低多少" in result.rewrites

    def test_rewrites_contain_expanded(self):
        """rewrites 列表包含扩展版本"""
        result = self.rewriter.rewrite("资本充足率最低多少")
        # 扩展版应包含 "CAR"
        expanded_versions = [r for r in result.rewrites if "CAR" in r]
        assert len(expanded_versions) > 0

    def test_exact_channel_extracts_key_terms(self):
        """exact 通道提取关键术语"""
        result = self.rewriter.rewrite("《商业银行资本管理办法》第43条")
        exact = result.channel_queries["exact"]
        assert "第43条" in exact
        assert "《商业银行资本管理办法》" in exact

    def test_with_query_spec(self):
        """带 query_spec 的改写"""
        query_spec = {
            "intent": "threshold",
            "entities": [
                {"entity_type": "metric_name", "value": "杠杆率", "confidence": 0.9},
            ],
        }
        result = self.rewriter.rewrite("杠杆率最低多少", query_spec=query_spec)
        assert "杠杆率" in result.channel_queries["exact"]
        assert "Leverage Ratio" in result.channel_queries["dense"]

    def test_no_session_context(self):
        """无会话上下文时不进行指代消解"""
        result = self.rewriter.rewrite("这个比例是多少")
        # 无上下文时，指代词保持原样
        assert result.contextualized_query == "这个比例是多少"
        assert result.ambiguity_flagged is False

    def test_mock_mode(self):
        """Mock 模式下使用规则消解"""
        # 默认 LLM 客户端应为 Mock 模式（无 API Key）
        assert self.rewriter._llm_client.is_mock is True
        ctx = SessionContext(mentioned_metrics=["资本充足率"])
        result = self.rewriter.rewrite("这个比例是多少", session_context=ctx)
        assert "资本充足率" in result.contextualized_query
        assert result.ambiguity_flagged is False

    def test_custom_synonym_dict(self):
        """自定义同义词词典"""
        custom_dict = SynonymDict({
            "自定义指标": ["Custom Metric", "CM"],
        })
        rewriter = QueryRewriter(synonym_dict=custom_dict)
        result = rewriter.rewrite("自定义指标是多少")
        assert "CM" in result.channel_queries["dense"]
        assert "Custom Metric" in result.channel_queries["dense"]


class TestResolutionResult:
    """指代消解结果测试"""

    def test_to_dict(self):
        """ResolutionResult 序列化"""
        result = ResolutionResult(
            resolved_query="核心一级资本充足率适用吗",
            was_resolved=True,
            ambiguity_flagged=False,
            ambiguity_reason="",
            resolved_entity="核心一级资本充足率",
            reference_type="metric",
        )
        d = result.to_dict()
        assert d["resolved_query"] == "核心一级资本充足率适用吗"
        assert d["was_resolved"] is True
        assert d["ambiguity_flagged"] is False
        assert d["resolved_entity"] == "核心一级资本充足率"
        assert d["reference_type"] == "metric"
