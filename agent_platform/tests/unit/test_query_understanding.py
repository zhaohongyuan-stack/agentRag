"""
查询理解单元测试

测试用例:
  - 条款查询: "《商业银行资本管理办法》第43条" → clause_query
  - 阈值查询: "核心一级资本充足率最低要求是多少" → threshold
  - 定义查询: "什么是系统重要性银行" → definition
  - 表格取数: "附件1表中非系统重要性银行的比例" → table_lookup
  - 歧义检测: "那个比例是多少" → ambiguity_flag=true
  - 问候: "你好" → greeting
"""

import pytest

from agent_platform.query_understanding import (
    AmbiguityDetector,
    ConstraintExtractor,
    EntityExtractor,
    IntentClassifier,
    QuerySpec,
    QuerySpecBuilder,
)


class TestIntentClassifier:
    """意图分类器测试"""

    def setup_method(self):
        self.classifier = IntentClassifier()

    def test_clause_query(self):
        """条款查询意图"""
        result = self.classifier.classify("《商业银行资本管理办法》第43条")
        assert result.intent == "clause_query"
        assert result.confidence > 0.8

    def test_clause_query_chinese_number(self):
        """中文数字条款号"""
        result = self.classifier.classify("第四十三条内容是什么")
        assert result.intent == "clause_query"

    def test_threshold_query(self):
        """阈值查询意图"""
        result = self.classifier.classify("核心一级资本充足率最低要求是多少")
        assert result.intent == "threshold"
        assert result.confidence > 0.7

    def test_definition_query(self):
        """定义查询意图"""
        result = self.classifier.classify("什么是系统重要性银行")
        assert result.intent == "definition"
        assert result.confidence > 0.8

    def test_table_lookup_query(self):
        """表格取数意图"""
        result = self.classifier.classify("附件1表中非系统重要性银行的比例")
        assert result.intent == "table_lookup"

    def test_greeting(self):
        """问候意图"""
        result = self.classifier.classify("你好")
        assert result.intent == "greeting"
        assert result.confidence > 0.9

    def test_comparison(self):
        """比较查询意图"""
        result = self.classifier.classify("比较核心一级资本充足率和一级资本充足率的区别")
        assert result.intent == "comparison"

    def test_compliance(self):
        """合规查询意图"""
        result = self.classifier.classify("银行是否符合资本充足率要求")
        assert result.intent == "compliance"

    def test_unknown(self):
        """未知意图"""
        result = self.classifier.classify("今天天气怎么样")
        assert result.intent == "unknown"

    def test_empty_query(self):
        """空问题"""
        result = self.classifier.classify("")
        assert result.intent == "unknown"
        assert result.confidence == 0.0


class TestEntityExtractor:
    """实体抽取器测试"""

    def setup_method(self):
        self.extractor = EntityExtractor()

    def test_extract_doc_name(self):
        """抽取文档名"""
        entities = self.extractor.extract("《商业银行资本管理办法》第43条")
        doc_names = [e for e in entities if e.entity_type == "doc_name"]
        assert len(doc_names) == 1
        assert doc_names[0].value == "商业银行资本管理办法"

    def test_extract_clause_number(self):
        """抽取条款号"""
        entities = self.extractor.extract("第43条内容")
        clauses = [e for e in entities if e.entity_type == "clause_number"]
        assert len(clauses) == 1
        assert clauses[0].value == "43"

    def test_extract_clause_number_chinese(self):
        """中文数字条款号"""
        entities = self.extractor.extract("第四十三条")
        clauses = [e for e in entities if e.entity_type == "clause_number"]
        assert len(clauses) == 1

    def test_extract_metric_name(self):
        """抽取指标名"""
        entities = self.extractor.extract("核心一级资本充足率最低要求是多少")
        metrics = [e for e in entities if e.entity_type == "metric_name"]
        assert any(e.value == "核心一级资本充足率" for e in metrics)

    def test_extract_attachment_no(self):
        """抽取附件号"""
        entities = self.extractor.extract("附件1表中的数据")
        attachments = [e for e in entities if e.entity_type == "attachment_no"]
        assert len(attachments) == 1

    def test_extract_percentage(self):
        """抽取百分比"""
        entities = self.extractor.extract("资本充足率不得低于8%")
        pcts = [e for e in entities if e.entity_type == "percentage"]
        assert len(pcts) == 1
        assert pcts[0].value == "8%"

    def test_extract_term(self):
        """抽取术语"""
        entities = self.extractor.extract("什么是系统重要性银行？")
        terms = [e for e in entities if e.entity_type == "term"]
        assert len(terms) >= 1

    def test_extract_organization(self):
        """抽取机构名"""
        entities = self.extractor.extract("人民银行发布的通知")
        orgs = [e for e in entities if e.entity_type == "organization"]
        assert any(e.value == "人民银行" for e in orgs)

    def test_extract_scope(self):
        """抽取适用范围"""
        entities = self.extractor.extract("系统重要性银行的附加资本要求")
        scopes = [e for e in entities if e.entity_type == "scope"]
        assert any(e.value == "系统重要性银行" for e in scopes)

    def test_deduplication(self):
        """去重"""
        entities = self.extractor.extract("核心一级资本充足率核心一级资本充足率")
        metrics = [e for e in entities if e.entity_type == "metric_name" and e.value == "核心一级资本充足率"]
        assert len(metrics) == 1

    def test_empty_query(self):
        """空问题"""
        entities = self.extractor.extract("")
        assert len(entities) == 0


class TestConstraintExtractor:
    """约束抽取器测试"""

    def setup_method(self):
        self.extractor = ConstraintExtractor()

    def test_extract_time_range(self):
        """抽取时间范围"""
        constraints = self.extractor.extract("2026年1月的数据")
        assert constraints.time_range is not None
        assert "2026" in constraints.time_range

    def test_extract_version_status(self):
        """抽取版本状态"""
        constraints = self.extractor.extract("现行有效的规定")
        assert "active" in constraints.version_status

    def test_extract_scope(self):
        """抽取适用范围"""
        constraints = self.extractor.extract("系统重要性银行的要求")
        assert constraints.applicable_scope == "系统重要性银行"

    def test_extract_normative_level(self):
        """抽取规范强度"""
        constraints = self.extractor.extract("银行不得违反资本充足率要求")
        assert constraints.normative_level == "prohibitive"

    def test_empty_constraints(self):
        """无约束"""
        constraints = self.extractor.extract("你好")
        assert constraints.is_empty()


class TestAmbiguityDetector:
    """歧义检测器测试"""

    def setup_method(self):
        self.detector = AmbiguityDetector()

    def test_vague_reference(self):
        """检测模糊指代"""
        ambiguities = self.detector.detect("那个规定是什么")
        assert len(ambiguities) >= 1
        assert ambiguities[0].ambiguity_type == "entity_ambiguous"

    def test_scope_missing(self):
        """检测范围缺失"""
        ambiguities = self.detector.detect(
            "比例是多少", intent="threshold", entities=[]
        )
        assert len(ambiguities) >= 1
        assert ambiguities[0].ambiguity_type == "scope_missing"

    def test_no_ambiguity(self):
        """无歧义"""
        ambiguities = self.detector.detect(
            "核心一级资本充足率最低要求是多少",
            intent="threshold",
            entities=[type("E", (), {"entity_type": "metric_name", "value": "核心一级资本充足率", "confidence": 0.9})()],
        )
        scope_ambiguities = [a for a in ambiguities if a.ambiguity_type == "scope_missing"]
        assert len(scope_ambiguities) == 0

    def test_polysemous_term(self):
        """检测多义术语"""
        ambiguities = self.detector.detect("资本的要求是什么")
        polysemous = [a for a in ambiguities if a.ambiguity_type == "term_polysemous"]
        assert len(polysemous) >= 1


class TestQuerySpecBuilder:
    """QuerySpec 构建器测试"""

    def setup_method(self):
        self.builder = QuerySpecBuilder()

    def test_build_clause_query(self):
        """构建条款查询 QuerySpec"""
        spec = self.builder.build("《商业银行资本管理办法》第43条")
        assert spec.intent == "clause_query"
        assert spec.complexity == "L1"
        assert spec.risk_level == "low"
        assert len(spec.entities) > 0
        assert spec.confidence > 0.8
        assert spec.query_id  # 非空
        assert spec.raw_query == "《商业银行资本管理办法》第43条"

    def test_build_threshold_query(self):
        """构建阈值查询 QuerySpec"""
        spec = self.builder.build("核心一级资本充足率最低要求是多少")
        assert spec.intent == "threshold"
        assert spec.complexity == "L2"
        assert len(spec.claims) > 0
        # 阈值查询应有指标名称和最低要求值声明
        claim_descs = [c["description"] for c in spec.claims]
        assert any("指标名称" in d for d in claim_descs)

    def test_build_definition_query(self):
        """构建定义查询 QuerySpec"""
        spec = self.builder.build("什么是系统重要性银行")
        assert spec.intent == "definition"
        assert spec.complexity == "L2"
        assert len(spec.claims) > 0

    def test_build_greeting(self):
        """构建问候 QuerySpec"""
        spec = self.builder.build("你好")
        assert spec.intent == "greeting"
        assert spec.complexity == "L0"
        assert spec.risk_level == "low"

    def test_build_with_ambiguity(self):
        """有歧义时复杂度升级"""
        spec = self.builder.build("那个比例是多少")
        # "那个" 是模糊指代，应检测到歧义
        assert len(spec.ambiguities) > 0
        # 歧义应导致复杂度从 L2 升级到 L3
        assert spec.complexity in ("L2", "L3")

    def test_build_retrieval_needs(self):
        """检索通道建议"""
        spec = self.builder.build("《商业银行资本管理办法》第43条")
        assert len(spec.retrieval_needs) > 0
        channels = [rn["channel"] for rn in spec.retrieval_needs]
        assert "exact" in channels

    def test_build_answer_shape(self):
        """回答形态"""
        spec = self.builder.build("核心一级资本充足率最低要求是多少")
        assert spec.answer_shape == "single_value"

    def test_build_to_dict(self):
        """to_dict 序列化"""
        spec = self.builder.build("第43条")
        d = spec.to_dict()
        assert "query_id" in d
        assert "intent" in d
        assert "complexity" in d
        assert "entities" in d
