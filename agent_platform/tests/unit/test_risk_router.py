"""
风险路由器 + 综合路由策略 单元测试

测试用例:
  - compliance + "是否符合" → high risk → P4
  - threshold + "最低要求" → medium risk → P2/P3
  - definition + "什么是" → low risk → P2
  - clause_query + "第43条" → low risk → P1
  - threshold + "不得低于" → high risk (prohibitive)
  - greeting → P0

附加测试:
  - RiskRouter 直接传参模式
  - RiskAssessment / ExecutionPath / ComprehensiveRouteDecision 序列化
  - PolicyLoader 加载与热更新
  - RiskLevel 枚举
"""

import pytest

from agent_platform.query_understanding import QuerySpecBuilder
from agent_platform.routing.risk_router import (
    RiskAssessment,
    RiskLevel,
    RiskRouter,
)
from agent_platform.routing.route_policy import (
    ComprehensiveRouteDecision,
    ExecutionPath,
    PolicyLoader,
    RoutePolicy,
)


# ============================================================
# RiskRouter 测试
# ============================================================
class TestRiskRouter:
    """风险路由器测试"""

    def setup_method(self):
        self.router = RiskRouter()
        self.spec_builder = QuerySpecBuilder()

    # ── 意图驱动的基础风险级别 ──

    def test_compliance_is_high_risk(self):
        """合规查询 → 高风险"""
        assessment = self.router.assess(
            intent="compliance", query_text="银行是否符合资本充足率要求"
        )
        assert assessment.level == "high"
        assert "intent=compliance" in assessment.factors

    def test_threshold_is_medium_risk(self):
        """阈值查询 → 中风险"""
        assessment = self.router.assess(
            intent="threshold", query_text="核心一级资本充足率最低要求是多少"
        )
        assert assessment.level == "medium"

    def test_definition_is_low_risk(self):
        """定义查询 → 低风险"""
        assessment = self.router.assess(
            intent="definition", query_text="什么是核心一级资本充足率"
        )
        assert assessment.level == "low"

    def test_clause_query_is_low_risk(self):
        """条款查询 → 低风险"""
        assessment = self.router.assess(
            intent="clause_query", query_text="第43条"
        )
        assert assessment.level == "low"

    def test_overview_is_low_risk(self):
        """概览查询 → 低风险"""
        assessment = self.router.assess(
            intent="overview", query_text="资本管理办法的主要内容"
        )
        assert assessment.level == "low"

    def test_greeting_is_low_risk(self):
        """问候 → 低风险"""
        assessment = self.router.assess(
            intent="greeting", query_text="你好"
        )
        assert assessment.level == "low"

    # ── 关键词升级测试 ──

    def test_prohibitive_keyword_upgrades_to_high(self):
        """禁止性关键词 '不得低于' → 升级为高风险"""
        assessment = self.router.assess(
            intent="threshold", query_text="核心一级资本充足率不得低于多少"
        )
        assert assessment.level == "high"
        assert any("prohibitive_keyword" in f for f in assessment.factors)

    def test_prohibitive_keyword_bukode(self):
        """禁止性关键词 '不得' → 升级为高风险"""
        assessment = self.router.assess(
            intent="definition", query_text="不得违反的规定是什么"
        )
        assert assessment.level == "high"

    def test_penalty_keyword_upgrades_to_high(self):
        """处罚关键词 '处罚' → 升级为高风险"""
        assessment = self.router.assess(
            intent="overview", query_text="违规处罚的概述"
        )
        assert assessment.level == "high"
        assert any("penalty_keyword" in f for f in assessment.factors)

    def test_compliance_check_keyword_upgrades_to_high(self):
        """合规判断关键词 '是否符合' → 升级为高风险"""
        assessment = self.router.assess(
            intent="overview", query_text="银行是否符合监管要求"
        )
        assert assessment.level == "high"
        assert any("compliance_check_keyword" in f for f in assessment.factors)

    # ── 数值敏感度测试 ──

    def test_numeric_sensitivity_percentage(self):
        """百分比 + 阈值关键词 → 高风险"""
        assessment = self.router.assess(
            intent="threshold", query_text="资本充足率最低要求是8%"
        )
        assert assessment.level == "high"
        assert any("numeric_present" in f for f in assessment.factors)

    def test_numeric_sensitivity_amount(self):
        """金额 + 阈值关键词 → 高风险"""
        assessment = self.router.assess(
            intent="threshold", query_text="最低注册资本要求是10亿元"
        )
        assert assessment.level == "high"

    def test_no_numeric_sensitivity_without_threshold(self):
        """有百分比但无阈值关键词 → 不升级"""
        assessment = self.router.assess(
            intent="definition", query_text="什么是8%的资本充足率"
        )
        assert assessment.level == "low"

    # ── QuerySpec 对象输入测试 ──

    def test_assess_with_query_spec(self):
        """通过 QuerySpec 对象评估风险"""
        spec = self.spec_builder.build("银行是否符合资本充足率要求")
        assessment = self.router.assess(spec)
        assert assessment.level == "high"

    def test_assess_with_dict(self):
        """通过 dict 评估风险"""
        spec = {
            "intent": "threshold",
            "raw_query": "核心一级资本充足率最低要求是多少",
            "entities": [],
        }
        assessment = self.router.assess(spec)
        assert assessment.level == "medium"

    # ── 序列化测试 ──

    def test_risk_assessment_to_dict(self):
        """风险评估结果序列化"""
        assessment = self.router.assess(
            intent="compliance", query_text="是否符合要求"
        )
        d = assessment.to_dict()
        assert "level" in d
        assert "reason" in d
        assert "factors" in d
        assert isinstance(d["factors"], list)

    def test_risk_assessment_has_reason(self):
        """风险评估结果包含原因说明"""
        assessment = self.router.assess(
            intent="threshold", query_text="不得低于8%"
        )
        assert assessment.reason  # 非空
        assert "意图" in assessment.reason or "关键词" in assessment.reason


# ============================================================
# RiskLevel 枚举测试
# ============================================================
class TestRiskLevel:
    """风险级别枚举测试"""

    def test_enum_values(self):
        """枚举值正确"""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"

    def test_from_str(self):
        """从字符串构建枚举"""
        assert RiskLevel.from_str("low") == RiskLevel.LOW
        assert RiskLevel.from_str("medium") == RiskLevel.MEDIUM
        assert RiskLevel.from_str("high") == RiskLevel.HIGH

    def test_from_str_invalid_defaults_to_medium(self):
        """无效字符串默认为 medium"""
        assert RiskLevel.from_str("invalid") == RiskLevel.MEDIUM

    def test_str_representation(self):
        """字符串表示"""
        assert str(RiskLevel.HIGH) == "high"


# ============================================================
# RoutePolicy 综合路由策略测试
# ============================================================
class TestRoutePolicy:
    """综合路由策略测试"""

    def setup_method(self):
        self.policy = RoutePolicy()
        self.spec_builder = QuerySpecBuilder()

    # ── 核心映射测试 ──

    def test_compliance_high_risk_to_p4(self):
        """合规 + '是否符合' → 高风险 → P4"""
        spec = self.spec_builder.build("银行是否符合资本充足率要求")
        decision = self.policy.decide(spec)
        assert decision.risk_level == "high"
        assert decision.path_id == "P4"
        assert decision.level == "L4"

    def test_threshold_medium_risk_to_p2_or_p3(self):
        """阈值 + '最低要求' → 中风险 → P2/P3"""
        spec = self.spec_builder.build("核心一级资本充足率最低要求是多少")
        decision = self.policy.decide(spec)
        assert decision.risk_level == "medium"
        assert decision.path_id in ("P2", "P3")

    def test_definition_low_risk_to_p2(self):
        """定义 + '什么是' → 低风险 → P2"""
        spec = self.spec_builder.build("什么是核心一级资本充足率")
        decision = self.policy.decide(spec)
        assert decision.risk_level == "low"
        assert decision.path_id == "P2"

    def test_clause_query_low_risk_to_p1(self):
        """条款查询 + '第43条' → 低风险 → P1"""
        spec = self.spec_builder.build("第43条")
        decision = self.policy.decide(spec)
        assert decision.risk_level == "low"
        assert decision.path_id == "P1"

    def test_greeting_to_p0(self):
        """问候 → P0"""
        spec = self.spec_builder.build("你好")
        decision = self.policy.decide(spec)
        assert decision.path_id == "P0"
        assert decision.level == "L0"
        assert decision.channels == []

    def test_threshold_prohibitive_to_high_risk(self):
        """阈值 + '不得低于' → 高风险（禁止性条款）"""
        spec = self.spec_builder.build("核心一级资本充足率不得低于多少")
        decision = self.policy.decide(spec)
        assert decision.risk_level == "high"

    # ── 执行路径配置验证 ──

    def test_p0_no_retrieval(self):
        """P0 路径无需检索"""
        spec = self.spec_builder.build("你好")
        decision = self.policy.decide(spec)
        assert decision.execution_path is not None
        assert decision.execution_path.retrieval is False
        assert decision.execution_path.cache_first is True

    def test_p1_exact_channels(self):
        """P1 路径使用 exact + metadata 通道"""
        spec = self.spec_builder.build("第43条")
        decision = self.policy.decide(spec)
        assert "exact" in decision.channels
        assert "metadata" in decision.channels

    def test_p4_has_decomposition(self):
        """P4 路径需要拆解"""
        spec = self.spec_builder.build("银行是否符合资本充足率要求")
        decision = self.policy.decide(spec)
        assert decision.need_decomposition is True

    def test_p4_full_channels(self):
        """P4 路径包含全通道"""
        spec = self.spec_builder.build("银行是否符合资本充足率要求")
        decision = self.policy.decide(spec)
        assert "lexical" in decision.channels
        assert "dense" in decision.channels
        assert "table" in decision.channels
        assert "relation" in decision.channels
        assert "neighborhood" in decision.channels

    # ── 综合决策序列化 ──

    def test_decision_to_dict(self):
        """综合决策序列化"""
        spec = self.spec_builder.build("什么是核心一级资本充足率")
        decision = self.policy.decide(spec)
        d = decision.to_dict()
        assert "intent" in d
        assert "level" in d
        assert "risk_level" in d
        assert "path_id" in d
        assert "execution_path" in d
        assert "risk_assessment" in d

    def test_decision_has_risk_assessment(self):
        """综合决策包含风险评估详情"""
        spec = self.spec_builder.build("银行是否符合资本充足率要求")
        decision = self.policy.decide(spec)
        assert decision.risk_assessment is not None
        assert decision.risk_assessment.level == "high"

    # ── 列表与获取 ──

    def test_list_paths(self):
        """列出所有执行路径"""
        paths = self.policy.list_paths()
        assert "P0" in paths
        assert "P1" in paths
        assert "P2" in paths
        assert "P3" in paths
        assert "P4" in paths

    def test_get_path(self):
        """获取指定执行路径"""
        path = self.policy.get_path("P4")
        assert path is not None
        assert path.path_id == "P4"
        assert path.need_decomposition is True


# ============================================================
# ExecutionPath 数据结构测试
# ============================================================
class TestExecutionPath:
    """执行路径数据结构测试"""

    def test_to_dict(self):
        """执行路径序列化"""
        path = ExecutionPath(
            path_id="P1",
            description="测试路径",
            channels=["exact"],
            top_k=5,
            rerank=False,
            budget_ms=2000,
        )
        d = path.to_dict()
        assert d["path_id"] == "P1"
        assert d["description"] == "测试路径"
        assert d["channels"] == ["exact"]
        assert d["top_k"] == 5
        assert d["rerank"] is False
        assert d["budget_ms"] == 2000

    def test_defaults(self):
        """默认值"""
        path = ExecutionPath(path_id="P0", description="测试")
        assert path.channels == []
        assert path.top_k == 10
        assert path.rerank is False
        assert path.budget_ms == 5000
        assert path.max_retries == 1
        assert path.need_decomposition is False
        assert path.cache_first is False
        assert path.retrieval is True


# ============================================================
# PolicyLoader 测试
# ============================================================
class TestPolicyLoader:
    """策略加载器测试"""

    def test_load_default(self):
        """加载默认配置"""
        loader = PolicyLoader()
        paths = loader.load()
        assert "P0" in paths
        assert "P1" in paths
        assert "P2" in paths
        assert "P3" in paths
        assert "P4" in paths

    def test_load_from_nonexistent_file(self):
        """文件不存在时回退到默认"""
        loader = PolicyLoader(config_path="/nonexistent/path.yaml")
        paths = loader.load()
        assert "P0" in paths
        assert paths["P0"].cache_first is True

    def test_reload(self):
        """热更新"""
        loader = PolicyLoader()
        loader.load()
        paths = loader.reload()
        assert "P4" in paths

    def test_get_path(self):
        """获取指定路径"""
        loader = PolicyLoader()
        loader.load()
        path = loader.get("P4")
        assert path is not None
        assert path.need_decomposition is True

    def test_list_path_ids(self):
        """列出路径 ID"""
        loader = PolicyLoader()
        loader.load()
        ids = loader.list_path_ids()
        assert len(ids) == 5
        assert ids == sorted(ids)

    def test_p0_config_values(self):
        """P0 配置值正确"""
        loader = PolicyLoader()
        paths = loader.load()
        p0 = paths["P0"]
        assert p0.cache_first is True
        assert p0.retrieval is False
        assert p0.top_k == 0
        assert p0.channels == []

    def test_p4_config_values(self):
        """P4 配置值正确"""
        loader = PolicyLoader()
        paths = loader.load()
        p4 = paths["P4"]
        assert p4.top_k == 50
        assert p4.rerank is True
        assert p4.rerank_top_n == 15
        assert p4.budget_ms == 15000
        assert p4.max_retries == 3
        assert p4.need_decomposition is True
