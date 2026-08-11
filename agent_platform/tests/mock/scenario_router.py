"""
场景路由 — 根据 RetrievalRequest 的 query 或 scenario 字段匹配预设场景

支持的场景：
  - normal:           正常检索，返回完整结果
  - empty:             空结果场景，返回空数组 []
  - timeout:          超时场景，抛出 MockTimeoutError 异常
  - version_conflict: 版本冲突场景，返回带版本冲突标记的结果
  - partial_failure:  部分失败场景，返回部分标记为失败的结果

匹配逻辑：
  1. 优先检查请求中的 "scenario" 字段（显式场景标识）
  2. 若无 scenario 字段，根据 query 内容做模糊匹配
  3. 默认回退到 "normal" 场景

用法：
    from agent_platform.tests.mock.scenario_router import ScenarioRouter

    router = ScenarioRouter(data_loader=loader, default_hits=my_hits)
    try:
        response = router.route({"query": "核心一级资本", "scenario": "normal"})
    except MockTimeoutError:
        print("检索超时")
"""

from typing import Any, Dict, List, Optional, Union

from .data_loader import DataLoader


class MockTimeoutError(Exception):
    """
    Mock 检索超时异常

    当请求匹配到 timeout 场景时，由 ScenarioRouter.route() 抛出。
    Mock API 层捕获此异常并返回 504 状态码。
    """

    def __init__(self, message: str = "Mock 检索超时：匹配到 timeout 场景", scenario: str = "timeout"):
        self.scenario = scenario
        super().__init__(message)


class ScenarioRouter:
    """
    场景路由器 — 根据检索请求匹配预设场景并返回对应响应

    优先级：
      1. 显式 scenario 字段 > 2. query 模糊匹配 > 3. 默认 "normal"

    Attributes:
        data_loader:  DataLoader 实例，用于加载预设场景的请求/响应
        default_hits: 默认检索结果列表（RetrievalHit dict 格式），
                      当 DataLoader 无预设响应时使用
    """

    # 支持的场景列表
    SCENARIOS = [
        "normal",
        "empty",
        "timeout",
        "version_conflict",
        "partial_failure",
    ]

    # query 模糊匹配关键词映射
    # 当请求没有显式 scenario 字段时，根据 query 内容匹配场景
    SCENARIO_KEYWORDS: Dict[str, List[str]] = {
        "timeout": ["timeout", "超时", "time out", "timed out", "超时场景"],
        "empty": ["empty", "空结果", "无结果", "no result", "空场景"],
        "version_conflict": [
            "version_conflict",
            "版本冲突",
            "版本不一致",
            "version conflict",
        ],
        "partial_failure": [
            "partial_failure",
            "部分失败",
            "partial",
            "部分错误",
        ],
    }

    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        default_hits: Optional[List[dict]] = None,
    ):
        """
        初始化场景路由器

        Args:
            data_loader: DataLoader 实例，用于加载 contracts/examples/ 下的预设场景
            default_hits: 默认检索结果列表，当 DataLoader 无对应预设响应时回退使用
        """
        self._data_loader = data_loader
        self._default_hits = default_hits or []

    # ============================================================
    # 核心路由方法
    # ============================================================
    def route(self, request: Union[dict, Any]) -> List[dict]:
        """
        根据请求路由到对应场景，返回检索结果

        对于 timeout 场景，抛出 MockTimeoutError 异常而非返回结果。
        对于 empty 场景，返回空数组 []。
        其余场景返回对应的结果列表。

        Args:
            request: 检索请求，可以是 dict 或带有 query/scenario 属性的对象

        Returns:
            检索结果列表（RetrievalHit dict 格式）

        Raises:
            MockTimeoutError: 当匹配到 timeout 场景时
        """
        scenario = self._determine_scenario(request)

        if scenario == "timeout":
            raise MockTimeoutError(
                message=f"Mock 检索超时：请求匹配到 timeout 场景 "
                f"(query={self._get_query(request)[:50]})",
                scenario=scenario,
            )

        if scenario == "empty":
            return []

        # 尝试从 DataLoader 加载预设响应
        preset_response = self._load_preset_response(scenario)
        if preset_response is not None:
            if scenario == "version_conflict":
                return self._add_version_conflict_markers(preset_response)
            if scenario == "partial_failure":
                return self._mark_partial_failures(preset_response)
            return preset_response

        # 回退到内置默认 hits
        if scenario == "version_conflict":
            return self._add_version_conflict_markers(self._default_hits)
        if scenario == "partial_failure":
            return self._mark_partial_failures(self._default_hits)

        return self._default_hits

    # ============================================================
    # 场景判定
    # ============================================================
    def _determine_scenario(self, request: Union[dict, Any]) -> str:
        """
        判定请求应路由到哪个场景

        优先级：
          1. 显式 scenario 字段
          2. query 内容模糊匹配
          3. 默认 "normal"

        Args:
            request: 检索请求

        Returns:
            场景名称字符串
        """
        # 1. 检查显式 scenario 字段
        scenario = self._get_field(request, "scenario")
        if scenario and scenario in self.SCENARIOS:
            return scenario

        # 2. query 模糊匹配
        query = self._get_query(request).lower()
        for scenario_name, keywords in self.SCENARIO_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in query:
                    return scenario_name

        # 3. 默认正常场景
        return "normal"

    # ============================================================
    # 预设响应加载
    # ============================================================
    def _load_preset_response(self, scenario: str) -> Optional[List[dict]]:
        """
        从 DataLoader 加载预设场景响应

        Args:
            scenario: 场景名称

        Returns:
            预设响应列表，若无则返回 None
        """
        if self._data_loader is None:
            return None

        _, response = self._data_loader.get_scenario(scenario)
        return response

    # ============================================================
    # 场景结果加工
    # ============================================================
    @staticmethod
    def _add_version_conflict_markers(hits: List[dict]) -> List[dict]:
        """
        为检索结果添加版本冲突标记

        在每条 hit 的 metadata 中注入 version_status 和 version_conflict 字段，
        模拟同一条款在不同版本间存在冲突的情况。

        Args:
            hits: 原始检索结果列表

        Returns:
            带版本冲突标记的结果列表（深拷贝，不修改原始数据）
        """
        import copy

        result = []
        for i, hit in enumerate(hits):
            hit_copy = copy.deepcopy(hit)
            meta = hit_copy.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}
            # 交替标记为 active / superseded，模拟版本冲突
            meta["version_status"] = "superseded" if i % 2 == 0 else "active"
            meta["version_conflict"] = True
            meta["conflicting_doc_version"] = "2023" if i % 2 == 0 else "2024"
            hit_copy["metadata"] = meta
            result.append(hit_copy)
        return result

    @staticmethod
    def _mark_partial_failures(hits: List[dict]) -> List[dict]:
        """
        标记部分检索结果为部分失败

        在部分 hit 的 trace 中注入检索通道失败信息，
        模拟某些检索通道（如 dense）返回失败的场景。

        Args:
            hits: 原始检索结果列表

        Returns:
            带部分失败标记的结果列表（深拷贝，不修改原始数据）
        """
        import copy

        result = []
        for i, hit in enumerate(hits):
            hit_copy = copy.deepcopy(hit)
            trace = hit_copy.get("trace", {})
            if not isinstance(trace, dict):
                trace = {}
            # 每隔一条标记一个通道失败
            if i % 2 == 1:
                trace["retrieval_status"] = "partial_failure"
                trace["failed_channels"] = ["dense"]
                trace["failure_detail"] = "Dense 检索通道超时，仅返回 BM25 结果"
            else:
                trace["retrieval_status"] = "success"
            hit_copy["trace"] = trace
            result.append(hit_copy)
        return result

    # ============================================================
    # 内部工具
    # ============================================================
    @staticmethod
    def _get_field(request: Union[dict, Any], field: str) -> Optional[str]:
        """从 dict 或对象中获取字段值"""
        if isinstance(request, dict):
            return request.get(field)
        return getattr(request, field, None)

    @classmethod
    def _get_query(cls, request: Union[dict, Any]) -> str:
        """从请求中提取 query 文本"""
        query = cls._get_field(request, "query")
        return query or ""

    def __repr__(self) -> str:
        return (
            f"ScenarioRouter(scenarios={self.SCENARIOS}, "
            f"data_loader={'yes' if self._data_loader else 'no'}, "
            f"default_hits={len(self._default_hits)})"
        )
