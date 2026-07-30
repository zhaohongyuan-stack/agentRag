"""
检索适配层 — B组调用A组检索服务的统一客户端

职责:
  1. 将B组的 QuerySpec/RouteDecision 转换为A组的 RetrievalRequest 格式
  2. 调用A组统一检索 API（/api/v1/search）
  3. 处理超时、错误、重试
  4. 返回 RetrievalHit 列表

支持两种模式:
  - HTTP 模式: 通过 HTTP 调用 Mock/真实检索服务
  - In-Process 模式: 直接调用 Mock 检索服务（用于单元测试，无需启动 HTTP 服务）

A组 API 接口:
  POST /api/v1/search  — 统一检索入口
  请求体: {query, strategy, top_k, filters, expand_context}
  响应: RetrievalHit 列表
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from urllib.error import URLError
from urllib.request import Request, urlopen
import urllib.error

logger = logging.getLogger(__name__)


# ============================================================
# 检索结果数据结构
# ============================================================
@dataclass
class RetrievalResult:
    """检索调用结果"""

    success: bool
    hits: List[dict] = field(default_factory=list)
    error: Optional[str] = None
    error_code: Optional[str] = None
    latency_ms: float = 0.0
    request_id: str = ""
    strategy: str = "hybrid"
    scenario: Optional[str] = None

    @property
    def hit_count(self) -> int:
        return len(self.hits)

    @property
    def is_empty(self) -> bool:
        return self.success and self.hit_count == 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "hit_count": self.hit_count,
            "error": self.error,
            "error_code": self.error_code,
            "latency_ms": self.latency_ms,
            "request_id": self.request_id,
            "strategy": self.strategy,
        }


# ============================================================
# 检索客户端
# ============================================================
class RetrievalClient:
    """
    检索适配客户端

    将B组的 QuerySpec 和 RouteDecision 转换为A组 API 请求，
    调用检索服务并返回结果。

    使用方式（HTTP 模式）:
        client = RetrievalClient(base_url="http://127.0.0.1:8001")
        result = client.search(query="核心一级资本充足率", strategy="hybrid", top_k=10)

    使用方式（In-Process 模式）:
        client = RetrievalClient(in_process=True)
        result = client.search(query="核心一级资本充足率")
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        timeout_ms: int = 5000,
        in_process: bool = False,
        mock_api: Optional[Any] = None,
    ):
        """
        Args:
            base_url: 检索服务基础 URL
            timeout_ms: 超时时间（毫秒）
            in_process: 是否使用进程内模式（直接调用 Mock API，不经 HTTP）
            mock_api: 进程内模式下的 Mock API 实例
        """
        self._base_url = base_url.rstrip("/")
        self._timeout_ms = timeout_ms
        self._in_process = in_process
        self._mock_api = mock_api

        if in_process and mock_api is None:
            # 延迟导入 Mock API，避免循环依赖
            from agent_platform.tests.mock.mock_retrieval_api import (
                MOCK_CHUNKS,
                _build_default_hits,
                _scenario_router,
            )
            self._mock_api = type("MockAPI", (), {
                "search": staticmethod(lambda req: _scenario_router.route(req)),
                "default_hits": _build_default_hits(),
            })()

    def search(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 10,
        filters: Optional[dict] = None,
        expand_context: bool = False,
        scenario: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> RetrievalResult:
        """
        执行检索

        Args:
            query: 查询文本
            strategy: 检索策略 (hybrid/bm25/dense/exact/metadata/relation/table)
            top_k: 返回结果数
            filters: 元数据过滤条件
            expand_context: 是否扩展上下文
            scenario: Mock 场景标识（仅 Mock 模式有效）
            timeout_ms: 本次请求超时（毫秒），为 None 时使用默认值

        Returns:
            RetrievalResult 对象
        """
        request_id = str(uuid.uuid4())
        timeout = (timeout_ms or self._timeout_ms) / 1000.0

        if self._in_process:
            return self._search_in_process(
                query=query,
                strategy=strategy,
                top_k=top_k,
                filters=filters or {},
                expand_context=expand_context,
                scenario=scenario,
                request_id=request_id,
                timeout_ms=timeout_ms or self._timeout_ms,
            )
        else:
            return self._search_http(
                query=query,
                strategy=strategy,
                top_k=top_k,
                filters=filters or {},
                expand_context=expand_context,
                scenario=scenario,
                request_id=request_id,
                timeout=timeout,
            )

    def search_by_spec(
        self,
        query_text: str,
        route_decision: Any,
        filters: Optional[dict] = None,
    ) -> RetrievalResult:
        """
        基于 RouteDecision 执行检索

        将 RouteDecision 的 channels 映射为 A组的 strategy，
        使用 top_k 和 budget_ms 配置。

        Args:
            query_text: 查询文本
            route_decision: RouteDecision 对象
            filters: 元数据过滤条件

        Returns:
            RetrievalResult 对象
        """
        # 通道映射: B组通道 → A组 strategy
        strategy = self._channels_to_strategy(route_decision.channels)

        # 如果 L0 无需检索
        if route_decision.level == "L0" or not route_decision.channels:
            return RetrievalResult(
                success=True,
                hits=[],
                request_id=str(uuid.uuid4()),
                strategy="none",
                latency_ms=0.0,
            )

        return self.search(
            query=query_text,
            strategy=strategy,
            top_k=route_decision.top_k,
            filters=filters,
            timeout_ms=route_decision.budget_ms,
        )

    # ============================================================
    # 内部实现
    # ============================================================

    def _search_http(
        self,
        query: str,
        strategy: str,
        top_k: int,
        filters: dict,
        expand_context: bool,
        scenario: Optional[str],
        request_id: str,
        timeout: float,
    ) -> RetrievalResult:
        """HTTP 模式检索"""
        request_body = json.dumps({
            "query": query,
            "strategy": strategy,
            "top_k": top_k,
            "filters": filters,
            "expand_context": expand_context,
            **({"scenario": scenario} if scenario else {}),
        }).encode("utf-8")

        url = f"{self._base_url}/api/v1/search"
        req = Request(url, data=request_body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-Request-Id", request_id)

        start_time = time.time()
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                latency_ms = (time.time() - start_time) * 1000

                if resp.status == 504:
                    return RetrievalResult(
                        success=False,
                        error="检索超时",
                        error_code="RT_TIMEOUT",
                        latency_ms=latency_ms,
                        request_id=request_id,
                        strategy=strategy,
                    )

                hits = json.loads(body)
                if not isinstance(hits, list):
                    hits = []

                return RetrievalResult(
                    success=True,
                    hits=hits,
                    latency_ms=latency_ms,
                    request_id=request_id,
                    strategy=strategy,
                    scenario=scenario,
                )

        except urllib.error.HTTPError as e:
            latency_ms = (time.time() - start_time) * 1000
            if e.code == 504:
                return RetrievalResult(
                    success=False,
                    error="检索超时",
                    error_code="RT_TIMEOUT",
                    latency_ms=latency_ms,
                    request_id=request_id,
                    strategy=strategy,
                )
            return RetrievalResult(
                success=False,
                error=f"检索服务返回错误: HTTP {e.code}",
                error_code="RT_HTTP_ERROR",
                latency_ms=latency_ms,
                request_id=request_id,
                strategy=strategy,
            )

        except URLError as e:
            latency_ms = (time.time() - start_time) * 1000
            return RetrievalResult(
                success=False,
                error=f"检索服务连接失败: {str(e)}",
                error_code="RT_CONNECTION_ERROR",
                latency_ms=latency_ms,
                request_id=request_id,
                strategy=strategy,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return RetrievalResult(
                success=False,
                error=f"检索异常: {str(e)}",
                error_code="RT_INTERNAL_ERROR",
                latency_ms=latency_ms,
                request_id=request_id,
                strategy=strategy,
            )

    def _search_in_process(
        self,
        query: str,
        strategy: str,
        top_k: int,
        filters: dict,
        expand_context: bool,
        scenario: Optional[str],
        request_id: str,
        timeout_ms: int,
    ) -> RetrievalResult:
        """进程内模式检索（直接调用 Mock API）"""
        from agent_platform.tests.mock.scenario_router import MockTimeoutError

        request_dict = {
            "query": query,
            "strategy": strategy,
            "top_k": top_k,
            "filters": filters,
            "expand_context": expand_context,
        }
        if scenario:
            request_dict["scenario"] = scenario

        start_time = time.time()
        try:
            hits = self._mock_api.search(request_dict)
            latency_ms = (time.time() - start_time) * 1000

            if not isinstance(hits, list):
                hits = []

            return RetrievalResult(
                success=True,
                hits=hits,
                latency_ms=latency_ms,
                request_id=request_id,
                strategy=strategy,
                scenario=scenario,
            )

        except MockTimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            return RetrievalResult(
                success=False,
                error="检索超时（Mock timeout 场景）",
                error_code="RT_TIMEOUT",
                latency_ms=latency_ms,
                request_id=request_id,
                strategy=strategy,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return RetrievalResult(
                success=False,
                error=f"检索异常: {str(e)}",
                error_code="RT_INTERNAL_ERROR",
                latency_ms=latency_ms,
                request_id=request_id,
                strategy=strategy,
            )

    @staticmethod
    def _channels_to_strategy(channels: List[str]) -> str:
        """
        将B组检索通道列表映射为A组的单策略

        A组策略: hybrid, bm25, dense, exact, metadata, relation, table

        映射规则（按优先级）:
          - 包含 exact → exact
          - 包含 lexical + dense → hybrid（RRF 融合，覆盖面最广）
          - 包含 relation/neighborhood → relation
          - 包含 table → table
          - 仅 lexical → bm25
          - 仅 dense → dense
          - 仅 metadata → metadata
          - 其他 → hybrid

        注意: hybrid 优先于 table，因为 hybrid 同时覆盖关键词和语义匹配，
        而 table 策略需要 table_name 过滤器，普通查询不具备。
        """
        if not channels:
            return "hybrid"

        channel_set = set(channels)

        if "exact" in channel_set:
            return "exact"
        if "lexical" in channel_set and "dense" in channel_set:
            return "hybrid"
        if "relation" in channel_set or "neighborhood" in channel_set:
            return "relation"
        if "table" in channel_set:
            return "table"
        if "lexical" in channel_set:
            return "bm25"
        if "dense" in channel_set:
            return "dense"
        if "metadata" in channel_set:
            return "metadata"

        return "hybrid"
