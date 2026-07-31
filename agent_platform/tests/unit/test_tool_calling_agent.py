"""
工具调用智能体单元测试 — M5.4

测试覆盖:
  - RedisToolCache: 缓存读写、命中/未命中、失败结果不缓存
  - ToolCallingAgent Mock 模式: 工具调用循环、最终回答生成
  - ToolCallingAgent: 最大轮数限制、错误处理
  - ToolCallingAgent: 引用提取
  - LLMClient Function Calling: LLMMessage/LLMResponse 扩展字段
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_platform.runtime.llm_client import LLMClient, LLMMessage, LLMResponse
from agent_platform.tools import (
    CALCULATOR_MANIFEST,
    ToolCallingAgent,
    ToolExecutor,
    ToolManifest,
    ToolRegistry,
    RedisToolCache,
    calculator_handler,
)
from agent_platform.tools.agent.tool_calling_agent import (
    AgentResult,
    ToolCallRecord,
)


# ============================================================
# RedisToolCache 测试
# ============================================================

class TestRedisToolCache:
    """Redis 工具结果缓存测试"""

    @pytest.fixture
    def cache(self):
        return RedisToolCache(mock=True)

    def test_set_and_get(self, cache):
        """写入 → 读取"""
        cache.set("search_chunks", {"query": "test"}, {"success": True, "hits": []})
        result = cache.get("search_chunks", {"query": "test"})
        assert result is not None
        assert result["success"] is True

    def test_cache_miss(self, cache):
        """未写入 → 返回 None"""
        result = cache.get("nonexistent", {"q": "x"})
        assert result is None

    def test_different_params_different_cache(self, cache):
        """不同参数 → 不同缓存"""
        cache.set("tool", {"a": 1}, {"result": "first"})
        cache.set("tool", {"a": 2}, {"result": "second"})

        assert cache.get("tool", {"a": 1})["result"] == "first"
        assert cache.get("tool", {"a": 2})["result"] == "second"

    def test_failed_result_not_cached(self, cache):
        """失败结果不缓存"""
        cache.set("tool", {"q": "x"}, {"success": False, "error": "failed"})
        assert cache.get("tool", {"q": "x"}) is None

    def test_delete(self, cache):
        """删除缓存"""
        cache.set("tool", {"q": "x"}, {"data": "val"})
        assert cache.delete("tool", {"q": "x"}) is True
        assert cache.get("tool", {"q": "x"}) is None

    def test_is_mock(self, cache):
        assert cache.is_mock is True


# ============================================================
# LLMMessage / LLMResponse 扩展字段测试
# ============================================================

class TestLLMExtensions:
    """LLM 客户端 Function Calling 扩展"""

    def test_llm_message_tool_calls(self):
        """LLMMessage 携带 tool_calls"""
        msg = LLMMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}],
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert "tool_calls" in d
        assert d["tool_calls"][0]["id"] == "call_1"

    def test_llm_message_tool_call_id(self):
        """LLMMessage 携带 tool_call_id"""
        msg = LLMMessage(role="tool", content="result", tool_call_id="call_1")
        d = msg.to_dict()
        assert d["tool_call_id"] == "call_1"

    def test_llm_message_no_tool_calls(self):
        """普通消息不含 tool_calls"""
        msg = LLMMessage(role="user", content="hello")
        d = msg.to_dict()
        assert "tool_calls" not in d
        assert "tool_call_id" not in d

    def test_llm_response_has_tool_calls(self):
        """LLMResponse.has_tool_calls"""
        resp = LLMResponse(content="", tool_calls=[{"id": "x"}], finish_reason="tool_calls")
        assert resp.has_tool_calls is True

    def test_llm_response_no_tool_calls(self):
        resp = LLMResponse(content="answer", finish_reason="stop")
        assert resp.has_tool_calls is False


# ============================================================
# ToolCallingAgent Mock 模式测试
# ============================================================

class TestToolCallingAgentMock:
    """工具调用智能体 Mock 模式测试"""

    @pytest.fixture
    def agent(self):
        """创建带 calculator 工具的 Mock 智能体"""
        registry = ToolRegistry()
        registry.register(CALCULATOR_MANIFEST, calculator_handler)

        # 注册一个 mock 检索工具
        search_manifest = ToolManifest(
            name="search_chunks",
            description="搜索法规文档",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        registry.register(search_manifest, lambda x: {
            "success": True,
            "hit_count": 1,
            "hits": [{"chunk_id": "c-001", "content": "核心一级资本充足率不得低于5%", "source_doc": "资本管理办法"}],
        })

        executor = ToolExecutor(registry)
        llm = LLMClient(mock=True)
        return ToolCallingAgent(
            llm_client=llm,
            tool_executor=executor,
            max_tool_calls=3,
            cache=None,
        )

    def test_agent_initialization(self, agent):
        """智能体初始化 → 工具定义和系统提示就绪"""
        assert len(agent._tools_def) == 2
        tool_names = [t["function"]["name"] for t in agent._tools_def]
        assert "calculator" in tool_names
        assert "search_chunks" in tool_names
        assert "银行业监管数据" in agent._system_prompt

    def test_mock_run_returns_result(self, agent):
        """Mock 模式运行 → 返回 AgentResult"""
        result = agent.run("核心一级资本充足率是多少？")
        assert isinstance(result, AgentResult)
        assert result.answer  # 非空
        assert result.iterations >= 1

    def test_mock_run_with_cache(self):
        """Mock 模式 + 缓存 → 工具结果被缓存"""
        registry = ToolRegistry()
        registry.register(CALCULATOR_MANIFEST, calculator_handler)
        executor = ToolExecutor(registry)
        llm = LLMClient(mock=True)
        cache = RedisToolCache(mock=True)

        agent = ToolCallingAgent(
            llm_client=llm,
            tool_executor=executor,
            max_tool_calls=3,
            cache=cache,
        )

        result = agent.run("计算 1+2")
        assert isinstance(result, AgentResult)
        # Mock 模式下可能会调用工具
        if result.tool_calls_made:
            # 验证缓存中有记录
            for tc in result.tool_calls_made:
                if tc.success:
                    cached = cache.get(tc.tool_name, tc.arguments)
                    # 如果不是来自缓存，则应该被缓存了
                    if not tc.cached:
                        assert cached is not None, f"工具 {tc.tool_name} 结果未被缓存"


# ============================================================
# ToolCallingAgent 手动 Mock LLM 测试
# ============================================================

class TestToolCallingAgentManualMock:
    """使用手动 Mock LLM 测试工具调用循环"""

    def _make_agent_with_mock_llm(self, responses: list, max_tool_calls=5):
        """
        创建智能体，LLM 返回预设的响应序列

        Args:
            responses: LLMResponse 列表，按顺序返回
        """
        registry = ToolRegistry()
        registry.register(CALCULATOR_MANIFEST, calculator_handler)

        executor = ToolExecutor(registry)

        # 创建 Mock LLM
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.is_mock = True
        mock_llm.model = "mock-model"
        mock_llm.chat_with_tools = MagicMock(side_effect=responses)
        mock_llm.chat = MagicMock(side_effect=responses)

        return ToolCallingAgent(
            llm_client=mock_llm,
            tool_executor=executor,
            max_tool_calls=max_tool_calls,
            cache=None,
        )

    def test_tool_call_then_answer(self):
        """LLM 先调用工具 → 再返回答案"""
        # 第一轮: LLM 请求调用 calculator
        resp1 = LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_001",
                "type": "function",
                "function": {"name": "calculator", "arguments": json.dumps({"operation": "add", "a": 10, "b": 20})},
            }],
            finish_reason="tool_calls",
        )
        # 第二轮: LLM 返回最终答案
        resp2 = LLMResponse(
            content="10 + 20 = 30",
            finish_reason="stop",
        )

        agent = self._make_agent_with_mock_llm([resp1, resp2])
        result = agent.run("计算 10+20")

        assert result.answer == "10 + 20 = 30"
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0].tool_name == "calculator"
        assert result.tool_calls_made[0].success is True
        assert result.tool_calls_made[0].result["result"] == 30.0
        assert result.is_refusal is False

    def test_multiple_tool_calls_in_one_round(self):
        """一轮中多个工具调用"""
        resp1 = LLMResponse(
            content="",
            tool_calls=[
                {"id": "call_001", "type": "function",
                 "function": {"name": "calculator", "arguments": json.dumps({"operation": "add", "a": 1, "b": 2})}},
                {"id": "call_002", "type": "function",
                 "function": {"name": "calculator", "arguments": json.dumps({"operation": "multiply", "a": 3, "b": 4})}},
            ],
            finish_reason="tool_calls",
        )
        resp2 = LLMResponse(content="计算完成", finish_reason="stop")

        agent = self._make_agent_with_mock_llm([resp1, resp2])
        result = agent.run("多步计算")

        assert len(result.tool_calls_made) == 2
        assert result.tool_calls_made[0].result["result"] == 3.0
        assert result.tool_calls_made[1].result["result"] == 12.0

    def test_max_tool_calls_limit(self):
        """超过最大轮数 → 强制终止"""
        # 每轮都请求工具调用，永不返回最终答案
        tool_call_resp = LLMResponse(
            content="",
            tool_calls=[{"id": "call_001", "type": "function",
                         "function": {"name": "calculator", "arguments": json.dumps({"operation": "add", "a": 1, "b": 1})}}],
            finish_reason="tool_calls",
        )
        final_resp = LLMResponse(content="最终回答", finish_reason="stop")

        # max_tool_calls=2，循环跑 3 轮全部 tool_calls → 强制终止
        # chat_with_tools 需要 3 个 tool_call 响应，chat 需要 1 个 final 响应
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.is_mock = True
        mock_llm.model = "mock-model"
        mock_llm.chat_with_tools = MagicMock(side_effect=[tool_call_resp, tool_call_resp, tool_call_resp])
        mock_llm.chat = MagicMock(side_effect=[final_resp])

        registry = ToolRegistry()
        registry.register(CALCULATOR_MANIFEST, calculator_handler)
        executor = ToolExecutor(registry)

        agent = ToolCallingAgent(
            llm_client=mock_llm,
            tool_executor=executor,
            max_tool_calls=2,
            cache=None,
        )
        result = agent.run("无限工具调用")

        # 应该执行了 3 轮工具调用（loop range(1, max+2) = [1,2,3]）
        assert len(result.tool_calls_made) == 3
        assert result.refusal_reason == "达到最大工具调用次数"

    def test_tool_execution_error_handled(self):
        """工具执行错误 → 不崩溃"""
        resp1 = LLMResponse(
            content="",
            tool_calls=[{"id": "call_001", "type": "function",
                         "function": {"name": "calculator", "arguments": json.dumps({"operation": "divide", "a": 10, "b": 0})}}],
            finish_reason="tool_calls",
        )
        resp2 = LLMResponse(content="除零错误，无法计算", finish_reason="stop")

        agent = self._make_agent_with_mock_llm([resp1, resp2])
        result = agent.run("10/0")

        assert len(result.tool_calls_made) == 1
        # calculator 的 divide by zero 返回 {success: False} 而非抛异常
        # ToolExecutor 视 handler 正常返回为成功，错误信息在 result data 中
        assert result.tool_calls_made[0].result["success"] is False

    def test_refusal_answer(self):
        """LLM 返回拒答"""
        resp1 = LLMResponse(content="依据不足，未找到相关数据", finish_reason="stop")

        agent = self._make_agent_with_mock_llm([resp1])
        result = agent.run("无法回答的问题")

        assert result.is_refusal is True
        assert result.refusal_reason == "证据不足"

    def test_citation_extraction(self):
        """引用提取"""
        # 注册带 hits 的检索工具
        registry = ToolRegistry()
        registry.register(CALCULATOR_MANIFEST, calculator_handler)
        search_manifest = ToolManifest(
            name="search_chunks",
            description="搜索",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        registry.register(search_manifest, lambda x: {
            "success": True,
            "hit_count": 2,
            "hits": [
                {"chunk_id": "c-001", "content": "法规A", "source_doc": "文档A"},
                {"chunk_id": "c-002", "content": "法规B", "source_doc": "文档B"},
            ],
        })

        executor = ToolExecutor(registry)
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.is_mock = True
        mock_llm.model = "mock"

        resp1 = LLMResponse(
            content="",
            tool_calls=[{"id": "call_001", "type": "function",
                         "function": {"name": "search_chunks", "arguments": json.dumps({"query": "资本充足率"})}}],
            finish_reason="tool_calls",
        )
        resp2 = LLMResponse(content="根据检索结果...", finish_reason="stop")
        mock_llm.chat_with_tools = MagicMock(side_effect=[resp1, resp2])
        mock_llm.chat = MagicMock(side_effect=[resp1, resp2])

        agent = ToolCallingAgent(
            llm_client=mock_llm,
            tool_executor=executor,
            max_tool_calls=3,
        )
        result = agent.run("资本充足率")

        assert len(result.citations) >= 1
        assert any(c.get("chunk_id") == "c-001" for c in result.citations)


# ============================================================
# create_agent 工厂函数测试
# ============================================================

class TestCreateAgent:
    """工厂函数测试"""

    def test_create_agent_with_mock_llm(self):
        """create_agent + Mock LLM → 智能体可创建"""
        from agent_platform.tools import create_agent

        llm = LLMClient(mock=True)
        agent = create_agent(
            retrieval_base_url="http://127.0.0.1:8000",
            max_tool_calls=3,
            use_cache=False,
            llm_client=llm,
        )
        assert isinstance(agent, ToolCallingAgent)
        # 应该注册了 calculator + 5 个检索工具 + version_checker + date_parser = 8 个
        assert len(agent._tools_def) >= 7
