"""
工具调用智能体 — M5.4 工具调用智能体模块

编排 LLM 与工具的交互循环，实现 Agentic RAG:
  1. LLM 根据用户问题决定调用哪些工具
  2. 通过 ToolExecutor 执行工具（检索/计算）
  3. 将工具结果回传给 LLM
  4. LLM 基于工具结果生成最终回答（或继续调用工具）
  5. 超过 max_tool_calls 时强制终止并返回当前最佳回答

核心循环:
  user_query → LLM(tool_calls?) → execute_tools → LLM(tool_calls?) → ... → final_answer

支持:
  - DeepSeek / OpenAI 兼容 API 的 Function Calling
  - Redis 工具结果缓存（避免相同参数重复调用）
  - Mock 模式（无 API Key 时用于开发/测试）
  - 逐步执行日志（每步打印进入的工具名、参数、结果摘要）
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...runtime.llm_client import LLMClient, LLMMessage, LLMResponse
from ..executor.executor import ToolExecutor
from ..registry.registry import ToolRegistry
from ..tool_models import ToolManifest, ToolResult
from .tool_cache import RedisToolCache

logger = logging.getLogger(__name__)


# ============================================================
# 智能体结果数据结构
# ============================================================

@dataclass
class ToolCallRecord:
    """单次工具调用记录（审计用）"""

    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    execution_time_ms: float
    cached: bool = False  # 是否来自缓存
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "cached": self.cached,
            "error": self.error,
        }


@dataclass
class AgentResult:
    """
    智能体运行结果

    Attributes:
        answer: 最终回答文本
        citations: 引用来源列表
        tool_calls_made: 工具调用历史记录
        total_tokens: 总 token 消耗
        is_refusal: 是否拒答
        refusal_reason: 拒答原因
        iterations: 实际迭代轮数
        latency_ms: 总耗时（毫秒）
        model: 使用的模型名称
    """

    answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_made: List[ToolCallRecord] = field(default_factory=list)
    total_tokens: int = 0
    is_refusal: bool = False
    refusal_reason: str = ""
    iterations: int = 0
    latency_ms: float = 0.0
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "tool_calls_made": [tc.to_dict() for tc in self.tool_calls_made],
            "total_tokens": self.total_tokens,
            "is_refusal": self.is_refusal,
            "refusal_reason": self.refusal_reason,
            "iterations": self.iterations,
            "latency_ms": self.latency_ms,
            "model": self.model,
        }


# ============================================================
# 系统提示模板
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """你是一个银行业监管数据智能问答助手。你可以使用工具来检索法规文档和表格数据，并进行数值计算。

## 核心规则
1. **必须使用工具获取数据**：不要凭记忆回答，必须先调用检索工具获取相关法规或数据
2. **数值计算用计算器工具**：涉及加减乘除、百分比、比率等计算时，使用 calculate 工具确保精确
3. **回答必须基于工具返回的证据**：不得编造或添加工具结果中没有的信息
4. **引用来源**：回答中必须标注数据来源（文档名、表格名、单元格位置等）

## 工具使用策略
- **search_chunks**: 首选检索工具，用于搜索法规条款、定义、阈值等内容
- **search_table**: 当问题涉及表格数据时使用（如"XX表格中YY的值"）
- **get_cell_value**: 当需要查询单元格具体数值时使用
- **get_chunk_content**: 当需要查看某个 chunk 的完整内容时使用
- **list_documents**: 当需要了解有哪些文档时使用
- **calculate**: 当需要数值计算时使用（加减乘除、百分比、比率、比较等）

## 回答格式
- 先给出直接答案（一句话回答核心数值或结论）
- 再补充必要的上下文说明（1-2句）
- 最后标注引用来源，格式：来源：《文档名》表格/单元格信息

## 如果证据不足
如果工具检索结果无法回答问题，直接回答"依据不足，未找到相关数据"，不要编造答案。

## 当前可用工具
{tool_descriptions}"""


# ============================================================
# 工具调用智能体
# ============================================================

class ToolCallingAgent:
    """
    工具调用智能体

    编排 LLM 与工具的交互循环，实现 Agentic RAG。

    用法:
        # 创建智能体
        agent = ToolCallingAgent(
            llm_client=LLMClient(),
            tool_executor=executor,  # 已注册所有工具的 ToolExecutor
            max_tool_calls=5,
        )

        # 运行查询
        result = agent.run("2023年10月人身险公司原保险保费收入是多少？")
        print(result.answer)
        print(f"工具调用 {len(result.tool_calls_made)} 次")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        max_tool_calls: int = 5,
        cache: Optional[RedisToolCache] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        Args:
            llm_client: LLM 客户端（支持 Function Calling）
            tool_executor: 工具执行器（已注册所有工具）
            max_tool_calls: 最大工具调用轮数（防止无限循环）
            cache: Redis 工具结果缓存，None 则不缓存
            temperature: LLM 温度参数
            max_tokens: LLM 最大生成 token 数
        """
        self._llm = llm_client
        self._executor = tool_executor
        self._registry: ToolRegistry = tool_executor._registry
        self._max_tool_calls = max_tool_calls
        self._cache = cache
        self._temperature = temperature
        self._max_tokens = max_tokens

        # 预构建 OpenAI 格式的工具定义
        self._tools_def = self._build_tools_definition()
        # 预构建系统提示
        self._system_prompt = self._build_system_prompt()

    # ============================================================
    # 公开方法
    # ============================================================

    def run(
        self,
        query: str,
        session_context: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResult:
        """
        运行工具调用智能体

        Args:
            query: 用户查询
            session_context: 会话上下文（历史对话），格式 [{"role": "user/assistant", "content": "..."}]

        Returns:
            AgentResult 对象
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("工具调用智能体启动 | 查询: %s", query[:100])
        logger.info("最大工具调用轮数: %d | 缓存: %s | LLM Mock: %s",
                     self._max_tool_calls,
                     "启用" if self._cache else "禁用",
                     self._llm.is_mock)
        logger.info("=" * 60)

        # 构建初始消息列表
        messages: List[LLMMessage] = [LLMMessage(role="system", content=self._system_prompt)]

        # 加入会话上下文（历史对话）
        if session_context:
            for ctx in session_context[-6:]:  # 保留最近 3 轮对话
                role = ctx.get("role", "user")
                content = ctx.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append(LLMMessage(role=role, content=content))

        # 加入当前查询
        messages.append(LLMMessage(role="user", content=query))

        # 工具调用记录
        tool_calls_made: List[ToolCallRecord] = []
        total_tokens = 0
        citations: List[Dict[str, Any]] = []
        iterations = 0

        # ============================================================
        # 核心循环: LLM → tool_calls → execute → LLM → ...
        # ============================================================
        for iteration in range(1, self._max_tool_calls + 2):
            iterations = iteration
            logger.info("-" * 40)
            logger.info("[轮次 %d] 调用 LLM (消息数: %d)", iteration, len(messages))

            try:
                response: LLMResponse = self._llm.chat_with_tools(
                    messages=messages,
                    tools=self._tools_def,
                    tool_choice="auto",
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            except Exception as e:
                logger.error("[轮次 %d] LLM 调用失败: %s", iteration, e)
                return AgentResult(
                    answer=f"LLM 调用失败: {e}",
                    is_refusal=True,
                    refusal_reason=f"LLM 调用异常: {type(e).__name__}",
                    tool_calls_made=tool_calls_made,
                    total_tokens=total_tokens,
                    iterations=iterations,
                    latency_ms=(time.time() - start_time) * 1000,
                    model=self._llm.model,
                )

            total_tokens += response.total_tokens
            logger.info("[轮次 %d] LLM 响应: finish_reason=%s, tool_calls=%d, tokens=%d",
                        iteration,
                        response.finish_reason,
                        len(response.tool_calls),
                        response.total_tokens)

            # ── 情况 1: LLM 返回了工具调用 ──
            if response.has_tool_calls:
                # 将 assistant 消息（含 tool_calls）加入对话
                messages.append(LLMMessage(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                ))

                # 逐个执行工具调用
                for tc in response.tool_calls:
                    record = self._execute_tool_call(tc)
                    tool_calls_made.append(record)

                    # 从工具结果中提取引用信息
                    self._extract_citations(record, citations)

                    # 将工具结果作为 tool 消息加入对话
                    tool_result_str = self._format_tool_result_for_llm(record)
                    messages.append(LLMMessage(
                        role="tool",
                        content=tool_result_str,
                        tool_call_id=tc.get("id", ""),
                    ))

                # 继续下一轮（让 LLM 看到工具结果后决定下一步）
                continue

            # ── 情况 2: LLM 返回了最终回答（无 tool_calls）──
            logger.info("[轮次 %d] LLM 返回最终回答", iteration)
            latency_ms = (time.time() - start_time) * 1000

            answer_text = response.content.strip()
            is_refusal = "依据不足" in answer_text or not answer_text

            result = AgentResult(
                answer=answer_text,
                citations=citations,
                tool_calls_made=tool_calls_made,
                total_tokens=total_tokens,
                is_refusal=is_refusal,
                refusal_reason="证据不足" if is_refusal else "",
                iterations=iterations,
                latency_ms=latency_ms,
                model=response.model or self._llm.model,
            )

            logger.info("=" * 60)
            logger.info("智能体完成 | 轮数: %d | 工具调用: %d | 耗时: %.0fms | tokens: %d",
                        iterations, len(tool_calls_made), latency_ms, total_tokens)
            logger.info("回答预览: %s", answer_text[:200])
            logger.info("=" * 60)
            return result

        # ── 情况 3: 超过最大轮数，强制终止 ──
        logger.warning("超过最大工具调用轮数 %d，强制终止", self._max_tool_calls)
        latency_ms = (time.time() - start_time) * 1000

        # 最后一次不带工具的调用，让 LLM 总结已有信息
        try:
            final_response = self._llm.chat(
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            answer_text = final_response.content.strip()
            total_tokens += final_response.total_tokens
        except Exception as e:
            logger.error("最终总结调用失败: %s", e)
            answer_text = "由于查询复杂度较高，已达到最大工具调用次数，无法生成完整回答。"

        return AgentResult(
            answer=answer_text,
            citations=citations,
            tool_calls_made=tool_calls_made,
            total_tokens=total_tokens,
            is_refusal=False,
            refusal_reason="达到最大工具调用次数",
            iterations=iterations,
            latency_ms=latency_ms,
            model=self._llm.model,
        )

    # ============================================================
    # 内部方法
    # ============================================================

    def _build_tools_definition(self) -> List[Dict[str, Any]]:
        """将 ToolManifest 列表转为 OpenAI Function Calling 格式"""
        tools = []
        for manifest in self._registry.list_tools():
            tool_def = {
                "type": "function",
                "function": {
                    "name": manifest.name,
                    "description": manifest.description,
                    "parameters": manifest.input_schema if manifest.input_schema else {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            tools.append(tool_def)
        logger.info("已注册 %d 个工具: %s",
                     len(tools),
                     [t["function"]["name"] for t in tools])
        return tools

    def _build_system_prompt(self) -> str:
        """构建系统提示，包含工具描述"""
        descriptions = []
        for manifest in self._registry.list_tools():
            params = manifest.input_schema.get("properties", {})
            param_str = ", ".join(
                f"{k}: {v.get('description', v.get('type', 'any'))}"
                for k, v in params.items()
            )
            descriptions.append(
                f"- **{manifest.name}**: {manifest.description}\n"
                f"  参数: {param_str}"
            )

        return SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions="\n".join(descriptions)
        )

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> ToolCallRecord:
        """
        执行单个工具调用

        Args:
            tool_call: OpenAI 格式的工具调用
                {"id": "call_xxx", "type": "function",
                 "function": {"name": "...", "arguments": "..."}}

        Returns:
            ToolCallRecord 记录
        """
        func_info = tool_call.get("function", {})
        tool_name = func_info.get("name", "")
        arguments_str = func_info.get("arguments", "{}")

        # 解析参数（LLM 返回的 arguments 是 JSON 字符串）
        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            arguments = {"_raw": arguments_str}

        logger.info("  [工具调用] %s | 参数: %s", tool_name,
                     json.dumps(arguments, ensure_ascii=False)[:200])

        # ── 缓存检查 ──
        if self._cache:
            cached = self._cache.get(tool_name, arguments)
            if cached is not None:
                logger.info("  [缓存命中] %s → 跳过执行", tool_name)
                return ToolCallRecord(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=cached,
                    success=cached.get("success", True) if isinstance(cached, dict) else True,
                    execution_time_ms=0.1,
                    cached=True,
                )

        # ── 执行工具 ──
        start = time.time()
        try:
            result: ToolResult = self._executor.invoke(
                tool_name=tool_name,
                input_data=arguments,
                caller_role="authenticated",
            )
            exec_ms = (time.time() - start) * 1000

            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result=result.data,
                success=result.success,
                execution_time_ms=exec_ms,
                error=result.error,
            )

            # ── 缓存结果 ──
            if self._cache and result.success:
                self._cache.set(tool_name, arguments, result.data)

            logger.info("  [工具结果] %s → success=%s, time=%.0fms",
                        tool_name, result.success, exec_ms)
            if result.success and isinstance(result.data, dict):
                # 打印结果摘要
                summary_keys = ["hit_count", "doc_count", "cell_count", "result", "compliant"]
                summary = {k: result.data.get(k) for k in summary_keys if k in result.data}
                if summary:
                    logger.info("  [结果摘要] %s", json.dumps(summary, ensure_ascii=False))

            return record

        except Exception as e:
            exec_ms = (time.time() - start) * 1000
            logger.error("  [工具异常] %s → %s", tool_name, e)
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                success=False,
                execution_time_ms=exec_ms,
                error=str(e),
            )

    def _format_tool_result_for_llm(self, record: ToolCallRecord) -> str:
        """将工具执行结果格式化为 LLM 可读的字符串"""
        if not record.success:
            return f"工具执行失败: {record.error}"

        result = record.result
        if isinstance(result, dict):
            # 对大型结果截断，避免 token 消耗过大
            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > 4000:
                result_str = result_str[:4000] + "...(结果已截断)"
            return result_str
        return str(result) if result else "工具返回空结果"

    def _extract_citations(
        self,
        record: ToolCallRecord,
        citations: List[Dict[str, Any]],
    ) -> None:
        """从工具结果中提取引用信息"""
        if not record.success or not isinstance(record.result, dict):
            return

        # 从 hits 列表中提取引用
        hits = record.result.get("hits", [])
        for hit in hits[:5]:
            if isinstance(hit, dict):
                citation = {
                    "tool": record.tool_name,
                    "chunk_id": hit.get("chunk_id", ""),
                    "source_doc": hit.get("source_doc", hit.get("doc_name", "")),
                    "content_preview": str(hit.get("content", ""))[:100],
                }
                # 去重
                if citation["chunk_id"] and not any(
                    c.get("chunk_id") == citation["chunk_id"] for c in citations
                ):
                    citations.append(citation)

        # 从 documents 列表中提取
        docs = record.result.get("documents", [])
        for doc in docs[:3]:
            if isinstance(doc, dict):
                citation = {
                    "tool": record.tool_name,
                    "doc_id": doc.get("doc_id", ""),
                    "source_doc": doc.get("doc_name", doc.get("doc_title", "")),
                }
                if citation["doc_id"] and not any(
                    c.get("doc_id") == citation["doc_id"] for c in citations
                ):
                    citations.append(citation)

        # 从 cells 列表中提取
        cells = record.result.get("cells", [])
        for cell in cells[:5]:
            if isinstance(cell, dict):
                citation = {
                    "tool": record.tool_name,
                    "chunk_id": cell.get("chunk_id", ""),
                    "source_doc": cell.get("table_name", ""),
                    "content_preview": str(cell.get("content", ""))[:100],
                }
                if citation["chunk_id"] and not any(
                    c.get("chunk_id") == citation["chunk_id"] for c in citations
                ):
                    citations.append(citation)
