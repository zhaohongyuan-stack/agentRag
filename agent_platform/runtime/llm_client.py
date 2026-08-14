"""
LLM 客户端抽象层

统一封装 OpenAI 兼容 API 调用，支持:
  - 真实模式（SDK）: 通过 OpenAI SDK 调用 DeepSeek / OpenAI / 阿里云等
  - 真实模式（httpx）: openai 库不可用时，通过 httpx 直连 API
  - Mock 模式: 无 API Key 时返回预设响应，用于开发和测试

配置来源:
  - 环境变量 LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_SMALL_MODEL
  - 或通过构造函数显式传入
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    """对话消息

    支持的 role:
      - system: 系统提示
      - user: 用户消息
      - assistant: 助手回复（可能携带 tool_calls）
      - tool: 工具执行结果（需附带 tool_call_id 关联对应调用）
    """

    role: str  # system / user / assistant / tool
    content: str
    # assistant 消息中的工具调用（OpenAI 兼容格式）
    # [{"id": "call_xxx", "type": "function", "function": {"name": "...", "arguments": "..."}}]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # tool 角色消息的关联 ID（对应 assistant 消息中某个 tool_call 的 id）
    tool_call_id: Optional[str] = None

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class LLMResponse:
    """LLM 响应"""

    content: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None
    # 工具调用列表（OpenAI 兼容格式），无调用时为空列表
    # [{"id": "call_xxx", "type": "function", "function": {"name": "...", "arguments": "..."}}]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # 结束原因: "stop" | "tool_calls" | "length" | ""
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def has_tool_calls(self) -> bool:
        """响应中是否包含工具调用"""
        return len(self.tool_calls) > 0


class KeyPool:
    """
    API Key 池 — 轮询 + 故障转移

    支持多个 API Key 的负载均衡:
      - round-robin 轮询分配
      - 失败次数过多的 Key 暂时禁用（冷却）
      - 全部 Key 都失败时抛出异常
    """

    def __init__(self, keys: List[str]):
        # 去重、去空
        self._keys = [k.strip() for k in keys if k and k.strip()]
        if not self._keys:
            self._keys = ["mock"]
        self._index = 0
        self._lock = threading.Lock()
        self._fail_counts: Dict[str, int] = {}
        self._disabled_until: Dict[str, float] = {}
        self._max_fails = 3          # 连续失败 N 次后禁用
        self._cooldown_s = 30.0      # 冷却时间（秒）
        self._rotate_on_fail = True  # 失败时轮换到下一个 Key

    def _now(self) -> float:
        import time
        return time.monotonic()

    def _is_available(self, key: str) -> bool:
        """检查 Key 是否可用（未被冷却禁用）"""
        until = self._disabled_until.get(key, 0)
        if until and self._now() < until:
            return False
        if until and self._now() >= until:
            # 冷却结束，重置失败计数
            self._disabled_until.pop(key, None)
            self._fail_counts.pop(key, None)
        return True

    def next_key(self) -> str:
        """获取下一个可用 Key（round-robin）"""
        with self._lock:
            if len(self._keys) == 1:
                return self._keys[0]
            for _ in range(len(self._keys)):
                key = self._keys[self._index % len(self._keys)]
                self._index += 1
                if self._is_available(key):
                    return key
            # 全部冷却 → 返回当前索引的 Key（降级可用但不保证成功）
            return self._keys[self._index % len(self._keys)]

    def record_success(self, key: str):
        """记录成功调用，重置失败计数"""
        with self._lock:
            self._fail_counts.pop(key, None)

    def record_failure(self, key: str):
        """记录失败调用，超过阈值则暂时禁用"""
        with self._lock:
            count = self._fail_counts.get(key, 0) + 1
            self._fail_counts[key] = count
            if count >= self._max_fails:
                self._disabled_until[key] = self._now() + self._cooldown_s
                logger.warning(
                    "[KeyPool] Key 连续失败 %d 次，冷却 %ds: %s...",
                    count, self._cooldown_s, key[:8],
                )

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return f"KeyPool(count={len(self._keys)}, disabled={len(self._disabled_until)})"


class LLMClient:
    """
    LLM 客户端 — OpenAI 兼容 API 调用

    优先使用 openai SDK；若不可用则回退到 httpx 直连。
    两者功能等价，仅传输层不同。

    用法:
        client = LLMClient()  # 自动从环境变量读取配置
        response = client.chat(
            messages=[LLMMessage(role="user", content="你好")],
            model="deepseek-chat",
        )

    Mock 模式:
        当 LLM_API_KEY 未设置或为 "mock" 时，自动启用 Mock 模式，
        返回基于关键词的预设响应，不发起真实网络请求。
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        small_model: Optional[str] = None,
        mock: Optional[bool] = None,
        api_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            api_base: API 基础 URL
            api_key: API Key（单个，向后兼容）
            model: 主模型名称（回答生成）
            small_model: 小模型名称（意图识别、改写等）
            mock: 是否强制 Mock 模式，None 时自动判断
            api_keys: 多 API Key 列表（优先级高于 api_key/LLM_API_KEY）
        """
        self._api_base = api_base or os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
        # 多 Key 配置来源优先级: api_keys 参数 > LLM_API_KEYS 环境变量 > api_key/LLM_API_KEY
        env_keys = os.getenv("LLM_API_KEYS", "")
        key_list = []
        if api_keys:
            key_list = list(api_keys)
        elif env_keys:
            key_list = [k.strip() for k in env_keys.split(",") if k.strip()]
        else:
            single_key = api_key or os.getenv("LLM_API_KEY", "")
            key_list = [single_key] if single_key else []

        self._api_key = api_key or os.getenv("LLM_API_KEY", "")
        self._key_pool = KeyPool(key_list)
        # 当前使用的 Key（随轮询更新）
        self._current_key = self._key_pool.next_key() if not self._is_all_mock() else ""
        self._model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self._small_model = small_model or os.getenv("LLM_SMALL_MODEL", "deepseek-chat")

        # Mock 模式判断
        if mock is not None:
            self._mock = mock
        else:
            self._mock = not self._api_key and not env_keys
        # 兼容: 单 Key 为 "mock" 时强制 Mock
        if self._api_key == "mock":
            self._mock = True

        # ── 初始化底层客户端 ──
        # 优先 openai SDK；不可用时回退 httpx 直连
        self._backend = "none"  # "sdk" | "httpx" | "none"(mock)
        self._sdk_clients: Dict[str, Any] = {}
        self._httpx_client = None

        if not self._mock:
            self._init_backend()

        if self._mock:
            logger.info("LLM 客户端运行在 Mock 模式（无真实 API 调用）")
        else:
            logger.info(
                "LLM 客户端: backend=%s, keys=%d, model=%s",
                self._backend, self._key_pool.key_count, self._model,
            )

    def _is_all_mock(self) -> bool:
        """所有 Key 是否都是 mock（无真实调用）"""
        return all(k == "mock" for k in self._key_pool._keys)

    def _init_backend(self) -> None:
        """初始化底层调用后端（支持多 Key）"""
        # 尝试 openai SDK
        try:
            from openai import OpenAI

            for key in self._key_pool._keys:
                if key == "mock":
                    continue
                self._sdk_clients[key] = OpenAI(
                    base_url=self._api_base,
                    api_key=key,
                    timeout=35.0,    # 35 秒超时（避免LLM调用堆积，DeepSeek V4 Flash 正常3-8s）
                    max_retries=0,   # 关闭 SDK 自动重试（由 KeyPool 轮询接管，避免 3x 等待）
                )
            if self._sdk_clients:
                self._backend = "sdk"
                logger.debug("LLM 后端: openai SDK (timeout=60s, retries=2, keys=%d)", len(self._sdk_clients))
                return
        except ImportError:
            logger.debug("openai 库未安装，尝试 httpx 后端")

        # 回退 httpx
        try:
            import httpx

            # 禁用代理，避免本地代理导致连接失败
            self._httpx_client = httpx.Client(
                timeout=35.0,
                proxy=None,  # 显式禁用代理
                verify=True,
            )
            self._backend = "httpx"
            logger.debug("LLM 后端: httpx 直连（多 Key 轮询）")
        except ImportError:
            logger.warning("openai 和 httpx 均不可用，回退到 Mock 模式")
            self._mock = True
            self._backend = "none"

    @property
    def is_mock(self) -> bool:
        return self._mock

    @property
    def model(self) -> str:
        return self._model

    @property
    def small_model(self) -> str:
        return self._small_model

    @property
    def backend(self) -> str:
        """当前使用的后端: sdk / httpx / none(mock)"""
        return self._backend

    def chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        同步聊天调用

        Args:
            messages: 消息列表
            model: 模型名称，None 时使用默认模型
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            response_format: 响应格式（如 {"type": "json_object"}）
            tools: 可用工具定义列表（OpenAI Function Calling 格式），
                每项形如 {"type": "function", "function": {"name", "description", "parameters"}}
            tool_choice: 工具选择策略，可选值:
                - "auto": 模型自行决定（默认）
                - "none": 禁止调用工具
                - {"type": "function", "function": {"name": "xxx"}}: 强制调用指定工具
            **kwargs: 其他 OpenAI API 参数

        Returns:
            LLMResponse 对象（可能包含 tool_calls）
        """
        use_model = model or self._model

        if self._mock:
            return self._mock_chat(
                messages, use_model, temperature, max_tokens, tools=tools
            )

        # 构建请求体
        api_kwargs: Dict[str, Any] = {
            "model": use_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            api_kwargs["response_format"] = response_format
        if tools:
            api_kwargs["tools"] = tools
            # tool_choice 默认为 "auto"，仅在显式传入时设置
            if tool_choice is not None:
                api_kwargs["tool_choice"] = tool_choice
        api_kwargs.update(kwargs)

        # 按后端分发（多 Key 轮询 + 故障转移）
        last_error: Optional[Exception] = None
        attempts = 0
        while attempts < max(1, self._key_pool.key_count):
            attempts += 1
            key = self._key_pool.next_key()
            try:
                if self._backend == "sdk":
                    result = self._chat_via_sdk(api_kwargs, key)
                elif self._backend == "httpx":
                    result = self._chat_via_httpx(api_kwargs, key)
                else:
                    # 不应到达此处
                    return self._mock_chat(
                        messages, use_model, temperature, max_tokens, tools=tools
                    )
                self._key_pool.record_success(key)
                return result
            except Exception as e:
                last_error = e
                self._key_pool.record_failure(key)
                # 是否 429 限流错误
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                if attempts >= self._key_pool.key_count or not is_rate_limit:
                    break  # 已尝试所有 Key，或非限流错误
                logger.warning(
                    "[LLMClient] Key 调用失败，轮换重试: %s (%d/%d)",
                    key[:8], attempts, self._key_pool.key_count,
                )

        logger.error(f"LLM 调用失败（所有 Key）: {last_error}", exc_info=True)
        raise last_error

    def chat_stream(
        self,
        messages: List[LLMMessage],
        use_model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs,
    ):
        """
        流式调用 LLM，逐 token yield。
        失败时抛异常（由调用方捕获回退非流式）。
        不支持 KeyPool 故障转移（流式已开始输出无法切换）。
        """
        model = use_model or self._model
        key = self._key_pool.next_key()
        sdk = self._sdk_clients.get(key)
        if sdk is None:
            raise ValueError(f"SDK client 未初始化: {key[:8]}...")

        api_kwargs = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        api_kwargs.update(kwargs)

        logger.debug(
            "[LLMClient] chat_stream → model=%s, key=%s..., msgs=%d",
            model, key[:8], len(messages),
        )

        response = sdk.chat.completions.create(**api_kwargs)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                # DeepSeek reasoning_content (思维链字段)
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning:
                    yield ("thinking", reasoning)
                if delta.content:
                    yield ("token", delta.content)

    def _chat_via_sdk(self, api_kwargs: Dict[str, Any], key: str) -> LLMResponse:
        """通过 openai SDK 调用（按 Key 选择客户端）"""
        sdk = self._sdk_clients.get(key)
        if sdk is None:
            raise ValueError(f"SDK client 未初始化: {key[:8]}...")
        response = sdk.chat.completions.create(**api_kwargs)
        choice = response.choices[0]
        message = choice.message

        # 提取 tool_calls（SDK 对象需转为可序列化 dict）
        tool_calls: List[Dict[str, Any]] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        return LLMResponse(
            content=message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            raw=response,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
        )

    def _chat_via_httpx(self, api_kwargs: Dict[str, Any], key: str) -> LLMResponse:
        """通过 httpx 直连 OpenAI 兼容 API（按 Key 轮换）"""
        url = f"{self._api_base.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

        resp = self._httpx_client.post(url, json=api_kwargs, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        message = choice.get("message", {})

        # 提取 tool_calls（OpenAI 兼容格式，直接透传 dict）
        tool_calls: List[Dict[str, Any]] = message.get("tool_calls") or []

        return LLMResponse(
            content=message.get("content") or "",
            model=data.get("model", api_kwargs["model"]),
            usage={
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": data.get("usage", {}).get("total_tokens", 0),
            },
            raw=data,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "") or "",
        )

    def chat_json(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        调用 LLM 并解析 JSON 响应

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数（JSON 输出建议低温度）
            max_tokens: 最大 token 数

        Returns:
            解析后的 JSON 字典

        Raises:
            ValueError: JSON 解析失败
        """
        response = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if not self._mock else None,
        )

        try:
            # 尝试直接解析
            return json.loads(response.content)
        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            content = response.content.strip()
            if "```json" in content:
                start = content.index("```json") + 7
                end = content.index("```", start)
                return json.loads(content[start:end].strip())
            elif "```" in content:
                start = content.index("```") + 3
                end = content.index("```", start)
                return json.loads(content[start:end].strip())
            else:
                raise ValueError(f"无法解析 LLM 响应为 JSON: {response.content[:200]}")

    def chat_with_tools(
        self,
        messages: List[LLMMessage],
        tools: List[Dict[str, Any]],
        tool_choice: Any = "auto",
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        带工具定义的聊天调用（单轮，不含循环）

        向 LLM 传入可用工具定义，返回可能包含 tool_calls 的响应。
        工具执行与多轮循环由调用方负责处理。

        Args:
            messages: 消息列表（含历史上下文）
            tools: 可用工具定义列表（OpenAI Function Calling 格式）
            tool_choice: 工具选择策略，默认 "auto"
            model: 模型名称，None 时使用默认模型
            temperature: 温度参数
            max_tokens: 最大生成 token 数

        Returns:
            LLMResponse，检查 has_tool_calls 判断是否需要执行工具；
            若 has_tool_calls 为 True，则 tool_calls 字段包含调用详情。
        """
        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )

    # ============================================================
    # Mock 实现
    # ============================================================

    def _mock_chat(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Mock 聊天 — 基于消息内容返回预设响应

        当传入 tools 时，模拟一次工具调用并返回 tool_calls 响应，
        便于在没有真实 API 的环境下测试 Function Calling 流程。
        """

        # 提取用户最后一条消息
        user_content = ""
        system_content = ""
        for msg in messages:
            if msg.role == "user":
                user_content = msg.content
            elif msg.role == "system":
                system_content = msg.content

        # ── 工具调用模拟分支 ──
        # 当传入工具定义，且系统提示或用户消息暗示需要调用工具时，
        # 返回模拟的 tool_calls 响应（finish_reason="tool_calls"）
        if tools:
            tool_call = self._mock_tool_call(tools, user_content, system_content)
            if tool_call is not None:
                mock_content = ""
                usage = {
                    "prompt_tokens": len(user_content) // 4 + 10,
                    "completion_tokens": 20,
                    "total_tokens": len(user_content) // 4 + 30,
                }
                return LLMResponse(
                    content=mock_content,
                    model=f"{model} (mock)",
                    usage=usage,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                )

        # 根据系统提示判断任务类型
        if "改写" in system_content or "rewrite" in system_content.lower():
            mock_content = self._mock_rewrite(user_content, system_content)
        elif "引用" in system_content or "回答" in system_content or "answer" in system_content.lower():
            mock_content = self._mock_answer(user_content, system_content, messages)
        else:
            mock_content = f"这是一条 Mock 响应。用户输入: {user_content[:200]}"

        return LLMResponse(
            content=mock_content,
            model=f"{model} (mock)",
            usage={
                "prompt_tokens": len(user_content) // 4 + 10,
                "completion_tokens": len(mock_content) // 4 + 10,
                "total_tokens": (len(user_content) + len(mock_content)) // 4 + 20,
            },
        )

    def _mock_tool_call(
        self,
        tools: List[Dict[str, Any]],
        user_content: str,
        system_content: str,
    ) -> Optional[Dict[str, Any]]:
        """构造模拟的 tool_call 响应

        选取第一个工具进行调用，arguments 基于用户输入构造。
        若工具列表为空则返回 None。

        Returns:
            OpenAI 兼容的 tool_call dict，或 None
        """
        if not tools:
            return None

        # 取第一个工具定义
        first_tool = tools[0]
        func = first_tool.get("function", {})
        func_name = func.get("name", "unknown_tool")

        # 基于 parameters 构造参数（尽可能填充用户输入）
        parameters = func.get("parameters", {})
        props = parameters.get("properties", {})
        required = parameters.get("required", [])

        args: Dict[str, Any] = {}
        for param_name in required:
            param_schema = props.get(param_name, {})
            param_type = param_schema.get("type", "string")
            if param_type == "string":
                args[param_name] = user_content[:200] if user_content else "mock_value"
            elif param_type == "integer":
                args[param_name] = 1
            elif param_type == "number":
                args[param_name] = 1.0
            elif param_type == "boolean":
                args[param_name] = True
            elif param_type == "array":
                args[param_name] = []
            elif param_type == "object":
                args[param_name] = {}
            else:
                args[param_name] = "mock_value"

        return {
            "id": "call_mock_0001",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }

    def _mock_rewrite(self, user_content: str, system_content: str) -> str:
        """Mock 查询改写"""
        # 尝试解析输入 JSON
        try:
            data = json.loads(user_content)
            original = data.get("original_query", user_content)
        except (json.JSONDecodeError, TypeError):
            original = user_content

        result = {
            "original_query": original,
            "contextualized_query": original,
            "channel_queries": {
                "lexical": original,
                "dense": original,
                "exact": original,
            },
            "rewrites": [original],
            "ambiguity_flagged": False,
            "ambiguity_reason": "",
        }
        return json.dumps(result, ensure_ascii=False)

    def _mock_answer(self, user_content: str, system_content: str, messages: List[LLMMessage]) -> str:
        """Mock 回答生成 — 从证据中提取关键信息"""

        # 尝试解析输入
        try:
            data = json.loads(user_content)
            question = data.get("question", user_content)
            evidence_items = data.get("evidence_items", [])
        except (json.JSONDecodeError, TypeError):
            question = user_content
            evidence_items = []

        if not evidence_items:
            return json.dumps({
                "answer": "依据不足，未找到充分的相关法规依据。",
                "citations": [],
                "confidence": 0.0,
                "is_refusal": True,
            }, ensure_ascii=False)

        # 构建基于证据的回答
        snippets = []
        citations = []
        for i, ev in enumerate(evidence_items[:5], 1):
            snippet = ev.get("evidence_snippet", ev.get("content", ""))[:200]
            citation = ev.get("citation", f"来源{i}")
            snippets.append(f"[{i}] {snippet}")
            citations.append({
                "index": i,
                "citation": citation,
                "source_doc": ev.get("source_doc", ""),
                "hierarchy_path": ev.get("hierarchy_path", ""),
            })

        answer_text = f"根据检索到的法规资料：\n\n" + "\n\n".join(snippets)

        return json.dumps({
            "answer": answer_text,
            "citations": citations,
            "confidence": 0.75,
            "is_refusal": False,
        }, ensure_ascii=False)

    def close(self) -> None:
        """关闭底层连接"""
        if self._httpx_client:
            self._httpx_client.close()
            self._httpx_client = None


# ============================================================
# 全局单例
# ============================================================

_global_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端单例"""
    global _global_client
    if _global_client is None:
        _global_client = LLMClient()
    return _global_client


def reset_llm_client():
    """重置全局 LLM 客户端（用于测试）"""
    global _global_client
    if _global_client is not None:
        _global_client.close()
    _global_client = None
