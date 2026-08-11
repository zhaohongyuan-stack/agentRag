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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    """对话消息"""

    role: str  # system / user / assistant
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """LLM 响应"""

    content: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


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
    ):
        """
        Args:
            api_base: API 基础 URL
            api_key: API Key
            model: 主模型名称（回答生成）
            small_model: 小模型名称（意图识别、改写等）
            mock: 是否强制 Mock 模式，None 时自动判断
        """
        self._api_base = api_base or os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
        self._api_key = api_key or os.getenv("LLM_API_KEY", "")
        self._model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self._small_model = small_model or os.getenv("LLM_SMALL_MODEL", "deepseek-chat")

        # Mock 模式判断
        if mock is not None:
            self._mock = mock
        else:
            self._mock = not self._api_key or self._api_key == "mock"

        # ── 初始化底层客户端 ──
        # 优先 openai SDK；不可用时回退 httpx 直连
        self._backend = "none"  # "sdk" | "httpx" | "none"(mock)
        self._sdk_client = None
        self._httpx_client = None

        if not self._mock:
            self._init_backend()

        if self._mock:
            logger.info("LLM 客户端运行在 Mock 模式（无真实 API 调用）")

    def _init_backend(self) -> None:
        """初始化底层调用后端"""
        # 尝试 openai SDK
        try:
            from openai import OpenAI

            self._sdk_client = OpenAI(
                base_url=self._api_base,
                api_key=self._api_key,
            )
            self._backend = "sdk"
            logger.debug("LLM 后端: openai SDK")
            return
        except ImportError:
            logger.debug("openai 库未安装，尝试 httpx 后端")

        # 回退 httpx
        try:
            import httpx

            # 禁用代理，避免本地代理导致连接失败
            self._httpx_client = httpx.Client(
                timeout=60.0,
                proxy=None,  # 显式禁用代理
                verify=True,
            )
            self._backend = "httpx"
            logger.debug("LLM 后端: httpx 直连")
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
            **kwargs: 其他 OpenAI API 参数

        Returns:
            LLMResponse 对象
        """
        use_model = model or self._model

        if self._mock:
            return self._mock_chat(messages, use_model, temperature, max_tokens)

        # 构建请求体
        api_kwargs: Dict[str, Any] = {
            "model": use_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            api_kwargs["response_format"] = response_format
        api_kwargs.update(kwargs)

        # 按后端分发
        try:
            if self._backend == "sdk":
                return self._chat_via_sdk(api_kwargs)
            elif self._backend == "httpx":
                return self._chat_via_httpx(api_kwargs)
            else:
                # 不应到达此处
                return self._mock_chat(messages, use_model, temperature, max_tokens)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}", exc_info=True)
            raise

    def _chat_via_sdk(self, api_kwargs: Dict[str, Any]) -> LLMResponse:
        """通过 openai SDK 调用"""
        response = self._sdk_client.chat.completions.create(**api_kwargs)
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            raw=response,
        )

    def _chat_via_httpx(self, api_kwargs: Dict[str, Any]) -> LLMResponse:
        """通过 httpx 直连 OpenAI 兼容 API"""
        url = f"{self._api_base.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        resp = self._httpx_client.post(url, json=api_kwargs, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", api_kwargs["model"]),
            usage={
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": data.get("usage", {}).get("total_tokens", 0),
            },
            raw=data,
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

    # ============================================================
    # Mock 实现
    # ============================================================

    def _mock_chat(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Mock 聊天 — 基于消息内容返回预设响应"""

        # 提取用户最后一条消息
        user_content = ""
        system_content = ""
        for msg in messages:
            if msg.role == "user":
                user_content = msg.content
            elif msg.role == "system":
                system_content = msg.content

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
