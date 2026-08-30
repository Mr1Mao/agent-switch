"""qcoder 后端适配器（真实实现，基于 qoder-agent-sdk）。

- ``run``：同步包装 —— 前置 hooks（adapter 层）→ ``asyncio.run`` 驱动异步 SDK
  ``query()`` → 收集消息直到 ``ResultMessage`` 终结 → 收尾 hooks。
- ``stream``：全异步 —— 逐条映射 SDK 消息为 AgentChunk，最后 yield ``is_finish``。
- hooks 分层：会话级事件（beforeAgent / beforePrompt / beforeLLM / afterLLM /
  afterAgent / afterStop）由 adapter 层触发；调用级事件（beforeTool / afterTool /
  afterToolError / beforePermission / beforeSubagent / afterSubagent）经
  ``build_qoder_hooks`` 桥接到 QoderAgentOptions.hooks，在 qoder CLI 内部触发。

限制：``run()`` 内部使用 ``asyncio.run``，因此在运行中的事件循环里调用同步
``run()`` 会抛 ``RuntimeError`` —— 异步场景请使用 ``stream()``。
真实运行要求本机安装 ``qodercli`` CLI 并登录（``qodercli auth``）。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from agent_core.adapter_base import BaseAgentAdapter
from agent_core.backends.qcoder.hooks_bridge import build_qoder_hooks
from agent_core.backends.qcoder.mapping import (
    build_qoder_agent_options,
    qoder_message_to_agent_chunks,
    qoder_message_to_agent_message,
    qoder_message_to_agent_response,
    qoder_wire_iter,
)
from agent_core.logging import get_logger
from agent_core.types import AgentChunk, AgentConfig, AgentMessage, AgentResponse, MessageRole

_logger = get_logger("agent_core.backends.qcoder")


class QcoderAdapter(BaseAgentAdapter):
    """基于 qoder-agent-sdk 的真实适配器。"""

    backend_name = "qcoder"

    def _query_impl(self) -> Any:
        """返回 SDK 模块级 ``query`` 函数（测试可 monkeypatch 此 seam）。"""
        from qoder_agent_sdk import query  # 延迟导入：import agent_core 不加载 SDK

        return query

    def _build_options(self, config: AgentConfig | None, session_id: str | None = None) -> Any:
        """构建 QoderAgentOptions；配置了 hooks 时注入调用级桥接。"""
        dispatcher = self._resolve_hooks_dispatcher(config)
        hooks = None
        if dispatcher is not None:
            hooks = build_qoder_hooks(
                dispatcher,
                backend=self.backend_name,
                session_provider=lambda: (self._session_id, self._correlation_id),
            )
        if config is None:
            return None
        return build_qoder_agent_options(config, hooks=hooks, session_id=session_id)

    def run(self, input: str | list[AgentMessage], config: AgentConfig | None = None) -> AgentResponse:
        """run 主流程：前置 hooks（adapter 层）→ SDK query() → 收尾 hooks。"""
        resolved = self._resolve_config(config)
        try:
            messages = self._prepare_messages_sync(input, resolved)
            messages = self._emit_before_llm_sync(messages, resolved)
            response = asyncio.run(self._run_async(messages, resolved))
            self._finalize_run_success_sync(resolved, response)
            return response
        except Exception as exc:
            self._finalize_run_error_sync(resolved, exc)
            raise

    async def _run_async(self, messages: list[AgentMessage], resolved: AgentConfig | None) -> AgentResponse:
        """异步执行 SDK query()，收集回复直到 ResultMessage 终结。"""
        options = self._build_options(resolved, session_id=self._session_id)
        collected: list[str] = []
        terminal_raw: Any = None
        async for message in self._query_impl()(prompt=qoder_wire_iter(messages), options=options):
            agent_message = qoder_message_to_agent_message(message)
            if agent_message is not None and agent_message.role is MessageRole.ASSISTANT:
                collected.append(agent_message.content)
            response = qoder_message_to_agent_response(message)
            if response is not None:
                terminal_raw = message
                # ResultMessage.result 缺失时回退为累计的 assistant 文本
                content = response.content or "".join(collected)
                return AgentResponse(content=content, raw=terminal_raw, backend=self.backend_name)
        # 流结束仍未收到 ResultMessage（异常终止）：用累计文本兜底
        return AgentResponse(content="".join(collected), raw=terminal_raw, backend=self.backend_name)

    async def stream(self, input: str | list[AgentMessage], config: AgentConfig | None = None) -> AsyncIterator[AgentChunk]:
        """stream 主流程：前置 hooks（async）→ query() 逐条映射 → is_finish → 收尾。"""
        resolved = self._resolve_config(config)
        messages = await self._prepare_messages_async(input, resolved)
        messages = await self._emit_before_llm_async(messages, resolved)
        options = self._build_options(resolved, session_id=self._session_id)
        final_content = ""
        terminal: AgentResponse | None = None
        async for message in self._query_impl()(prompt=qoder_wire_iter(messages), options=options):
            for agent_chunk in qoder_message_to_agent_chunks(message):
                final_content += agent_chunk.delta_content
                yield agent_chunk
            response = qoder_message_to_agent_response(message)
            if response is not None:
                terminal = response
                yield AgentChunk(is_finish=True)
                break
        if terminal is None:
            # 未收到终结消息：仍按约定 yield is_finish，保证消费方一定能看到结束分片
            yield AgentChunk(is_finish=True)
        response = terminal or self._build_stream_response(final_content)
        await self._finalize_stream_success_async(resolved, response)
