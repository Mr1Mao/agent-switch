"""Qcoder 后端适配器（stub 实现，继承 StubAgentAdapter）。"""

from __future__ import annotations

from typing import AsyncIterator

from agent_core.backends.stub import StubAgentAdapter
from agent_core.types import AgentChunk, AgentConfig, AgentMessage, AgentResponse


class QcoderAdapter(StubAgentAdapter):
    """qcoder 后端（当前为 stub，用于验证 hooks 生命周期与流式行为）。"""

    backend_name = "qcoder"

    def run(self, input: str | list[AgentMessage], config: AgentConfig | None = None) -> AgentResponse:
        """run 主流程：beforeAgent → beforePrompt → beforeLLM → stub → afterLLM → afterStop。"""
        resolved = self._resolve_config(config)
        try:
            messages = self._prepare_messages_sync(input, resolved)
            messages = self._emit_before_llm_sync(messages, resolved)
            response = self._stub_response(messages)
            self._finalize_run_success_sync(resolved, response)
            return response
        except Exception as exc:
            self._finalize_run_error_sync(resolved, exc)
            raise

    async def stream(self, input: str | list[AgentMessage], config: AgentConfig | None = None) -> AsyncIterator[AgentChunk]:
        """stream 主流程：hooks + _stub_stream；流结束后 finalize（stub 自身 yield is_finish）。"""
        resolved = self._resolve_config(config)
        messages = await self._prepare_messages_async(input, resolved)
        messages = await self._emit_before_llm_async(messages, resolved)
        final_content = ""
        async for chunk in self._stub_stream(messages):
            final_content += chunk.delta_content
            yield chunk
        response = self._build_stream_response(final_content)
        await self._finalize_stream_success_async(resolved, response)
