"""stub 后端适配器（提供确定性输出，供测试与示例使用）。"""

from __future__ import annotations

from typing import AsyncIterator

from agent_core.adapter_base import BaseAgentAdapter
from agent_core.types import AgentChunk, AgentMessage, AgentResponse, MessageRole, ToolCall
from agent_core.utils.input import format_input_preview


class StubAgentAdapter(BaseAgentAdapter):
    """stub 适配器：返回固定的 ``[stub]`` 回复，行为完全可预测。"""

    backend_name = "stub"

    def _stub_response(self, messages: list[AgentMessage]) -> AgentResponse:
        """构造带 ``[stub]`` 标记的 AgentResponse，含 thinking、tool_calls 与 raw 快照。"""
        preview = format_input_preview(messages)
        content = f"[stub] {preview}"
        response_message = AgentMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            thinking="[stub] thinking...",
            tool_calls=[
                ToolCall(id="stub_echo", name="stub_echo", arguments={"echo": preview})
            ],
        )
        raw_snapshot: dict[str, object] = {
            "count": len(messages),
            "preview": preview,
            "messages": [message.to_dict() for message in messages],
        }
        return AgentResponse(
            content=content,
            message=response_message,
            raw=raw_snapshot,
            backend=self.backend_name,
        )

    async def _stub_stream(self, messages: list[AgentMessage]) -> AsyncIterator[AgentChunk]:
        """yield 思考 → 正文 ×2 → tool_call → is_finish（共 5 条 chunk，含 finish）。"""
        preview = format_input_preview(messages)
        yield AgentChunk(delta_thinking="[stub] thinking...")
        yield AgentChunk(delta_content=f"[stub] {preview}")
        yield AgentChunk(delta_content=" [stub] done")
        yield AgentChunk(
            delta_tool_call=ToolCall(
                id="stub_echo", name="stub_echo", arguments={"echo": preview}
            )
        )
        yield AgentChunk(is_finish=True)
