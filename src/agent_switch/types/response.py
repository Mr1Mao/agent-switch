"""AgentChunk / AgentResponse：统一的输出模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_switch.types.message import AgentMessage, ToolCall


class AgentChunk(BaseModel):
    """流式分片（不是完整消息）。"""

    model_config = ConfigDict(extra="forbid")

    delta_content: str = ""
    delta_thinking: str = ""
    delta_tool_call: ToolCall | None = None
    is_finish: bool = False
    meta: dict[str, Any] = {}


class AgentResponse(BaseModel):
    """统一的 Agent 运行结果。

    - ``content``：快捷文本（通常等于 message.content）。
    - ``message``：完整消息（含 thinking / tool_calls 等）。
    - ``raw``：后端原始结果（调试用）。
    - ``backend``：产生该响应的后端名称。
    """

    model_config = ConfigDict(extra="forbid")

    content: str
    message: AgentMessage | None = None
    raw: Any = None
    backend: str
