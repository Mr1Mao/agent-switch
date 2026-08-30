"""Agent hooks 的 Context 模型定义。

每个事件 Context 带固定的 ``event: Literal[AgentHookEvent.XXX]`` 及专用字段。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_switch.hooks.enums import AgentHookEvent
from agent_switch.types.config import AgentConfig
from agent_switch.types.message import AgentMessage, ToolCall
from agent_switch.types.response import AgentResponse
from agent_switch.types.subagent import AgentSubagent


class AgentHookContext(BaseModel):
    """hooks Context 基类：所有事件上下文共有的字段。

    - ``backend``：产生该事件的后端名称（如 deepagents / qcoder）；
    - ``session_id`` / ``correlation_id``：本次 run/stream 生成的相关 ID，
      用于把同一次调用的多个事件关联起来；
    - ``timestamp``：事件发生时间（默认当前 UTC 时间）；
    - ``meta``：透传元信息。

    ``extra="forbid"``：事件上下文不允许出现未声明的字段，保证结构可控；
    ``arbitrary_types_allowed``：允许携带任意异常对象（error 字段）等非 pydantic 类型。
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    backend: str
    session_id: str | None = None
    correlation_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    meta: dict[str, Any] = {}


class BeforeAgentHookContext(AgentHookContext):
    """beforeAgent 事件 Context：agent 执行开始前。

    - ``model``：本次使用的模型名（来自 summarize_config 的 model_name key）；
    - ``input_messages``：本次调用的输入消息；
    - ``config``：产生该事件的 AgentConfig（供 hook 读取完整配置）。
    """

    event: Literal[AgentHookEvent.BEFORE_AGENT] = AgentHookEvent.BEFORE_AGENT
    model: str | None = None
    input_messages: list[AgentMessage] = []
    config: AgentConfig | None = None


class AfterAgentHookContext(AgentHookContext):
    """afterAgent 事件 Context：agent 执行结束后。

    - ``response``：成功时的响应；``error``：失败时携带的异常（两者互斥）。
    """

    event: Literal[AgentHookEvent.AFTER_AGENT] = AgentHookEvent.AFTER_AGENT
    response: AgentResponse | None = None
    error: BaseException | None = None


class BeforePromptHookContext(AgentHookContext):
    """beforePrompt 事件 Context：提示词发送前（可拦截事件）。

    - ``messages``：当前消息列表（MODIFY 时用 data["messages"] 替换）；
    - ``system_prompt``：系统提示词。
    """

    event: Literal[AgentHookEvent.BEFORE_PROMPT] = AgentHookEvent.BEFORE_PROMPT
    messages: list[AgentMessage] = []
    system_prompt: str | None = None


class BeforeToolHookContext(AgentHookContext):
    """beforeTool 事件 Context。"""

    event: Literal[AgentHookEvent.BEFORE_TOOL] = AgentHookEvent.BEFORE_TOOL
    tool_name: str
    tool_call: ToolCall | None = None


class AfterToolHookContext(AgentHookContext):
    """afterTool 事件 Context。"""

    event: Literal[AgentHookEvent.AFTER_TOOL] = AgentHookEvent.AFTER_TOOL
    tool_name: str
    result: Any = None


class AfterToolErrorHookContext(AgentHookContext):
    """afterToolError 事件 Context。"""

    event: Literal[AgentHookEvent.AFTER_TOOL_ERROR] = AgentHookEvent.AFTER_TOOL_ERROR
    tool_name: str
    error: BaseException | None = None


class BeforePermissionHookContext(AgentHookContext):
    """beforePermission 事件 Context。"""

    event: Literal[AgentHookEvent.BEFORE_PERMISSION] = AgentHookEvent.BEFORE_PERMISSION
    action: str
    details: dict[str, Any] = {}


class AfterStopHookContext(AgentHookContext):
    """afterStop 事件 Context。"""

    event: Literal[AgentHookEvent.AFTER_STOP] = AgentHookEvent.AFTER_STOP
    reason: str = "complete"
    response: AgentResponse | None = None
    error: BaseException | None = None


class BeforeSubagentHookContext(AgentHookContext):
    """beforeSubagent 事件 Context。"""

    event: Literal[AgentHookEvent.BEFORE_SUBAGENT] = AgentHookEvent.BEFORE_SUBAGENT
    subagent: AgentSubagent | None = None
    task: str = ""


class AfterSubagentHookContext(AgentHookContext):
    """afterSubagent 事件 Context。"""

    event: Literal[AgentHookEvent.AFTER_SUBAGENT] = AgentHookEvent.AFTER_SUBAGENT
    subagent: AgentSubagent | None = None
    result: Any = None


class BeforeLLMHookContext(AgentHookContext):
    """beforeLLM 事件 Context。"""

    event: Literal[AgentHookEvent.BEFORE_LLM] = AgentHookEvent.BEFORE_LLM
    model: str | None = None
    messages: list[AgentMessage] = []


class AfterLLMHookContext(AgentHookContext):
    """afterLLM 事件 Context。"""

    event: Literal[AgentHookEvent.AFTER_LLM] = AgentHookEvent.AFTER_LLM
    response: AgentResponse | None = None
    content: str = ""
    error: BaseException | None = None


#: 事件 → Context 类型映射
HookContextMap: dict[AgentHookEvent, type[AgentHookContext]] = {
    AgentHookEvent.BEFORE_AGENT: BeforeAgentHookContext,
    AgentHookEvent.AFTER_AGENT: AfterAgentHookContext,
    AgentHookEvent.BEFORE_PROMPT: BeforePromptHookContext,
    AgentHookEvent.BEFORE_TOOL: BeforeToolHookContext,
    AgentHookEvent.AFTER_TOOL: AfterToolHookContext,
    AgentHookEvent.AFTER_TOOL_ERROR: AfterToolErrorHookContext,
    AgentHookEvent.BEFORE_PERMISSION: BeforePermissionHookContext,
    AgentHookEvent.AFTER_STOP: AfterStopHookContext,
    AgentHookEvent.BEFORE_SUBAGENT: BeforeSubagentHookContext,
    AgentHookEvent.AFTER_SUBAGENT: AfterSubagentHookContext,
    AgentHookEvent.BEFORE_LLM: BeforeLLMHookContext,
    AgentHookEvent.AFTER_LLM: AfterLLMHookContext,
}
