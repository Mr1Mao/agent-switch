"""Agent hooks 事件枚举。"""

from __future__ import annotations

from enum import Enum


class AgentHookEvent(str, Enum):
    """Hook 事件（value 为 camelCase 短名）。"""

    BEFORE_AGENT = "beforeAgent"
    AFTER_AGENT = "afterAgent"
    BEFORE_PROMPT = "beforePrompt"
    BEFORE_TOOL = "beforeTool"
    AFTER_TOOL = "afterTool"
    AFTER_TOOL_ERROR = "afterToolError"
    BEFORE_PERMISSION = "beforePermission"
    AFTER_STOP = "afterStop"
    BEFORE_SUBAGENT = "beforeSubagent"
    AFTER_SUBAGENT = "afterSubagent"
    BEFORE_LLM = "beforeLLM"
    AFTER_LLM = "afterLLM"


def resolve_hook_event(name: AgentHookEvent | str) -> AgentHookEvent:
    """解析事件名：接受枚举或精确 value 字符串；未知时抛 ``ValueError``。"""
    if isinstance(name, AgentHookEvent):
        return name
    if isinstance(name, str):
        try:
            return AgentHookEvent(name)
        except ValueError:
            pass
    raise ValueError(f"Unknown hook event: {name}")
