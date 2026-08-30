"""输入归一化工具。"""

from __future__ import annotations

from agent_switch.types import AgentMessage, MessageRole


def normalize_input(input: str | list[AgentMessage]) -> list[AgentMessage]:
    """归一化输入：``str`` → ``[AgentMessage(user, content)]``；``list`` → 原样拷贝。"""
    if isinstance(input, str):
        return [AgentMessage(role=MessageRole.USER, content=input)]
    if isinstance(input, list):
        return list(input)
    raise TypeError(f"input 必须是 str 或 list[AgentMessage]，收到 {type(input).__name__}")


def format_input_preview(messages: list[AgentMessage]) -> str:
    """取最后一条消息的 content 作为输入预览。"""
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last.content, str):
        return last.content
    return str(last.content)
