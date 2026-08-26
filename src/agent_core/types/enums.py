"""agent_core 基础枚举。"""

from __future__ import annotations

from enum import Enum


class AgentBackend(str, Enum):
    """支持的 Agent 后端。"""

    DEEPAGENTS = "deepagents"
    QCODER = "qcoder"


class MessageRole(str, Enum):
    """消息角色。"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
