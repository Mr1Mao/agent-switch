"""agent_switch 类型系统。"""

from __future__ import annotations

from agent_switch.types.config import AgentConfig
from agent_switch.types.enums import AgentBackend, MessageRole
from agent_switch.types.mcp import AgentMcpConfig, AgentMcpServer
from agent_switch.types.message import AgentMessage, ToolCall, ToolResult
from agent_switch.types.model import AgentModel
from agent_switch.types.response import AgentChunk, AgentResponse
from agent_switch.types.skill import AgentSkillsConfig
from agent_switch.types.subagent import AgentSubagent
from agent_switch.types.tool import AgentTool

__all__ = [
    "AgentBackend",
    "AgentChunk",
    "AgentConfig",
    "AgentMcpConfig",
    "AgentMcpServer",
    "AgentMessage",
    "AgentModel",
    "AgentResponse",
    "AgentSkillsConfig",
    "AgentSubagent",
    "AgentTool",
    "MessageRole",
    "ToolCall",
    "ToolResult",
]
