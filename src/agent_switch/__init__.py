"""agent-switch：Agent SDK 统一抽象层（deepagents、Qcoder SDK 等）。

业务代码通过统一 API（``create_agent`` + ``run`` / ``stream``）切换底层框架，
上层类型与调用方式不变。
"""

from __future__ import annotations

from agent_switch.abc import AgentAdapter
from agent_switch.adapter_base import BaseAgentAdapter
from agent_switch.backends import DeepAgentsAdapter, QcoderAdapter  # 触发后端注册
from agent_switch.exceptions import (
    AgentConfigError,
    AgentCoreError,
    BackendNotFoundError,
    BackendNotImplementedError,
    HookBlockedError,
)
from agent_switch.factory import create_agent
from agent_switch.hooks import (
    AgentHookEvent,
    BaseAgentHooks,
    HookOutcome,
    HookResult,
    resolve_hook_event,
)
from agent_switch.logging import configure_logging
from agent_switch.registry import BackendRegistry
from agent_switch.types import (
    AgentBackend,
    AgentChunk,
    AgentConfig,
    AgentMcpConfig,
    AgentMcpServer,
    AgentMessage,
    AgentModel,
    AgentResponse,
    AgentSkillsConfig,
    AgentSubagent,
    AgentTool,
    MessageRole,
    ToolCall,
    ToolResult,
)

__version__ = "0.2.0"

__all__ = [
    "AgentAdapter",
    "AgentBackend",
    "AgentChunk",
    "AgentConfig",
    "AgentConfigError",
    "AgentCoreError",
    "AgentHookEvent",
    "AgentMcpConfig",
    "AgentMcpServer",
    "AgentMessage",
    "AgentModel",
    "AgentResponse",
    "AgentSkillsConfig",
    "AgentSubagent",
    "AgentTool",
    "BackendNotFoundError",
    "BackendNotImplementedError",
    "BackendRegistry",
    "BaseAgentAdapter",
    "BaseAgentHooks",
    "DeepAgentsAdapter",
    "HookBlockedError",
    "HookOutcome",
    "HookResult",
    "MessageRole",
    "QcoderAdapter",
    "ToolCall",
    "ToolResult",
    "configure_logging",
    "create_agent",
    "resolve_hook_event",
]
