"""agent-core：Agent SDK 统一抽象层（deepagents、Qcoder SDK 等）。

业务代码通过统一 API（``create_agent`` + ``run`` / ``stream``）切换底层框架，
上层类型与调用方式不变。
"""

from __future__ import annotations

from agent_core.abc import AgentAdapter
from agent_core.adapter_base import BaseAgentAdapter
from agent_core.backends import DeepAgentsAdapter, QcoderAdapter  # 触发后端注册
from agent_core.exceptions import (
    AgentConfigError,
    AgentCoreError,
    BackendNotFoundError,
    BackendNotImplementedError,
    HookBlockedError,
)
from agent_core.factory import create_agent
from agent_core.hooks import (
    AgentHookEvent,
    BaseAgentHooks,
    HookOutcome,
    HookResult,
    resolve_hook_event,
)
from agent_core.logging import configure_logging
from agent_core.registry import BackendRegistry
from agent_core.types import (
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

__version__ = "0.1.0"

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
