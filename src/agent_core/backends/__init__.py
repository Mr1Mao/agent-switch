"""后端注册：import 本包即注册 deepagents / qcoder 适配器。"""

from __future__ import annotations

from agent_core.backends.deepagents import DeepAgentsAdapter
from agent_core.backends.qcoder import QcoderAdapter
from agent_core.registry import BackendRegistry
from agent_core.types import AgentBackend

# 自动注册内置后端（重复注册覆盖）
BackendRegistry.register(AgentBackend.DEEPAGENTS, DeepAgentsAdapter)
BackendRegistry.register(AgentBackend.QCODER, QcoderAdapter)

__all__ = ["DeepAgentsAdapter", "QcoderAdapter"]
