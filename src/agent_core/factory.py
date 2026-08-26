"""create_agent 工厂：根据 backend 创建适配器实例。"""

from __future__ import annotations

from agent_core.abc import AgentAdapter
from agent_core.logging import get_logger, log_fields
from agent_core.registry import BackendRegistry
from agent_core.types import AgentBackend, AgentConfig

_logger = get_logger("agent_core.factory")


def create_agent(backend: AgentBackend | str, config: AgentConfig | None = None) -> AgentAdapter:
    """创建指定 backend 的适配器实例。

    :param backend: ``AgentBackend`` 枚举或后端名字符串（如 ``"deepagents"``）。
    :param config: 统一的 Agent 配置。
    """
    backend_value = backend.value if isinstance(backend, AgentBackend) else backend
    _logger.info("agent.create.start", extra=log_fields(backend=backend_value))
    adapter_cls = BackendRegistry.get(backend_value)
    adapter = adapter_cls(config=config)
    _logger.info(
        "agent.create.success",
        extra=log_fields(backend=backend_value, adapter=adapter_cls.__name__),
    )
    return adapter
