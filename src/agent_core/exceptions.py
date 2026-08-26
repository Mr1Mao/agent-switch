"""agent_core 异常体系。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.hooks.enums import AgentHookEvent


class AgentCoreError(Exception):
    """agent_core 所有异常的基类。"""


class BackendNotFoundError(AgentCoreError):
    """后端未注册。"""

    backend: str

    def __init__(self, backend: str) -> None:
        self.backend = backend
        super().__init__(f"Backend not found: {backend!r}")


class BackendNotImplementedError(AgentCoreError):
    """后端尚未实现（预留）。"""


class AgentConfigError(AgentCoreError):
    """Agent 配置错误。"""


class HookBlockedError(AgentCoreError):
    """hook 返回 BLOCK 时抛出。"""

    hook_event: AgentHookEvent
    reason: str

    def __init__(self, hook_event: AgentHookEvent, reason: str = "") -> None:
        self.hook_event = hook_event
        self.reason = reason
        super().__init__(f"Hook {hook_event} blocked: {reason}")


class BackendDependencyError(AgentCoreError):
    """后端依赖未安装（如 deepagents / qoder-agent-sdk）。"""

    backend: str
    install_hint: str

    def __init__(self, backend: str, install_hint: str) -> None:
        self.backend = backend
        self.install_hint = install_hint
        super().__init__(f"Backend {backend!r} dependency missing. {install_hint}")
