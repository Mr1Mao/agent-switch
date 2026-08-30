"""后端注册表：backend 名称 → 适配器类。"""

from __future__ import annotations

from typing import ClassVar

from agent_switch.abc import AgentAdapter
from agent_switch.exceptions import BackendNotFoundError
from agent_switch.types import AgentBackend


def _backend_key(backend: AgentBackend | str) -> str:
    """统一 backend 的注册表 key：枚举取 value，字符串原样。"""
    return backend.value if isinstance(backend, AgentBackend) else backend


class BackendRegistry:
    """注册 backend 名称到 ``AgentAdapter`` 子类。"""

    _adapters: ClassVar[dict[str, type[AgentAdapter]]] = {}

    @classmethod
    def register(cls, backend: AgentBackend | str, adapter_cls: type[AgentAdapter]) -> None:
        """注册适配器；重复注册时覆盖。"""
        cls._adapters[_backend_key(backend)] = adapter_cls

    @classmethod
    def get(cls, backend: AgentBackend | str) -> type[AgentAdapter]:
        """获取适配器类；未注册时抛 ``BackendNotFoundError``。"""
        key = _backend_key(backend)
        if key not in cls._adapters:
            raise BackendNotFoundError(key)
        return cls._adapters[key]

    @classmethod
    def available(cls) -> list[str]:
        """返回已注册的 backend 名称（排序后）。"""
        return sorted(cls._adapters)

    @classmethod
    def clear(cls) -> None:
        """清空注册表（仅测试用）。"""
        cls._adapters.clear()
