"""BackendRegistry 测试。"""

import pytest

from agent_switch import (
    AgentBackend,
    BackendNotFoundError,
    BackendRegistry,
    DeepAgentsAdapter,
    QcoderAdapter,
)


def test_register_and_get():
    """register 后用字符串或枚举都能 get 到。"""
    BackendRegistry.register("custom", QcoderAdapter)
    assert BackendRegistry.get("custom") is QcoderAdapter
    assert BackendRegistry.get(AgentBackend.QCODER) is QcoderAdapter


def test_get_unknown_raises():
    """未注册的 backend 抛 BackendNotFoundError，且带 backend 属性。"""
    with pytest.raises(BackendNotFoundError) as exc_info:
        BackendRegistry.get("missing-backend")
    assert exc_info.value.backend == "missing-backend"


def test_available_sorted():
    """available() 返回排序后的名称，包含内置 deepagents / qcoder。"""
    available = BackendRegistry.available()
    assert available == sorted(available)
    assert "deepagents" in available
    assert "qcoder" in available


def test_register_overwrites():
    """重复注册同一 backend 时覆盖。"""
    BackendRegistry.register("dup", QcoderAdapter)
    BackendRegistry.register("dup", DeepAgentsAdapter)
    assert BackendRegistry.get("dup") is DeepAgentsAdapter


def test_clear_removes_all_adapters():
    """clear() 清空注册表（仅测试用），随后恢复内置注册。"""
    BackendRegistry.clear()
    assert BackendRegistry.available() == []
    BackendRegistry.register(AgentBackend.DEEPAGENTS, DeepAgentsAdapter)
    BackendRegistry.register(AgentBackend.QCODER, QcoderAdapter)
