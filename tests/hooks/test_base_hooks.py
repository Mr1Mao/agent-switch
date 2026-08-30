"""BaseAgentHooks 基类与派发行为测试。"""

import asyncio

import pytest
from pydantic import ValidationError

from agent_switch import AgentConfig
from agent_switch.hooks.base import BaseAgentHooks
from agent_switch.hooks.context import BeforePromptHookContext
from agent_switch.hooks.dispatcher import AgentHooksDispatcher
from agent_switch.hooks.enums import AgentHookEvent
from agent_switch.hooks.result import HookOutcome, HookResult


def test_single_hooks_normalized_to_list():
    """AgentConfig(hooks=单实例) 归一化为单元素列表。"""
    config = AgentConfig(hooks=BaseAgentHooks())
    assert isinstance(config.hooks, list)
    assert len(config.hooks) == 1
    assert isinstance(config.hooks[0], BaseAgentHooks)


def test_list_hooks_normalized():
    """AgentConfig(hooks=[...]) 保持列表。"""
    config = AgentConfig(hooks=[BaseAgentHooks(), BaseAgentHooks()])
    assert len(config.hooks) == 2


def test_invalid_hooks_type_raises():
    """非法 hooks 类型抛 TypeError / ValidationError。"""
    with pytest.raises((TypeError, ValidationError)):
        AgentConfig(hooks="not-hooks")


def test_unoverridden_hooks_not_dispatched():
    """未覆写的基类空方法不参与派发。"""
    dispatcher = AgentHooksDispatcher([BaseAgentHooks()])
    assert dispatcher._effective_hooks(AgentHookEvent.BEFORE_PROMPT) == []
    assert dispatcher._effective_hooks(AgentHookEvent.AFTER_LLM) == []


def test_multiple_hooks_order_and_block_short_circuit():
    """多 hooks 按顺序执行；BLOCK 短路后续 hooks。"""
    calls: list[str] = []

    class HookA(BaseAgentHooks):
        async def before_prompt(self, context):
            calls.append("A")

    class HookB(BaseAgentHooks):
        async def before_prompt(self, context):
            calls.append("B")
            return HookResult(outcome=HookOutcome.BLOCK, reason="denied")

    class HookC(BaseAgentHooks):
        async def before_prompt(self, context):
            calls.append("C")

    dispatcher = AgentHooksDispatcher([HookA(), HookB(), HookC()])
    context = BeforePromptHookContext(backend="qcoder", messages=[])
    result = asyncio.run(dispatcher.emit(AgentHookEvent.BEFORE_PROMPT, context))
    assert result.outcome is HookOutcome.BLOCK
    assert result.reason == "denied"
    assert calls == ["A", "B"]  # HookC 被短路
