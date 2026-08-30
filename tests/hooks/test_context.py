"""hooks Context 模型测试。"""

import pytest
from pydantic import ValidationError

from agent_switch.hooks.context import (
    AgentHookContext,
    BeforeAgentHookContext,
    BeforePromptHookContext,
    HookContextMap,
)
from agent_switch.hooks.enums import AgentHookEvent


def test_base_context_common_fields():
    """Context 基类包含 backend / session_id / correlation_id / timestamp / meta。"""
    ctx = BeforeAgentHookContext(backend="qcoder", session_id="s1", correlation_id="c1")
    assert ctx.backend == "qcoder"
    assert ctx.session_id == "s1"
    assert ctx.correlation_id == "c1"
    assert ctx.timestamp is not None
    assert ctx.meta == {}
    assert isinstance(ctx, AgentHookContext)


def test_event_literal_is_fixed_per_context():
    """每个事件 Context 的 event 为固定 Literal，传其他事件报错。"""
    ctx = BeforePromptHookContext(backend="qcoder", messages=[])
    assert ctx.event is AgentHookEvent.BEFORE_PROMPT
    with pytest.raises(ValidationError):
        BeforePromptHookContext(backend="qcoder", messages=[], event=AgentHookEvent.AFTER_STOP)


def test_hook_context_map_covers_all_events():
    """HookContextMap 覆盖全部 12 个事件，且类型为 AgentHookContext 子类。"""
    assert set(HookContextMap) == set(AgentHookEvent)
    for event, context_type in HookContextMap.items():
        assert issubclass(context_type, AgentHookContext)
