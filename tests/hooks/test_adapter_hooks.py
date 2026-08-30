"""适配器 hooks 生命周期测试（qcoder 后端，mock SDK 调用点）。"""

import pytest

from agent_core import (
    AgentBackend,
    AgentConfig,
    AgentMessage,
    HookBlockedError,
    MessageRole,
    create_agent,
)
from agent_core.hooks.base import BaseAgentHooks
from agent_core.hooks.enums import AgentHookEvent
from agent_core.hooks.result import HookOutcome, HookResult

EXPECTED_SIX_EVENTS = [
    "beforeAgent",
    "beforePrompt",
    "beforeLLM",
    "afterLLM",
    "afterAgent",
    "afterStop",
]


def _tracking_hooks(events: list[str]) -> BaseAgentHooks:
    """记录 6 个已桥接事件的调用顺序。"""

    class TrackingHooks(BaseAgentHooks):
        async def before_agent(self, context):
            events.append("beforeAgent")

        async def before_prompt(self, context):
            events.append("beforePrompt")

        async def before_llm(self, context):
            events.append("beforeLLM")

        async def after_llm(self, context):
            events.append("afterLLM")

        async def after_agent(self, context):
            events.append("afterAgent")

        async def after_stop(self, context):
            events.append("afterStop")

    return TrackingHooks()


def test_qcoder_run_triggers_six_events(qcoder_fake_query):
    """qcoder run 按顺序触发 6 个阶段事件（fake query 回显输入）。"""
    events: list[str] = []
    agent = create_agent(AgentBackend.QCODER, AgentConfig(hooks=_tracking_hooks(events)))
    response = agent.run("hello")
    assert "[stub] hello" in response.content
    assert events == EXPECTED_SIX_EVENTS


async def test_qcoder_stream_triggers_six_events(qcoder_fake_query):
    """qcoder stream 同样按顺序触发 6 个阶段事件。"""
    events: list[str] = []
    agent = create_agent(AgentBackend.QCODER, AgentConfig(hooks=_tracking_hooks(events)))
    chunks = [chunk async for chunk in agent.stream("hello")]
    assert chunks[-1].is_finish is True
    assert events == EXPECTED_SIX_EVENTS


def test_block_before_prompt_raises_hook_blocked_error():
    """BLOCK 抛 HookBlockedError，携带事件与 reason（在调用 SDK 前即中止）。"""

    class BlockHooks(BaseAgentHooks):
        async def before_prompt(self, context):
            return HookResult(outcome=HookOutcome.BLOCK, reason="no thank you")

    agent = create_agent(AgentBackend.QCODER, AgentConfig(hooks=BlockHooks()))
    with pytest.raises(HookBlockedError) as exc_info:
        agent.run("hello")
    assert exc_info.value.hook_event is AgentHookEvent.BEFORE_PROMPT
    assert exc_info.value.reason == "no thank you"


def test_modify_before_prompt_rewrites_prompt(qcoder_fake_query):
    """MODIFY 改写 prompt 后，fake query 回显的回复体现修改后的消息。"""

    class ModifyHooks(BaseAgentHooks):
        async def before_prompt(self, context):
            return HookResult(
                outcome=HookOutcome.MODIFY,
                data={
                    "messages": [
                        AgentMessage(role=MessageRole.USER, content="modified prompt")
                    ]
                },
            )

    agent = create_agent(AgentBackend.QCODER, AgentConfig(hooks=ModifyHooks()))
    response = agent.run("original prompt")
    assert "modified prompt" in response.content
    assert "original prompt" not in response.content


def test_multiple_hooks_follow_config_order(qcoder_fake_query):
    """多个 hooks 按 config 中的顺序执行（同一事件内亦按序）。"""
    order: list[str] = []

    class FirstHooks(BaseAgentHooks):
        async def before_agent(self, context):
            order.append("first")

        async def before_prompt(self, context):
            order.append("first-prompt")

    class SecondHooks(BaseAgentHooks):
        async def before_agent(self, context):
            order.append("second")

        async def before_prompt(self, context):
            order.append("second-prompt")

    agent = create_agent(
        AgentBackend.QCODER, AgentConfig(hooks=[FirstHooks(), SecondHooks()])
    )
    agent.run("hello")
    assert order == ["first", "second", "first-prompt", "second-prompt"]
