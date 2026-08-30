"""qcoder hooks 桥接测试。"""

from agent_switch.backends.qcoder.hooks_bridge import build_qoder_hooks
from agent_switch.hooks.base import BaseAgentHooks
from agent_switch.hooks.dispatcher import AgentHooksDispatcher
from agent_switch.hooks.result import HookOutcome, HookResult


def test_build_qoder_hooks_only_registers_overridden_events():
    """只注册有覆写回调的事件；未覆写的事件不注册。"""

    class ToolHooks(BaseAgentHooks):
        async def before_tool(self, context):
            return None

    hooks = build_qoder_hooks(AgentHooksDispatcher([ToolHooks()]), backend="qcoder")
    assert hooks is not None
    assert set(hooks.keys()) == {"PreToolUse"}
    assert "PostToolUse" not in hooks


def test_build_qoder_hooks_none_when_nothing_overridden():
    """全部事件都未覆写时返回 None（不注入空 hooks）。"""
    hooks = build_qoder_hooks(AgentHooksDispatcher([BaseAgentHooks()]), backend="qcoder")
    assert hooks is None


def test_all_six_call_level_events_registered():
    """6 个调用级事件全部覆写时全部注册。"""

    class AllHooks(BaseAgentHooks):
        async def before_tool(self, context):
            return None

        async def after_tool(self, context):
            return None

        async def after_tool_error(self, context):
            return None

        async def before_permission(self, context):
            return None

        async def before_subagent(self, context):
            return None

        async def after_subagent(self, context):
            return None

    hooks = build_qoder_hooks(AgentHooksDispatcher([AllHooks()]), backend="qcoder")
    assert set(hooks.keys()) == {
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PermissionRequest",
        "SubagentStart",
        "SubagentStop",
    }


async def test_before_tool_callback_maps_block():
    """PreToolUse 回调：BLOCK → continue_:False + decision:block + permissionDecision:deny。"""
    events: list[str] = []

    class BlockHooks(BaseAgentHooks):
        async def before_tool(self, context):
            events.append("beforeTool")
            return HookResult(outcome=HookOutcome.BLOCK, reason="nope")

    hooks = build_qoder_hooks(AgentHooksDispatcher([BlockHooks()]), backend="qcoder")
    callback = hooks["PreToolUse"][0].hooks[0]
    output = await callback(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"cmd": "ls"},
            "tool_use_id": "tool_use_1",
        },
        "tool_use_1",
        {},
    )
    assert events == ["beforeTool"]
    assert output["continue_"] is False
    assert output["decision"] == "block"
    assert output["reason"] == "nope"
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"


async def test_after_tool_callback_maps_modify():
    """PostToolUse 回调：MODIFY(data.updated_tool_output) → updatedToolOutput。"""

    class ModifyHooks(BaseAgentHooks):
        async def after_tool(self, context):
            return HookResult(outcome=HookOutcome.MODIFY, data={"updated_tool_output": "fixed"})

    hooks = build_qoder_hooks(AgentHooksDispatcher([ModifyHooks()]), backend="qcoder")
    callback = hooks["PostToolUse"][0].hooks[0]
    output = await callback(
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {},
            "tool_use_id": "tool_use_1",
            "tool_response": "raw",
        },
        "tool_use_1",
        {},
    )
    assert output["continue_"] is True
    assert output["hookSpecificOutput"]["updatedToolOutput"] == "fixed"
