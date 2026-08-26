"""hooks 事件枚举测试。"""

import pytest

from agent_core.hooks.enums import AgentHookEvent, resolve_hook_event


def test_hook_event_short_values():
    """事件 value 为 camelCase 短名。"""
    expected = {
        "BEFORE_AGENT": "beforeAgent",
        "AFTER_AGENT": "afterAgent",
        "BEFORE_PROMPT": "beforePrompt",
        "BEFORE_TOOL": "beforeTool",
        "AFTER_TOOL": "afterTool",
        "AFTER_TOOL_ERROR": "afterToolError",
        "BEFORE_PERMISSION": "beforePermission",
        "AFTER_STOP": "afterStop",
        "BEFORE_SUBAGENT": "beforeSubagent",
        "AFTER_SUBAGENT": "afterSubagent",
        "BEFORE_LLM": "beforeLLM",
        "AFTER_LLM": "afterLLM",
    }
    assert {member.name: member.value for member in AgentHookEvent} == expected


def test_resolve_hook_event_accepts_enum_and_string():
    """resolve_hook_event 接受枚举或精确 value 字符串。"""
    assert resolve_hook_event(AgentHookEvent.BEFORE_PROMPT) is AgentHookEvent.BEFORE_PROMPT
    assert resolve_hook_event("beforeAgent") is AgentHookEvent.BEFORE_AGENT
    assert resolve_hook_event("afterStop") is AgentHookEvent.AFTER_STOP


def test_resolve_hook_event_unknown_raises_value_error():
    """未知事件抛 ValueError（无 legacy alias）。"""
    with pytest.raises(ValueError, match="Unknown hook event"):
        resolve_hook_event("beforeSesh")
