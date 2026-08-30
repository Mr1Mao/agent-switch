"""agent_switch hooks ↔ qoder-agent-sdk hooks 桥接。

把 agent_switch 的 hooks dispatcher 编译成 ``QoderAgentOptions.hooks``
（``{HookEvent: [HookMatcher]}``），让调用级事件在 qoder CLI 内部触发：
- ``beforeTool`` → ``PreToolUse``
- ``afterTool`` → ``PostToolUse``
- ``afterToolError`` → ``PostToolUseFailure``
- ``beforePermission`` → ``PermissionRequest``
- ``beforeSubagent`` → ``SubagentStart``
- ``afterSubagent`` → ``SubagentStop``

会话级事件（beforeAgent / beforePrompt / beforeLLM / afterLLM / afterAgent /
afterStop）保持由 adapter 层触发（Qoder 的 SessionStart / UserPromptSubmit hooks
与我们的消息改写语义不兼容，且 Qoder 没有 beforeLLM / afterLLM 原生 hook）。

BLOCK / MODIFY 映射（SyncHookJSONOutput）：
- BLOCK → ``{"continue_": False, "decision": "block", "stopReason": reason}``；
  工具 / 权限事件额外附 ``hookSpecificOutput.permissionDecision="deny"``；
- MODIFY（afterTool）→ ``updatedToolOutput``；MODIFY（beforeTool）→ ``updatedInput``。
"""

from __future__ import annotations

from typing import Any, Callable

from agent_switch.backends.qcoder.mapping import _import_sdk
from agent_switch.hooks.context import (
    AfterSubagentHookContext,
    AfterToolErrorHookContext,
    AfterToolHookContext,
    BeforePermissionHookContext,
    BeforeSubagentHookContext,
    BeforeToolHookContext,
)
from agent_switch.hooks.dispatcher import AgentHooksDispatcher
from agent_switch.hooks.enums import AgentHookEvent
from agent_switch.hooks.result import HookOutcome, HookResult
from agent_switch.types import ToolCall

#: 会话标识：(session_id, correlation_id)
SessionIds = tuple[str | None, str | None]
#: 会话标识提供者（adapter 每次 run/stream 更新，hook 回调实时读取）
SessionProvider = Callable[[], SessionIds]

#: agent_switch 事件 → Qoder HookEvent 名（仅调用级，会话级由 adapter 层触发）
EVENT_TO_QODER_HOOK: dict[AgentHookEvent, str] = {
    AgentHookEvent.BEFORE_TOOL: "PreToolUse",
    AgentHookEvent.AFTER_TOOL: "PostToolUse",
    AgentHookEvent.AFTER_TOOL_ERROR: "PostToolUseFailure",
    AgentHookEvent.BEFORE_PERMISSION: "PermissionRequest",
    AgentHookEvent.BEFORE_SUBAGENT: "SubagentStart",
    AgentHookEvent.AFTER_SUBAGENT: "SubagentStop",
}


def build_qoder_hooks(
    dispatcher: AgentHooksDispatcher,
    backend: str,
    session_provider: SessionProvider | None = None,
) -> dict[str, list[Any]] | None:
    """构建 ``QoderAgentOptions.hooks``；无已覆写事件时返回 None。

    只注册「有 hook 覆写了对应方法」的事件（复用 dispatcher 的
    ``_effective_hooks`` 判断），避免给 CLI 注册空回调。
    """
    sdk = _import_sdk()
    provider = session_provider or (lambda: (None, None))
    result: dict[str, list[Any]] = {}
    for event, hook_name in EVENT_TO_QODER_HOOK.items():
        if not dispatcher._effective_hooks(event):
            continue
        result[hook_name] = [
            sdk.HookMatcher(
                matcher=None,  # None 匹配全部工具
                hooks=[_make_callback(dispatcher, backend, provider, event)],
            )
        ]
    return result or None


def _make_callback(
    dispatcher: AgentHooksDispatcher,
    backend: str,
    session_provider: SessionProvider,
    event: AgentHookEvent,
) -> Callable[..., Any]:
    """构造 Qoder ``HookCallback``：``async (hook_input, tool_use_id, context) -> dict``。"""

    async def _callback(hook_input: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        session_id, correlation_id = session_provider()
        agent_context = _build_context(event, hook_input, backend, session_id, correlation_id)
        result = await dispatcher.emit(event, agent_context)
        return _hook_result_to_output(event, result, hook_input)

    return _callback


def _build_context(
    event: AgentHookEvent,
    hook_input: Any,
    backend: str,
    session_id: str | None,
    correlation_id: str | None,
) -> Any:
    """把 Qoder hook_input（TypedDict）翻译成对应的 agent_switch Context。"""
    if isinstance(hook_input, dict):
        data: dict[str, Any] = hook_input
    else:
        data = getattr(hook_input, "__dict__", {}) or {}
    tool_name = str(data.get("tool_name") or "")
    tool_input = data.get("tool_input") or {}
    if event is AgentHookEvent.BEFORE_TOOL:
        return BeforeToolHookContext(
            backend=backend,
            session_id=session_id,
            correlation_id=correlation_id,
            tool_name=tool_name,
            tool_call=ToolCall(
                id=str(data.get("tool_use_id") or ""),
                name=tool_name,
                arguments=tool_input,
            ),
        )
    if event is AgentHookEvent.AFTER_TOOL:
        return AfterToolHookContext(
            backend=backend,
            session_id=session_id,
            correlation_id=correlation_id,
            tool_name=tool_name,
            result=data.get("tool_response"),
        )
    if event is AgentHookEvent.AFTER_TOOL_ERROR:
        return AfterToolErrorHookContext(
            backend=backend,
            session_id=session_id,
            correlation_id=correlation_id,
            tool_name=tool_name,
            error=RuntimeError(str(data.get("error") or "tool call failed")),
        )
    if event is AgentHookEvent.BEFORE_PERMISSION:
        return BeforePermissionHookContext(
            backend=backend,
            session_id=session_id,
            correlation_id=correlation_id,
            action=tool_name,
            details=tool_input,
        )
    if event is AgentHookEvent.BEFORE_SUBAGENT:
        return BeforeSubagentHookContext(
            backend=backend,
            session_id=session_id,
            correlation_id=correlation_id,
            task=str(data.get("agent_type") or data.get("agent_id") or ""),
        )
    if event is AgentHookEvent.AFTER_SUBAGENT:
        return AfterSubagentHookContext(
            backend=backend,
            session_id=session_id,
            correlation_id=correlation_id,
            result=data,
        )
    raise ValueError(f"不支持的桥接事件: {event}")


def _hook_result_to_output(event: AgentHookEvent, result: HookResult, hook_input: Any) -> dict[str, Any]:
    """把 agent_switch HookResult 转成 Qoder SyncHookJSONOutput。"""
    if result is None or result.outcome is HookOutcome.CONTINUE:
        return {}
    if result.outcome is HookOutcome.BLOCK:
        output: dict[str, Any] = {
            "continue_": False,
            "decision": "block",
            "stopReason": result.reason,
            "reason": result.reason,
        }
        if event in (AgentHookEvent.BEFORE_TOOL, AgentHookEvent.BEFORE_PERMISSION):
            output["hookSpecificOutput"] = {
                "hookEventName": EVENT_TO_QODER_HOOK[event],
                "permissionDecision": "deny",
                "permissionDecisionReason": result.reason,
            }
        return output
    # MODIFY
    data = result.data or {}
    if event is AgentHookEvent.AFTER_TOOL and "updated_tool_output" in data:
        return {
            "continue_": True,
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "updatedToolOutput": str(data["updated_tool_output"]),
            },
        }
    if event is AgentHookEvent.BEFORE_TOOL and "updated_input" in data:
        return {
            "continue_": True,
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "updatedInput": data["updated_input"],
            },
        }
    return {}
