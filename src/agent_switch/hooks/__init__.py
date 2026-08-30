"""agent_switch hooks 子包：事件枚举、Context、结果模型与基类。"""

from __future__ import annotations

from agent_switch.hooks.base import BaseAgentHooks
from agent_switch.hooks.context import (
    AfterAgentHookContext,
    AfterLLMHookContext,
    AfterStopHookContext,
    AfterSubagentHookContext,
    AfterToolErrorHookContext,
    AfterToolHookContext,
    AgentHookContext,
    BeforeAgentHookContext,
    BeforeLLMHookContext,
    BeforePermissionHookContext,
    BeforePromptHookContext,
    BeforeSubagentHookContext,
    BeforeToolHookContext,
    HookContextMap,
)
from agent_switch.hooks.enums import AgentHookEvent, resolve_hook_event
from agent_switch.hooks.result import INTERCEPTABLE_HOOK_EVENTS, HookOutcome, HookResult

__all__ = [
    "AgentHookContext",
    "AgentHookEvent",
    "BeforeAgentHookContext",
    "AfterAgentHookContext",
    "BeforePromptHookContext",
    "BeforeToolHookContext",
    "AfterToolHookContext",
    "AfterToolErrorHookContext",
    "BeforePermissionHookContext",
    "AfterStopHookContext",
    "BeforeSubagentHookContext",
    "AfterSubagentHookContext",
    "BeforeLLMHookContext",
    "AfterLLMHookContext",
    "HookContextMap",
    "HookOutcome",
    "HookResult",
    "INTERCEPTABLE_HOOK_EVENTS",
    "BaseAgentHooks",
    "resolve_hook_event",
]

# 循环导入处理：将 BaseAgentHooks 注入 types.config 模块命名空间，
# 供 AgentConfig 的字符串注解（from __future__ import annotations）解析，并重建模型。
import agent_switch.types.config as _agent_config_module

setattr(_agent_config_module, "BaseAgentHooks", BaseAgentHooks)
_agent_config_module.AgentConfig.model_rebuild()
