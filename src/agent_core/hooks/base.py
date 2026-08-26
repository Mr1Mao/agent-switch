"""BaseAgentHooks：可继承的 hooks 基类。

12 个 async 生命周期方法，默认返回 ``None``。子类按需覆写；
未覆写的基类空方法不参与派发（dispatcher 通过比较 ``__func__`` 与基类方法判断）。
"""

from __future__ import annotations

from agent_core.hooks.context import (
    AfterAgentHookContext,
    AfterLLMHookContext,
    AfterStopHookContext,
    AfterSubagentHookContext,
    AfterToolErrorHookContext,
    AfterToolHookContext,
    BeforeAgentHookContext,
    BeforeLLMHookContext,
    BeforePermissionHookContext,
    BeforePromptHookContext,
    BeforeSubagentHookContext,
    BeforeToolHookContext,
)
from agent_core.hooks.result import HookResult


class BaseAgentHooks:
    """Agent hooks 基类。

    方法可返回 ``HookResult``（BLOCK 短路 / MODIFY 修改数据）或 ``None``。
    """

    async def before_agent(self, context: BeforeAgentHookContext) -> HookResult | None:
        """会话开始前触发。"""
        return None

    async def after_agent(self, context: AfterAgentHookContext) -> HookResult | None:
        """会话结束后触发。"""
        return None

    async def before_prompt(self, context: BeforePromptHookContext) -> HookResult | None:
        """提示词发送前触发（可拦截）。"""
        return None

    async def before_tool(self, context: BeforeToolHookContext) -> HookResult | None:
        """工具调用前触发（可拦截）。"""
        return None

    async def after_tool(self, context: AfterToolHookContext) -> HookResult | None:
        """工具调用后触发。"""
        return None

    async def after_tool_error(self, context: AfterToolErrorHookContext) -> HookResult | None:
        """工具调用出错时触发。"""
        return None

    async def before_permission(self, context: BeforePermissionHookContext) -> HookResult | None:
        """权限检查前触发（可拦截）。"""
        return None

    async def after_stop(self, context: AfterStopHookContext) -> HookResult | None:
        """Agent 停止后触发。"""
        return None

    async def before_subagent(self, context: BeforeSubagentHookContext) -> HookResult | None:
        """子代理执行前触发。"""
        return None

    async def after_subagent(self, context: AfterSubagentHookContext) -> HookResult | None:
        """子代理执行后触发。"""
        return None

    async def before_llm(self, context: BeforeLLMHookContext) -> HookResult | None:
        """LLM 调用前触发。"""
        return None

    async def after_llm(self, context: AfterLLMHookContext) -> HookResult | None:
        """LLM 调用后触发。"""
        return None
