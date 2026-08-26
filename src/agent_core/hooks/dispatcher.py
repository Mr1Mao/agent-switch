"""Agent hooks 内部派发器（不对外导出）。

按 hooks 列表顺序调用对应事件方法：
- 返回 ``None`` 跳过；
- ``BLOCK`` 短路返回；
- ``MODIFY`` 合并 data（后覆盖同 key）；
- 支持同步 ``def`` hook（经 ``asyncio.to_thread`` 执行）。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from typing import Any

from agent_core.hooks.base import BaseAgentHooks
from agent_core.hooks.context import AgentHookContext
from agent_core.hooks.enums import AgentHookEvent
from agent_core.hooks.result import HookOutcome, HookResult

#: 事件 → snake_case 方法名
HOOK_EVENT_METHOD_MAP: dict[AgentHookEvent, str] = {
    AgentHookEvent.BEFORE_AGENT: "before_agent",
    AgentHookEvent.AFTER_AGENT: "after_agent",
    AgentHookEvent.BEFORE_PROMPT: "before_prompt",
    AgentHookEvent.BEFORE_TOOL: "before_tool",
    AgentHookEvent.AFTER_TOOL: "after_tool",
    AgentHookEvent.AFTER_TOOL_ERROR: "after_tool_error",
    AgentHookEvent.BEFORE_PERMISSION: "before_permission",
    AgentHookEvent.AFTER_STOP: "after_stop",
    AgentHookEvent.BEFORE_SUBAGENT: "before_subagent",
    AgentHookEvent.AFTER_SUBAGENT: "after_subagent",
    AgentHookEvent.BEFORE_LLM: "before_llm",
    AgentHookEvent.AFTER_LLM: "after_llm",
}


class AgentHooksDispatcher:
    """按事件把 Context 派发到 hooks 列表（内部组件，不对外导出）。"""

    def __init__(self, hooks: Sequence[BaseAgentHooks]) -> None:
        # 拷贝一份：防止调用方在派发过程中修改列表导致遍历异常
        self.hooks: list[BaseAgentHooks] = list(hooks)

    def _method_name(self, event: AgentHookEvent) -> str:
        """事件枚举 → 对应 hooks 方法名（camelCase 事件 → snake_case 方法）。"""
        return HOOK_EVENT_METHOD_MAP[event]

    def _effective_hooks(self, event: AgentHookEvent) -> list[BaseAgentHooks]:
        """仅返回「覆写了对应方法」的 hooks。

        判断方式：把 hook 实例上的绑定方法与基类 ``BaseAgentHooks`` 的原始方法
        做 ``__func__`` 身份比较 —— 相同说明子类没有覆写（空实现），
        直接跳过，避免无谓地 await 一个什么都不做的空方法。
        这保证了「未覆写的基类空方法不参与派发」的语义。
        """
        method_name = self._method_name(event)
        # getattr(BaseAgentHooks, name) 返回普通函数（基类是普通类，无绑定）
        base_method: Any = getattr(BaseAgentHooks, method_name)
        effective: list[BaseAgentHooks] = []
        for hook in self.hooks:
            # 实例上的绑定方法；__func__ 才是真正定义它的那个函数对象
            candidate: Any = getattr(hook, method_name, None)
            if candidate is None or getattr(candidate, "__func__", None) is base_method:
                continue
            effective.append(hook)
        return effective

    async def emit(self, event: AgentHookEvent, context: AgentHookContext) -> HookResult:
        """按 hooks 列表顺序派发事件。

        返回语义：
        - 所有 hook 返回 None / CONTINUE → 返回 CONTINUE；
        - 某个 hook 返回 BLOCK → 立即短路返回该 BLOCK 结果（后续 hooks 不再执行）；
        - 一个或多个 hook 返回 MODIFY → 把各自的 ``data`` 合并后返回
          MODIFY 结果（同 key 后写覆盖），供调用方（adapter）改写输入。
        """
        merged_data: dict[str, Any] = {}
        for hook in self._effective_hooks(event):
            result = await self._invoke(hook, event, context)
            if result is None:
                continue
            if result.outcome is HookOutcome.BLOCK:
                return result
            if result.outcome is HookOutcome.MODIFY:
                merged_data.update(result.data or {})
        if merged_data:
            return HookResult(outcome=HookOutcome.MODIFY, data=merged_data)
        return HookResult()

    async def _invoke(self, hook: BaseAgentHooks, event: AgentHookEvent, context: AgentHookContext) -> HookResult | None:
        """执行单个 hook 方法，并把返回值归一化为 HookResult | None。

        - hook 方法声明为 ``async def`` → 直接 await；
        - hook 方法声明为同步 ``def`` → 丢到线程池（asyncio.to_thread）执行，
          避免阻塞事件循环（注意：这样同步 hook 里的共享状态需要自行加锁）；
        - 返回值不是 HookResult（如意外返回了 dict）→ 按 None 处理（忽略）。
        """
        method: Any = getattr(hook, self._method_name(event))
        if inspect.iscoroutinefunction(method):
            raw: Any = await method(context)
        else:
            # 同步 def hook：放到线程池执行，避免阻塞事件循环
            raw = await asyncio.to_thread(method, context)
        if isinstance(raw, HookResult):
            return raw
        return None

    def emit_sync(self, event: AgentHookEvent, context: AgentHookContext) -> HookResult:
        """同步场景使用 ``asyncio.run`` 驱动异步 hooks。

        限制：如果调用时已经存在运行中的事件循环（例如在 async 代码里同步调用
        run()），``asyncio.run`` 会抛 RuntimeError —— 此时应改用异步 ``emit``。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的事件循环：安全地新建一个
            return asyncio.run(self.emit(event, context))
        raise RuntimeError(
            "AgentHooksDispatcher.emit_sync 不能在运行中的事件循环内调用，请使用异步 emit()"
        )
