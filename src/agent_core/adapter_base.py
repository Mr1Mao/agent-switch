"""适配器基类：提供 hooks 生命周期编排。

事件分两层触发：

- **agent 级**（每次 agent 执行一次）：``beforeAgent → beforePrompt → afterAgent``。
  对 deepagents 后端，这三个事件由注入的 ``AgentHooksMiddleware`` 在 SDK 内部的
  ``before_agent`` / ``after_agent``（图的 entry / exit 节点）触发；对 stub 后端
  由本类在 ``run`` / ``stream`` 主流程中触发。
- **调用级**（每次 LLM / 工具调用一次）：``beforeLLM / afterLLM / beforeTool /
  afterTool / afterToolError``。对 deepagents 后端由中间件的 ``wrap_model_call`` /
  ``wrap_tool_call`` 触发。
- ``afterStop``（reason 为 ``complete`` / ``error``）始终由本类在 run/stream 边界触发。

尚未桥接（不触发）的事件：``beforePermission / beforeSubagent / afterSubagent``。
"""

from __future__ import annotations

import uuid

from agent_core.abc import AgentAdapter
from agent_core.exceptions import HookBlockedError
from agent_core.hooks.context import AgentHookContext
from agent_core.hooks.dispatcher import AgentHooksDispatcher
from agent_core.hooks.emitter import (
    apply_messages_modify,
    build_after_agent_context,
    build_after_llm_context,
    build_after_stop_context,
    build_before_agent_context,
    build_before_llm_context,
    build_before_prompt_context,
)
from agent_core.hooks.enums import AgentHookEvent
from agent_core.hooks.result import HookOutcome, HookResult
from agent_core.logging import get_logger
from agent_core.types import AgentConfig, AgentMessage, AgentResponse
from agent_core.utils.input import normalize_input

_logger = get_logger("agent_core.adapter")


class BaseAgentAdapter(AgentAdapter):
    """带 hooks 生命周期管理的适配器基类。

    子类（如 QcoderAdapter / DeepAgentsAdapter）只需实现 ``run`` / ``stream``，
    并调用本类提供的生命周期方法即可获得统一的 hooks 行为。
    """

    #: True 时 beforeLLM / afterLLM 由后端中间件按「每次调用」触发，
    #: adapter 层不再重复触发（deepagents 通过 wrap_model_call 桥接）。
    call_hooks_via_middleware: bool = False
    #: True 时 beforeAgent / beforePrompt / afterAgent 由后端中间件的
    #: before_agent / after_agent 节点钩子触发（deepagents 的 entry / exit 节点）。
    #: 注意：错误路径的 afterAgent 仍在 adapter 层触发 —— 因为图抛异常时
    #: after_agent 节点不会执行，只能由 _finalize_run_error_* 补发。
    agent_hooks_via_middleware: bool = False

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config)
        # 每次 run/stream 生成一对相关 ID：
        # session_id 标记一次会话，correlation_id 用于把同一次调用产生的
        # 日志 / hooks 上下文关联起来。中间件通过 session_provider 闭包读取它们。
        self._session_id: str | None = None
        self._correlation_id: str | None = None

    # ---- 内部工具 ----

    def _resolve_config(self, config: AgentConfig | None) -> AgentConfig | None:
        """配置解析：调用时传入的 config 优先，否则使用构造时配置。

        这样既支持 ``create_agent(backend, config)`` 一次配置多次使用，
        也支持 ``agent.run(input, AgentConfig(...))`` 按次覆盖（不污染默认配置）。
        """
        return config if config is not None else self._default_config

    def _resolve_hooks_dispatcher(self, config: AgentConfig | None) -> AgentHooksDispatcher | None:
        """根据（解析后的）config 构建 hooks 派发器；无 hooks 时返回 None。

        返回 None 意味着后续所有 emit 都走「无 hooks」快捷路径（直接返回
        CONTINUE 结果），避免为没有 hooks 的调用创建派发器。
        """
        resolved = self._resolve_config(config)
        if resolved is None or not resolved.hooks:
            return None
        return AgentHooksDispatcher(resolved.hooks)

    @staticmethod
    def _new_session_ids() -> tuple[str, str]:
        """生成 session_id / correlation_id：取 uuid 前 12 位十六进制，足够短且唯一。"""
        return uuid.uuid4().hex[:12], uuid.uuid4().hex[:12]

    def _emit_hook_sync(self, dispatcher: AgentHooksDispatcher | None, event: AgentHookEvent, context: AgentHookContext) -> HookResult:
        """同步派发单个事件。

        dispatcher 为 None（未配置 hooks）时直接返回默认 CONTINUE 结果；
        否则调用 dispatcher.emit_sync —— 其内部用 asyncio.run 驱动异步 hook，
        因此不能在已有运行中事件循环的上下文中调用（会抛 RuntimeError）。
        """
        if dispatcher is None:
            return HookResult()
        return dispatcher.emit_sync(event, context)

    async def _emit_hook_async(self, dispatcher: AgentHooksDispatcher | None, event: AgentHookEvent, context: AgentHookContext) -> HookResult:
        """异步派发单个事件（stream 等异步路径使用）。"""
        if dispatcher is None:
            return HookResult()
        return await dispatcher.emit(event, context)

    def _raise_if_hook_blocked(self, event: AgentHookEvent, hook_result: HookResult | None) -> None:
        """hook 返回 BLOCK 时抛出 HookBlockedError。

        BLOCK 语义：拦截并终止本次调用。抛出后由 run/stream 的异常路径接管
        （触发 afterAgent(error) + afterStop(error) 后重抛给调用方）。
        """
        if hook_result is not None and hook_result.outcome is HookOutcome.BLOCK:
            raise HookBlockedError(hook_event=event, reason=hook_result.reason)

    # ---- 生命周期：前置 ----

    def _prepare_messages_sync(self, input: str | list[AgentMessage], config: AgentConfig | None) -> list[AgentMessage]:
        """同步前置：normalize → beforeAgent → beforePrompt → apply_messages_modify。

        流程说明：
        1. 先把输入归一化为 list[AgentMessage]（str 会被包装成 user 消息）；
        2. 生成本次调用的 session_id / correlation_id（后续事件共用）；
        3. 依次触发 beforeAgent / beforePrompt，任一返回 BLOCK 即中止；
        4. beforePrompt 返回 MODIFY 时，用 data["messages"] 替换输入消息。
        """
        dispatcher = self._resolve_hooks_dispatcher(config)
        messages = normalize_input(input)
        # 会话 ID 必须在这里生成：即使没有 hooks，后续中间件 / 日志也需要它
        self._session_id, self._correlation_id = self._new_session_ids()
        if self.agent_hooks_via_middleware:
            # beforeAgent / beforePrompt 由中间件 before_agent（entry 节点）触发，
            # adapter 层只做归一化与会话 ID 生成，避免重复触发
            return messages
        session_result = self._emit_hook_sync(
            dispatcher,
            AgentHookEvent.BEFORE_AGENT,
            build_before_agent_context(
                self.backend_name,
                messages,
                config,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.BEFORE_AGENT, session_result)
        prompt_result = self._emit_hook_sync(
            dispatcher,
            AgentHookEvent.BEFORE_PROMPT,
            build_before_prompt_context(
                self.backend_name,
                messages,
                config,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.BEFORE_PROMPT, prompt_result)
        # MODIFY 时替换消息列表；未修改时原样返回
        return apply_messages_modify(messages, prompt_result)

    async def _prepare_messages_async(self, input: str | list[AgentMessage], config: AgentConfig | None) -> list[AgentMessage]:
        """异步版前置：normalize → beforeAgent → beforePrompt → apply_messages_modify。"""
        dispatcher = self._resolve_hooks_dispatcher(config)
        messages = normalize_input(input)
        self._session_id, self._correlation_id = self._new_session_ids()
        if self.agent_hooks_via_middleware:
            # beforeAgent / beforePrompt 由中间件 before_agent 节点触发
            return messages
        session_result = await self._emit_hook_async(
            dispatcher,
            AgentHookEvent.BEFORE_AGENT,
            build_before_agent_context(
                self.backend_name,
                messages,
                config,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.BEFORE_AGENT, session_result)
        prompt_result = await self._emit_hook_async(
            dispatcher,
            AgentHookEvent.BEFORE_PROMPT,
            build_before_prompt_context(
                self.backend_name,
                messages,
                config,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.BEFORE_PROMPT, prompt_result)
        return apply_messages_modify(messages, prompt_result)

    def _emit_before_llm_sync(self, messages: list[AgentMessage], config: AgentConfig | None) -> list[AgentMessage]:
        """beforeLLM → apply_messages_modify，返回可能被改写的消息列表。

        这是发给 SDK 前最后一次改写机会（例如注入上下文、替换 prompt）。
        """
        if self.call_hooks_via_middleware:
            # beforeLLM 由后端中间件按调用触发，adapter 层跳过
            return messages
        dispatcher = self._resolve_hooks_dispatcher(config)
        result = self._emit_hook_sync(
            dispatcher,
            AgentHookEvent.BEFORE_LLM,
            build_before_llm_context(
                self.backend_name,
                messages,
                config,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.BEFORE_LLM, result)
        return apply_messages_modify(messages, result)

    async def _emit_before_llm_async(self, messages: list[AgentMessage], config: AgentConfig | None) -> list[AgentMessage]:
        """异步版：beforeLLM → apply_messages_modify。"""
        if self.call_hooks_via_middleware:
            # beforeLLM 由后端中间件按调用触发，adapter 层跳过
            return messages
        dispatcher = self._resolve_hooks_dispatcher(config)
        result = await self._emit_hook_async(
            dispatcher,
            AgentHookEvent.BEFORE_LLM,
            build_before_llm_context(
                self.backend_name,
                messages,
                config,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.BEFORE_LLM, result)
        return apply_messages_modify(messages, result)

    # ---- 生命周期：收尾 ----

    def _finalize_run_success_sync(self, config: AgentConfig | None, response: AgentResponse) -> None:
        """run 成功收尾：afterLLM → afterAgent → afterStop（complete）。

        中间件桥接开关（call/agent hooks via middleware）为 True 时跳过对应事件，
        避免 adapter 层与 SDK 内部重复触发。
        """
        dispatcher = self._resolve_hooks_dispatcher(config)
        if not self.call_hooks_via_middleware:
            after_llm = self._emit_hook_sync(
                dispatcher,
                AgentHookEvent.AFTER_LLM,
                build_after_llm_context(
                    self.backend_name,
                    response=response,
                    session_id=self._session_id,
                    correlation_id=self._correlation_id,
                ),
            )
            self._raise_if_hook_blocked(AgentHookEvent.AFTER_LLM, after_llm)
        if not self.agent_hooks_via_middleware:
            after_agent = self._emit_hook_sync(
                dispatcher,
                AgentHookEvent.AFTER_AGENT,
                build_after_agent_context(
                    self.backend_name,
                    response=response,
                    session_id=self._session_id,
                    correlation_id=self._correlation_id,
                ),
            )
            self._raise_if_hook_blocked(AgentHookEvent.AFTER_AGENT, after_agent)
        after_stop = self._emit_hook_sync(
            dispatcher,
            AgentHookEvent.AFTER_STOP,
            build_after_stop_context(
                self.backend_name,
                reason="complete",
                response=response,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.AFTER_STOP, after_stop)

    async def _finalize_run_success_async(self, config: AgentConfig | None, response: AgentResponse) -> None:
        """异步版 run 成功收尾：afterLLM → afterAgent → afterStop（complete）。"""
        dispatcher = self._resolve_hooks_dispatcher(config)
        if not self.call_hooks_via_middleware:
            after_llm = await self._emit_hook_async(
                dispatcher,
                AgentHookEvent.AFTER_LLM,
                build_after_llm_context(
                    self.backend_name,
                    response=response,
                    session_id=self._session_id,
                    correlation_id=self._correlation_id,
                ),
            )
            self._raise_if_hook_blocked(AgentHookEvent.AFTER_LLM, after_llm)
        if not self.agent_hooks_via_middleware:
            after_agent = await self._emit_hook_async(
                dispatcher,
                AgentHookEvent.AFTER_AGENT,
                build_after_agent_context(
                    self.backend_name,
                    response=response,
                    session_id=self._session_id,
                    correlation_id=self._correlation_id,
                ),
            )
            self._raise_if_hook_blocked(AgentHookEvent.AFTER_AGENT, after_agent)
        after_stop = await self._emit_hook_async(
            dispatcher,
            AgentHookEvent.AFTER_STOP,
            build_after_stop_context(
                self.backend_name,
                reason="complete",
                response=response,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.AFTER_STOP, after_stop)

    def _finalize_run_error_sync(self, config: AgentConfig | None, error: BaseException) -> None:
        """run 失败收尾：afterAgent → afterStop（error）。

        与成功路径不同：错误路径没有 afterLLM（LLM 调用可能根本没发生）。
        即使 agent_hooks_via_middleware=True，afterAgent(error) 也在此补发 ——
        因为 SDK 图抛异常时 exit 节点（after_agent）不会执行。
        """
        dispatcher = self._resolve_hooks_dispatcher(config)
        after_agent = self._emit_hook_sync(
            dispatcher,
            AgentHookEvent.AFTER_AGENT,
            build_after_agent_context(
                self.backend_name,
                error=error,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.AFTER_AGENT, after_agent)
        after_stop = self._emit_hook_sync(
            dispatcher,
            AgentHookEvent.AFTER_STOP,
            build_after_stop_context(
                self.backend_name,
                reason="error",
                error=error,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.AFTER_STOP, after_stop)

    async def _finalize_run_error_async(self, config: AgentConfig | None, error: BaseException) -> None:
        """异步版 run 失败收尾：afterAgent → afterStop（error）。"""
        dispatcher = self._resolve_hooks_dispatcher(config)
        after_agent = await self._emit_hook_async(
            dispatcher,
            AgentHookEvent.AFTER_AGENT,
            build_after_agent_context(
                self.backend_name,
                error=error,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.AFTER_AGENT, after_agent)
        after_stop = await self._emit_hook_async(
            dispatcher,
            AgentHookEvent.AFTER_STOP,
            build_after_stop_context(
                self.backend_name,
                reason="error",
                error=error,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.AFTER_STOP, after_stop)

    async def _finalize_stream_success_async(self, config: AgentConfig | None, response: AgentResponse) -> None:
        """stream 成功收尾：afterLLM → afterAgent → afterStop。

        流式场景在全部 chunk 消费完后调用；response 由流中收集到的
        delta_content 拼接而成（见 _build_stream_response）。
        """
        dispatcher = self._resolve_hooks_dispatcher(config)
        if not self.call_hooks_via_middleware:
            after_llm = await self._emit_hook_async(
                dispatcher,
                AgentHookEvent.AFTER_LLM,
                build_after_llm_context(
                    self.backend_name,
                    response=response,
                    session_id=self._session_id,
                    correlation_id=self._correlation_id,
                ),
            )
            self._raise_if_hook_blocked(AgentHookEvent.AFTER_LLM, after_llm)
        if not self.agent_hooks_via_middleware:
            after_agent = await self._emit_hook_async(
                dispatcher,
                AgentHookEvent.AFTER_AGENT,
                build_after_agent_context(
                    self.backend_name,
                    response=response,
                    session_id=self._session_id,
                    correlation_id=self._correlation_id,
                ),
            )
            self._raise_if_hook_blocked(AgentHookEvent.AFTER_AGENT, after_agent)
        after_stop = await self._emit_hook_async(
            dispatcher,
            AgentHookEvent.AFTER_STOP,
            build_after_stop_context(
                self.backend_name,
                reason="complete",
                response=response,
                session_id=self._session_id,
                correlation_id=self._correlation_id,
            ),
        )
        self._raise_if_hook_blocked(AgentHookEvent.AFTER_STOP, after_stop)

    def _build_stream_response(self, final_content: str) -> AgentResponse:
        """基于收集到的 delta_content 构造流式收尾用响应。

        流式过程中逐块累加 delta_content 得到完整文本，收尾时用它构造
        AgentResponse 供 afterLLM / afterAgent / afterStop 的 Context 使用。
        """
        return AgentResponse(content=final_content, backend=self.backend_name)
