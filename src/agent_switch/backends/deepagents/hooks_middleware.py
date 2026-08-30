"""deepagents 中间件桥接：把 agent_switch 的 hooks dispatcher 桥接到 deepagents 中间件协议。

触发事件（全部位于 SDK 内部，顺序与文档一致）：
- ``beforeAgent`` + ``beforePrompt`` → ``before_agent`` / ``abefore_agent``（entry 节点，每次执行一次）
- ``beforeLLM`` / ``afterLLM`` → ``wrap_model_call`` / ``awrap_model_call``（每次 LLM 调用一次）
- ``beforeTool`` / ``afterTool`` / ``afterToolError`` → ``wrap_tool_call`` / ``awrap_tool_call``
- ``afterAgent`` → ``after_agent`` / ``aafter_agent``（exit 节点，仅成功路径）

``afterStop``（含 complete / error reason）仍由 adapter 层触发；异常路径下
``after_agent`` 不会执行，adapter 的错误收尾会补发 ``afterAgent(error)`` + ``afterStop(error)``。

注意：本模块依赖 ``langchain.agents.middleware``，仅由 deepagents 后端
（``adapter._build_agent``）延迟导入，``import agent_switch`` 不会加载它。
"""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents.middleware.types import AgentMiddleware

from agent_switch.backends.deepagents.mapping import (
    agent_messages_to_langchain,
    langchain_message_to_agent_message,
)
from agent_switch.exceptions import HookBlockedError
from agent_switch.hooks.context import (
    AfterToolErrorHookContext,
    AfterToolHookContext,
    BeforeToolHookContext,
)
from agent_switch.hooks.dispatcher import AgentHooksDispatcher
from agent_switch.hooks.emitter import (
    apply_messages_modify,
    build_after_agent_context,
    build_after_llm_context,
    build_before_agent_context,
    build_before_llm_context,
    build_before_prompt_context,
)
from agent_switch.hooks.enums import AgentHookEvent
from agent_switch.hooks.result import HookOutcome, HookResult
from agent_switch.types import AgentConfig, AgentMessage, AgentResponse, ToolCall

#: 会话标识：(session_id, correlation_id)
SessionIds = tuple[str | None, str | None]
#: 会话标识提供者（adapter 每次 run 时更新，中间件每次调用时读取）
SessionProvider = Callable[[], SessionIds]


class AgentHooksMiddleware(AgentMiddleware[Any, Any, Any]):
    """把 agent_switch hooks dispatcher 桥接到 deepagents 的 agent 级与调用级事件。"""

    def __init__(
        self,
        dispatcher: AgentHooksDispatcher,
        backend: str,
        config: AgentConfig | None = None,
        session_provider: SessionProvider | None = None,
    ) -> None:
        super().__init__()
        self._dispatcher = dispatcher
        self._backend = backend
        # 构建图时所用的 config：供 Context 构建器读取 system_prompt / model 等
        self._config = config
        # session_provider 闭包绑定 adapter 的 _session_id / _correlation_id：
        # 每次 run/stream 更新，中间件按调用实时读取，保证与 adapter 层同一会话
        self._session_provider: SessionProvider = session_provider or (lambda: (None, None))

    def _session_ids(self) -> SessionIds:
        return self._session_provider()

    @staticmethod
    def _raise_if_blocked(event: AgentHookEvent, result: HookResult) -> None:
        """hook 返回 BLOCK 时抛 HookBlockedError。

        注意：从中间件方法里抛出会沿着 langgraph 节点传播到 invoke/astream
        调用方（即 DeepAgentsAdapter.run/stream），从而中断整个 agent 运行 ——
        这与 adapter 层 BLOCK 的语义保持一致。
        """
        if result.outcome is HookOutcome.BLOCK:
            raise HookBlockedError(hook_event=event, reason=result.reason)

    # ---- agent 级（before_agent / after_agent，每次执行一次）----

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """agent 执行开始（entry 节点）：触发 beforeAgent → beforePrompt。

        langgraph 把实现了 ``before_agent`` 的中间件注册为图的入口节点，
        每次 agent 执行恰好运行一次。返回的 dict 会作为 state 更新合并进图，
        因此 beforePrompt 的 MODIFY 可以借此改写初始消息。
        """
        session_id, correlation_id = self._session_ids()
        # state["messages"] 是 LangChain 消息列表（不含 system）
        messages = _state_messages(state)
        session_result = self._dispatcher.emit_sync(
            AgentHookEvent.BEFORE_AGENT,
            build_before_agent_context(
                self._backend,
                messages,
                self._config,
                session_id=session_id,
                correlation_id=correlation_id,
            ),
        )
        self._raise_if_blocked(AgentHookEvent.BEFORE_AGENT, session_result)
        prompt_result = self._dispatcher.emit_sync(
            AgentHookEvent.BEFORE_PROMPT,
            build_before_prompt_context(
                self._backend,
                messages,
                self._config,
                session_id=session_id,
                correlation_id=correlation_id,
            ),
        )
        self._raise_if_blocked(AgentHookEvent.BEFORE_PROMPT, prompt_result)
        modified = apply_messages_modify(messages, prompt_result)
        if modified is not messages:
            # MODIFY：通过 state 更新改写初始消息（转回 LangChain 消息）
            return {"messages": agent_messages_to_langchain(modified)}
        return None

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """异步版：agent 执行开始，触发 beforeAgent → beforePrompt。"""
        session_id, correlation_id = self._session_ids()
        messages = _state_messages(state)
        session_result = await self._dispatcher.emit(
            AgentHookEvent.BEFORE_AGENT,
            build_before_agent_context(
                self._backend,
                messages,
                self._config,
                session_id=session_id,
                correlation_id=correlation_id,
            ),
        )
        self._raise_if_blocked(AgentHookEvent.BEFORE_AGENT, session_result)
        prompt_result = await self._dispatcher.emit(
            AgentHookEvent.BEFORE_PROMPT,
            build_before_prompt_context(
                self._backend,
                messages,
                self._config,
                session_id=session_id,
                correlation_id=correlation_id,
            ),
        )
        self._raise_if_blocked(AgentHookEvent.BEFORE_PROMPT, prompt_result)
        modified = apply_messages_modify(messages, prompt_result)
        if modified is not messages:
            return {"messages": agent_messages_to_langchain(modified)}
        return None

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """agent 执行结束（exit 节点，仅成功路径）：触发 afterAgent。"""
        session_id, correlation_id = self._session_ids()
        response = _state_to_response(state, self._backend)
        self._dispatcher.emit_sync(
            AgentHookEvent.AFTER_AGENT,
            build_after_agent_context(
                self._backend,
                response=response,
                session_id=session_id,
                correlation_id=correlation_id,
            ),
        )
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """异步版：agent 执行结束，触发 afterAgent。"""
        session_id, correlation_id = self._session_ids()
        response = _state_to_response(state, self._backend)
        await self._dispatcher.emit(
            AgentHookEvent.AFTER_AGENT,
            build_after_agent_context(
                self._backend,
                response=response,
                session_id=session_id,
                correlation_id=correlation_id,
            ),
        )
        return None

    # ---- LLM 调用级 ----

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """同步版：beforeLLM →（MODIFY 改写 messages）→ handler → afterLLM。"""
        session_id, correlation_id = self._session_ids()
        agent_messages = _request_messages(request)
        before_result = self._dispatcher.emit_sync(
            AgentHookEvent.BEFORE_LLM,
            build_before_llm_context(
                self._backend,
                agent_messages,
                session_id=session_id,
                correlation_id=correlation_id,
            ),
        )
        self._raise_if_blocked(AgentHookEvent.BEFORE_LLM, before_result)
        modified = apply_messages_modify(agent_messages, before_result)
        if modified is not agent_messages:
            request = request.override(messages=agent_messages_to_langchain(modified))
        response = handler(request)
        self._dispatcher.emit_sync(
            AgentHookEvent.AFTER_LLM,
            build_after_llm_context(
                self._backend,
                response=_response_to_agent_response(response, self._backend),
                session_id=session_id,
                correlation_id=correlation_id,
            ),
        )
        return response

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """异步版：beforeLLM →（MODIFY 改写 messages）→ handler → afterLLM。"""
        session_id, correlation_id = self._session_ids()
        agent_messages = _request_messages(request)
        before_result = await self._dispatcher.emit(
            AgentHookEvent.BEFORE_LLM,
            build_before_llm_context(
                self._backend,
                agent_messages,
                session_id=session_id,
                correlation_id=correlation_id,
            ),
        )
        self._raise_if_blocked(AgentHookEvent.BEFORE_LLM, before_result)
        modified = apply_messages_modify(agent_messages, before_result)
        if modified is not agent_messages:
            request = request.override(messages=agent_messages_to_langchain(modified))
        response = await handler(request)
        await self._dispatcher.emit(
            AgentHookEvent.AFTER_LLM,
            build_after_llm_context(
                self._backend,
                response=_response_to_agent_response(response, self._backend),
                session_id=session_id,
                correlation_id=correlation_id,
            ),
        )
        return response

    # ---- 工具调用级 ----

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """同步版：beforeTool → handler → afterTool；handler 异常时 afterToolError 后重抛。"""
        session_id, correlation_id = self._session_ids()
        tool_name = _request_tool_name(request)
        before_result = self._dispatcher.emit_sync(
            AgentHookEvent.BEFORE_TOOL,
            BeforeToolHookContext(
                backend=self._backend,
                session_id=session_id,
                correlation_id=correlation_id,
                tool_name=tool_name,
                tool_call=_request_tool_call(request),
            ),
        )
        self._raise_if_blocked(AgentHookEvent.BEFORE_TOOL, before_result)
        try:
            result = handler(request)
        except Exception as exc:
            self._dispatcher.emit_sync(
                AgentHookEvent.AFTER_TOOL_ERROR,
                AfterToolErrorHookContext(
                    backend=self._backend,
                    session_id=session_id,
                    correlation_id=correlation_id,
                    tool_name=tool_name,
                    error=exc,
                ),
            )
            raise
        self._dispatcher.emit_sync(
            AgentHookEvent.AFTER_TOOL,
            AfterToolHookContext(
                backend=self._backend,
                session_id=session_id,
                correlation_id=correlation_id,
                tool_name=tool_name,
                result=result,
            ),
        )
        return result

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """异步版：beforeTool → handler → afterTool；handler 异常时 afterToolError 后重抛。"""
        session_id, correlation_id = self._session_ids()
        tool_name = _request_tool_name(request)
        before_result = await self._dispatcher.emit(
            AgentHookEvent.BEFORE_TOOL,
            BeforeToolHookContext(
                backend=self._backend,
                session_id=session_id,
                correlation_id=correlation_id,
                tool_name=tool_name,
                tool_call=_request_tool_call(request),
            ),
        )
        self._raise_if_blocked(AgentHookEvent.BEFORE_TOOL, before_result)
        try:
            result = await handler(request)
        except Exception as exc:
            await self._dispatcher.emit(
                AgentHookEvent.AFTER_TOOL_ERROR,
                AfterToolErrorHookContext(
                    backend=self._backend,
                    session_id=session_id,
                    correlation_id=correlation_id,
                    tool_name=tool_name,
                    error=exc,
                ),
            )
            raise
        await self._dispatcher.emit(
            AgentHookEvent.AFTER_TOOL,
            AfterToolHookContext(
                backend=self._backend,
                session_id=session_id,
                correlation_id=correlation_id,
                tool_name=tool_name,
                result=result,
            ),
        )
        return result


# ---------------------------------------------------------------- 辅助函数

def _state_field(state: Any, key: str) -> Any:
    """读取图 state 的字段（兼容 dict 与对象两种形态）。"""
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)


def _state_messages(state: Any) -> list[AgentMessage]:
    """``state["messages"]``（LangChain 消息）→ agent_switch 消息。"""
    raw_messages = _state_field(state, "messages")
    if not isinstance(raw_messages, list):
        return []
    return [langchain_message_to_agent_message(message) for message in raw_messages]


def _state_to_response(state: Any, backend: str) -> AgentResponse:
    """由结束态构造 AgentResponse（content 取最后一条消息）。"""
    messages = _state_messages(state)
    agent_message = messages[-1] if messages else None
    return AgentResponse(
        content=agent_message.content if agent_message else "",
        message=agent_message,
        raw=state,
        backend=backend,
    )


def _request_messages(request: Any) -> list[AgentMessage]:
    """``ModelRequest.messages``（LangChain 消息，不含 system）→ agent_switch 消息。"""
    return [
        langchain_message_to_agent_message(message)
        for message in (getattr(request, "messages", None) or [])
    ]


def _response_to_agent_response(response: Any, backend: str) -> AgentResponse:
    """``ModelResponse`` → AgentResponse（content 取最后一条消息）。"""
    result = getattr(response, "result", None)
    agent_message: AgentMessage | None = None
    content = ""
    if isinstance(result, list) and result:
        agent_message = langchain_message_to_agent_message(result[-1])
        content = agent_message.content or ""
    return AgentResponse(content=content, message=agent_message, raw=response, backend=backend)


def _tool_call_field(tool_call: Any, field: str) -> Any:
    """兼容 dict（TypedDict）与对象两种形态的 ToolCall。"""
    if isinstance(tool_call, dict):
        return tool_call.get(field)
    return getattr(tool_call, field, None)


def _request_tool_call(request: Any) -> ToolCall | None:
    """``ToolCallRequest.tool_call`` → agent_switch ToolCall。"""
    tool_call = getattr(request, "tool_call", None)
    if tool_call is None:
        return None
    return ToolCall(
        id=str(_tool_call_field(tool_call, "id") or ""),
        name=str(_tool_call_field(tool_call, "name") or ""),
        arguments=_tool_call_field(tool_call, "args") or {},
    )


def _request_tool_name(request: Any) -> str:
    """工具名：优先 tool_call.name，其次 tool.name。"""
    name = _tool_call_field(getattr(request, "tool_call", None), "name")
    if name:
        return str(name)
    tool = getattr(request, "tool", None)
    return str(getattr(tool, "name", "") or "")
