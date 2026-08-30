"""deepagents hooks 中间件桥接测试。"""

from typing import Any

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_switch import AgentBackend, AgentConfig, AgentMessage, HookBlockedError, MessageRole, create_agent
from agent_switch.backends.deepagents.hooks_middleware import AgentHooksMiddleware
from agent_switch.hooks.base import BaseAgentHooks
from agent_switch.hooks.dispatcher import AgentHooksDispatcher
from agent_switch.hooks.enums import AgentHookEvent
from agent_switch.hooks.result import HookOutcome, HookResult


class TrackingHooks(BaseAgentHooks):
    """记录 agent 级与调用级事件。"""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def before_agent(self, context) -> None:
        self.events.append("beforeAgent")

    async def before_prompt(self, context) -> None:
        self.events.append("beforePrompt")

    async def before_llm(self, context) -> None:
        self.events.append("beforeLLM")

    async def after_llm(self, context) -> None:
        self.events.append("afterLLM")

    async def before_tool(self, context) -> None:
        self.events.append("beforeTool")

    async def after_tool(self, context) -> None:
        self.events.append("afterTool")

    async def after_tool_error(self, context) -> None:
        self.events.append("afterToolError")

    async def after_agent(self, context) -> None:
        self.events.append("afterAgent")


class FakeRequest:
    """模拟 ModelRequest：messages + override()。"""

    def __init__(self, messages: list[Any] | None = None) -> None:
        self.messages = messages if messages is not None else []

    def override(self, **overrides: Any) -> "FakeRequest":
        kwargs: dict[str, Any] = {"messages": self.messages}
        kwargs.update(overrides)
        return FakeRequest(**kwargs)


class FakeToolRequest:
    """模拟 ToolCallRequest。"""

    tool_call = {"id": "call_1", "name": "search", "args": {"q": "x"}}
    tool = None


def _middleware(events: list[str]) -> AgentHooksMiddleware:
    return AgentHooksMiddleware(AgentHooksDispatcher([TrackingHooks(events)]), backend="deepagents")


def test_before_agent_fires_session_and_prompt_events():
    """before_agent（entry 节点）：触发 beforeAgent → beforePrompt（保持顺序）。"""
    events: list[str] = []
    middleware = _middleware(events)
    state = {"messages": [HumanMessage(content="hello")]}
    updates = middleware.before_agent(state, None)
    assert events == ["beforeAgent", "beforePrompt"]
    assert updates is None  # 无 MODIFY 时不返回 state 更新


def test_before_agent_prompt_modify_returns_state_update():
    """beforePrompt 返回 MODIFY：before_agent 返回 state 更新 {"messages": [...]}。"""

    class ModifyHooks(BaseAgentHooks):
        async def before_prompt(self, context) -> HookResult | None:
            return HookResult(
                outcome=HookOutcome.MODIFY,
                data={"messages": [AgentMessage(role=MessageRole.USER, content="modified prompt")]},
            )

    middleware = AgentHooksMiddleware(AgentHooksDispatcher([ModifyHooks()]), backend="deepagents")
    updates = middleware.before_agent({"messages": [HumanMessage(content="original")]}, None)
    assert updates is not None
    messages = updates["messages"]
    assert len(messages) == 1
    assert messages[0].content == "modified prompt"


def test_after_agent_fires_after_agent_with_response():
    """after_agent（exit 节点）：触发 afterAgent，且带由结束态构造的 AgentResponse。"""
    events: list[str] = []
    middleware = _middleware(events)
    responses: list[Any] = []

    class CapturingHooks(BaseAgentHooks):
        async def after_agent(self, context) -> None:
            events.append("afterAgent")
            responses.append(context.response)

    middleware = AgentHooksMiddleware(AgentHooksDispatcher([CapturingHooks()]), backend="deepagents")
    updates = middleware.after_agent({"messages": [AIMessage(content="final")]}, None)
    assert events == ["afterAgent"]
    assert updates is None
    assert responses[0].content == "final"
    assert responses[0].backend == "deepagents"


async def test_abefore_agent_and_aafter_agent_async():
    """异步版：abefore_agent / aafter_agent 触发 beforeAgent/beforePrompt/afterAgent。"""
    events: list[str] = []
    middleware = _middleware(events)
    state = {"messages": [HumanMessage(content="hello")]}
    await middleware.abefore_agent(state, None)
    assert events == ["beforeAgent", "beforePrompt"]
    events.clear()
    await middleware.aafter_agent({"messages": [AIMessage(content="final")]}, None)
    assert events == ["afterAgent"]


def test_wrap_model_call_fires_before_and_after_llm():
    """同步 wrap_model_call：beforeLLM → handler → afterLLM。"""
    events: list[str] = []
    middleware = _middleware(events)

    def handler(request: Any) -> Any:
        return ModelResponse(result=[AIMessage(content="answer")])

    response = middleware.wrap_model_call(FakeRequest([HumanMessage(content="hello")]), handler)
    assert events == ["beforeLLM", "afterLLM"]
    assert response.result[0].content == "answer"


async def test_awrap_model_call_fires_before_and_after_llm():
    """异步 awrap_model_call：beforeLLM → handler → afterLLM。"""
    events: list[str] = []
    middleware = _middleware(events)

    async def handler(request: Any) -> Any:
        return ModelResponse(result=[AIMessage(content="answer")])

    response = await middleware.awrap_model_call(FakeRequest([HumanMessage(content="hello")]), handler)
    assert events == ["beforeLLM", "afterLLM"]
    assert response.result[0].content == "answer"


def test_before_llm_block_short_circuits_handler():
    """beforeLLM 返回 BLOCK：抛 HookBlockedError，handler 不被调用。"""

    class BlockHooks(BaseAgentHooks):
        async def before_llm(self, context) -> HookResult | None:
            return HookResult(outcome=HookOutcome.BLOCK, reason="blocked by policy")

    middleware = AgentHooksMiddleware(AgentHooksDispatcher([BlockHooks()]), backend="deepagents")
    called = False

    def handler(request: Any) -> Any:
        nonlocal called
        called = True
        return ModelResponse(result=[AIMessage(content="x")])

    with pytest.raises(HookBlockedError) as exc_info:
        middleware.wrap_model_call(FakeRequest(), handler)
    assert exc_info.value.hook_event is AgentHookEvent.BEFORE_LLM
    assert exc_info.value.reason == "blocked by policy"
    assert called is False


def test_before_llm_modify_rewrites_request_messages():
    """beforeLLM 返回 MODIFY：handler 收到的 request.messages 被替换。"""

    class ModifyHooks(BaseAgentHooks):
        async def before_llm(self, context) -> HookResult | None:
            return HookResult(
                outcome=HookOutcome.MODIFY,
                data={"messages": [AgentMessage(role=MessageRole.USER, content="modified")]},
            )

    middleware = AgentHooksMiddleware(AgentHooksDispatcher([ModifyHooks()]), backend="deepagents")
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(FakeRequest([HumanMessage(content="original")]), handler)
    assert len(seen[0].messages) == 1
    assert seen[0].messages[0].content == "modified"


def test_wrap_tool_call_events_and_error():
    """wrap_tool_call：beforeTool → handler → afterTool；异常时 afterToolError 后重抛。"""
    events: list[str] = []
    middleware = _middleware(events)

    # 正常路径
    result = middleware.wrap_tool_call(
        FakeToolRequest(), lambda request: ToolMessage(content="ok", tool_call_id="call_1")
    )
    assert events == ["beforeTool", "afterTool"]
    assert result.content == "ok"

    # 异常路径
    events.clear()

    def failing_handler(request: Any) -> Any:
        raise RuntimeError("tool boom")

    with pytest.raises(RuntimeError):
        middleware.wrap_tool_call(FakeToolRequest(), failing_handler)
    assert events == ["beforeTool", "afterToolError"]


def test_build_agent_injects_hooks_middleware_and_caches(monkeypatch):
    """_build_agent：配置 hooks 时注入 AgentHooksMiddleware；同一 config 复用缓存图。"""
    from agent_switch.backends import deepagents as deepagents_package
    from agent_switch.backends.deepagents import adapter as deepagents_adapter_module

    created: list[list[Any]] = []

    def fake_create_deep_agent(**kwargs: Any) -> Any:
        created.append(kwargs)
        return object()

    monkeypatch.setattr(
        deepagents_adapter_module, "import_create_deep_agent", lambda: fake_create_deep_agent
    )
    assert deepagents_package.DeepAgentsAdapter is not None  # 仅保证包可导入

    agent = create_agent(AgentBackend.DEEPAGENTS, AgentConfig(hooks=TrackingHooks([])))
    graph1 = agent._build_agent(agent._resolve_config(None))
    graph2 = agent._build_agent(agent._resolve_config(None))
    assert graph1 is graph2  # 缓存命中：只构建一次
    assert len(created) == 1
    assert any(isinstance(m, AgentHooksMiddleware) for m in created[0]["middleware"])
