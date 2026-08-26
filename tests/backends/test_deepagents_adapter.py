"""DeepAgentsAdapter 测试（mock _build_agent，不依赖真实 deepagents 图）。"""

from langchain_core.messages import AIMessage, AIMessageChunk

from agent_core import AgentBackend, AgentConfig, AgentResponse, BaseAgentHooks, create_agent


def _fake_graph():
    """模拟 deepagents 图：invoke 返回 AIMessage；astream 产生 (namespace, chunk) 元组。"""

    class FakeGraph:
        def invoke(self, state):
            return {"messages": [AIMessage(content="final answer")]}

        async def astream(self, state, stream_mode="messages"):
            yield (("messages", 0), AIMessageChunk(content="hello"))
            yield (("messages", 1), AIMessageChunk(content=" world"))

    return FakeGraph()


def test_run_with_mocked_build_agent():
    """run：mock 图后返回 AgentResponse，content 取最后一条消息。"""
    agent = create_agent(AgentBackend.DEEPAGENTS)
    agent._build_agent = lambda config: _fake_graph()  # type: ignore[method-assign]
    response = agent.run("hi")
    assert isinstance(response, AgentResponse)
    assert response.content == "final answer"
    assert response.backend == "deepagents"
    assert response.message is not None


async def test_stream_yields_chunks_and_finish():
    """stream：LangChain chunk 映射为 AgentChunk，最后 yield is_finish。"""
    agent = create_agent(AgentBackend.DEEPAGENTS)
    agent._build_agent = lambda config: _fake_graph()  # type: ignore[method-assign]
    chunks = [chunk async for chunk in agent.stream("hi")]
    assert chunks[-1].is_finish is True
    content = "".join(chunk.delta_content for chunk in chunks)
    assert content == "hello world"


def test_run_success_path_emits_after_stop_only():
    """deepagents 成功路径：adapter 层只触发 afterStop（其余事件由中间件在 SDK 内触发）。"""
    events: list[str] = []

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

    agent = create_agent(AgentBackend.DEEPAGENTS, AgentConfig(hooks=TrackingHooks()))
    agent._build_agent = lambda config: _fake_graph()  # type: ignore[method-assign]
    response = agent.run("hi")
    assert response.content == "final answer"
    # session/call 级事件由注入的 AgentHooksMiddleware 触发；mock 图没有中间件，故只有 afterStop
    assert events == ["afterStop"]


def test_run_error_path_emits_after_agent_and_after_stop():
    """deepagents 错误路径：after_agent 不执行，adapter 补发 afterAgent(error) + afterStop(error)。"""
    events: list[str] = []

    class TrackingHooks(BaseAgentHooks):
        async def before_agent(self, context):
            events.append("beforeAgent")

        async def after_agent(self, context):
            events.append("afterAgent")

        async def after_stop(self, context):
            events.append(f"afterStop:{context.reason}")

    agent = create_agent(AgentBackend.DEEPAGENTS, AgentConfig(hooks=TrackingHooks()))
    import pytest

    class ExplodingGraph:
        def invoke(self, state):
            raise RuntimeError("sdk exploded")

    agent._build_agent = lambda config: ExplodingGraph()  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        agent.run("hi")
    assert events == ["afterAgent", "afterStop:error"]


async def test_run_success_async_finalize_with_middleware_flags():
    """回归：_finalize_run_success_async 在 agent_hooks_via_middleware=True 时不崩溃。"""
    events: list[str] = []

    class TrackingHooks(BaseAgentHooks):
        async def after_llm(self, context):
            events.append("afterLLM")

        async def after_agent(self, context):
            events.append("afterAgent")

        async def after_stop(self, context):
            events.append("afterStop")

    agent = create_agent(AgentBackend.DEEPAGENTS, AgentConfig(hooks=TrackingHooks()))
    response = AgentResponse(content="ok", backend="deepagents")
    await agent._finalize_run_success_async(agent._resolve_config(None), response)
    # 中间件开关为 True：adapter 层只保留 afterStop，其余事件由 SDK 内部触发
    assert events == ["afterStop"]
