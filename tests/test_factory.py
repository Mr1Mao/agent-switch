"""create_agent 工厂测试。"""

import pytest
from langchain_core.messages import AIMessage

from agent_switch import (
    AgentBackend,
    AgentResponse,
    BackendNotFoundError,
    DeepAgentsAdapter,
    QcoderAdapter,
    create_agent,
)


def test_create_agent_deepagents_returns_adapter():
    """AgentBackend 枚举创建 DeepAgentsAdapter。"""
    agent = create_agent(AgentBackend.DEEPAGENTS)
    assert isinstance(agent, DeepAgentsAdapter)
    assert agent.backend_name == "deepagents"


def test_create_agent_qcoder_returns_adapter():
    """字符串 backend 创建 QcoderAdapter。"""
    agent = create_agent("qcoder")
    assert isinstance(agent, QcoderAdapter)
    assert agent.backend_name == "qcoder"


def test_create_agent_unknown_backend_raises():
    """未注册的 backend 抛 BackendNotFoundError。"""
    with pytest.raises(BackendNotFoundError) as exc_info:
        create_agent("not-a-backend")
    assert exc_info.value.backend == "not-a-backend"


def test_deepagents_run_with_mocked_build_agent():
    """mock _build_agent 后验证 deepagents run 的完整流程与结果解析。"""
    agent = create_agent(AgentBackend.DEEPAGENTS)

    class FakeGraph:
        def invoke(self, state):
            return {"messages": [AIMessage(content="hello from deepagents")]}

    agent._build_agent = lambda config: FakeGraph()  # type: ignore[method-assign]
    response = agent.run("hello")
    assert isinstance(response, AgentResponse)
    assert response.backend == "deepagents"
    assert response.content == "hello from deepagents"
    assert response.message is not None
    assert response.message.role == "assistant"


async def test_qcoder_stream_yields_chunks_and_finish(qcoder_fake_query):
    """qcoder stream：mock _query_impl 后验证 AgentChunk 映射与 is_finish。"""
    agent = create_agent(AgentBackend.QCODER)
    chunks = [chunk async for chunk in agent.stream("hello")]
    assert chunks[-1].is_finish is True
    # fake query 回显输入文本 → 一条 delta_content，随后 ResultMessage 终结
    assert chunks[0].delta_content == "[stub] hello"
    assert len(chunks) == 2
