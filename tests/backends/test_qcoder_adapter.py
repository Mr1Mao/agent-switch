"""QcoderAdapter 测试（mock _query_impl，不依赖真实 qodercli）。"""

import pytest
from qoder_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

from agent_core import AgentBackend, AgentConfig, AgentResponse, BaseAgentHooks, create_agent
from agent_core.backends.qcoder.adapter import QcoderAdapter


async def _msg_stream(*, prompt=None, options=None, transport=None):
    """fake query：一条 assistant 回复 + ResultMessage 终结。"""
    yield AssistantMessage(content=[TextBlock(text="real answer")], model="mock")
    yield ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        result="real answer",
        stop_reason="end_turn",
    )


def test_run_returns_response_from_result_message(monkeypatch):
    """run：ResultMessage 终结 → AgentResponse。"""
    monkeypatch.setattr(QcoderAdapter, "_query_impl", lambda self: _msg_stream)
    agent = create_agent(AgentBackend.QCODER)
    response = agent.run("hello")
    assert isinstance(response, AgentResponse)
    assert response.content == "real answer"
    assert response.backend == "qcoder"


async def test_stream_maps_chunks_and_finish(monkeypatch):
    """stream：assistant 消息映射为 AgentChunk，最后 yield is_finish。"""
    monkeypatch.setattr(QcoderAdapter, "_query_impl", lambda self: _msg_stream)
    agent = create_agent(AgentBackend.QCODER)
    chunks = [chunk async for chunk in agent.stream("hello")]
    assert chunks[0].delta_content == "real answer"
    assert chunks[-1].is_finish is True
    assert len(chunks) == 2


def test_run_invokes_qoder_hooks_bridge(monkeypatch):
    """配置了工具 hook 时，QoderAgentOptions.hooks 注入 PreToolUse 桥接。"""
    captured: dict[str, object] = {}

    async def capturing_query(*, prompt=None, options=None, transport=None):
        captured["options"] = options
        yield AssistantMessage(content=[TextBlock(text="ok")], model="mock")
        yield ResultMessage(
            subtype="result", duration_ms=1, duration_api_ms=1, is_error=False,
            num_turns=1, session_id="s", result="ok", stop_reason="end_turn",
        )

    class ToolHooks(BaseAgentHooks):
        async def before_tool(self, context):
            return None

    monkeypatch.setattr(QcoderAdapter, "_query_impl", lambda self: capturing_query)
    agent = create_agent(AgentBackend.QCODER, AgentConfig(hooks=ToolHooks()))
    agent.run("hello")
    hooks = captured["options"].hooks  # type: ignore[union-attr]
    assert hooks is not None
    assert "PreToolUse" in hooks


def test_run_falls_back_to_collected_text(monkeypatch):
    """未收到 ResultMessage（流提前结束）时用累计文本兜底。"""

    async def stream_no_result(*, prompt=None, options=None, transport=None):
        yield AssistantMessage(content=[TextBlock(text="partial")], model="mock")

    monkeypatch.setattr(QcoderAdapter, "_query_impl", lambda self: stream_no_result)
    agent = create_agent(AgentBackend.QCODER)
    response = agent.run("hello")
    assert response.content == "partial"


def test_run_error_path_emits_after_agent_and_after_stop(monkeypatch):
    """SDK 抛异常：错误路径触发 afterAgent + afterStop(error)。"""
    events: list[str] = []

    async def failing_query(*, prompt=None, options=None, transport=None):
        raise RuntimeError("sdk exploded")
        yield  # pragma: no cover - 不可达

    class TrackingHooks(BaseAgentHooks):
        async def after_agent(self, context):
            events.append("afterAgent")

        async def after_stop(self, context):
            events.append(f"afterStop:{context.reason}")

    monkeypatch.setattr(QcoderAdapter, "_query_impl", lambda self: failing_query)
    agent = create_agent(AgentBackend.QCODER, AgentConfig(hooks=TrackingHooks()))
    with pytest.raises(RuntimeError):
        agent.run("hello")
    assert events == ["afterAgent", "afterStop:error"]
