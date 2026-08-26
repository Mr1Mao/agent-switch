"""deepagents 映射层测试。"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_core import (
    AgentConfig,
    AgentMessage,
    AgentModel,
    AgentSkillsConfig,
    AgentSubagent,
    AgentTool,
    MessageRole,
    ToolCall,
    ToolResult,
)
from agent_core.backends.deepagents.mapping import (
    agent_messages_to_langchain,
    build_create_agent_kwargs,
    langchain_message_to_agent_message,
    resolve_chat_model,
)


def test_user_message_to_langchain():
    """user → HumanMessage。"""
    langchain_messages = agent_messages_to_langchain(
        [AgentMessage(role=MessageRole.USER, content="hello")]
    )
    assert len(langchain_messages) == 1
    assert isinstance(langchain_messages[0], HumanMessage)
    assert langchain_messages[0].content == "hello"


def test_system_message_to_langchain():
    """system → SystemMessage。"""
    langchain_messages = agent_messages_to_langchain(
        [AgentMessage(role=MessageRole.SYSTEM, content="be terse")]
    )
    assert isinstance(langchain_messages[0], SystemMessage)
    assert langchain_messages[0].content == "be terse"


def test_assistant_tool_calls_roundtrip():
    """assistant + tool_calls 与 LangChain AIMessage 双向转换。"""
    source = AgentMessage(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "pydantic"})],
    )
    langchain_message = agent_messages_to_langchain([source])[0]
    assert isinstance(langchain_message, AIMessage)
    assert langchain_message.tool_calls[0]["id"] == "call_1"
    assert langchain_message.tool_calls[0]["name"] == "search"
    assert langchain_message.tool_calls[0]["args"] == {"q": "pydantic"}

    back = langchain_message_to_agent_message(langchain_message)
    assert back.role is MessageRole.ASSISTANT
    assert back.tool_calls[0].id == "call_1"
    assert back.tool_calls[0].name == "search"
    assert back.tool_calls[0].arguments == {"q": "pydantic"}
    assert back.meta["langchain_type"] == "ai"
    assert back._raw is langchain_message


def test_tool_message_roundtrip():
    """tool 消息与 LangChain ToolMessage 双向转换。"""
    source = AgentMessage(
        role=MessageRole.TOOL,
        content="42",
        tool_result=ToolResult(tool_call_id="call_1", content="42"),
    )
    langchain_message = agent_messages_to_langchain([source])[0]
    assert isinstance(langchain_message, ToolMessage)
    assert langchain_message.tool_call_id == "call_1"

    back = langchain_message_to_agent_message(langchain_message)
    assert back.role is MessageRole.TOOL
    assert back.content == "42"
    assert back.tool_result is not None
    assert back.tool_result.tool_call_id == "call_1"


def test_thinking_extraction_priority():
    """thinking 提取优先级：reasoning_content > content_blocks 中的 reasoning 块。"""
    message = AIMessage(
        content="answer",
        additional_kwargs={"reasoning_content": "r1", "thinking": "r2"},
    )
    back = langchain_message_to_agent_message(message)
    assert back.thinking == "r1"

    class FakeMessage:
        type = "ai"
        content = "answer"
        additional_kwargs = {}
        tool_calls = []
        content_blocks = [{"type": "reasoning", "text": "block thinking"}]

    back2 = langchain_message_to_agent_message(FakeMessage())
    assert back2.thinking == "block thinking"


def test_resolve_chat_model_priority_order():
    """resolve_chat_model 优先级：extra['model'] > 纯 name 字符串。"""

    class FakeChatModel:
        pass

    # 1) extra["model"] 优先：已构建的 ChatModel 直接透传
    fake = FakeChatModel()
    config = AgentConfig(model=AgentModel(name="deepseek-v4-flash"), extra={"model": fake})
    assert resolve_chat_model(config) is fake

    # 2) 仅 AgentModel.name（无 api_key/base_url）→ 返回模型名字符串
    config2 = AgentConfig(model=AgentModel(name="openai:gpt-4o-mini"))
    assert resolve_chat_model(config2) == "openai:gpt-4o-mini"


def test_build_create_agent_kwargs_mapping():
    """AgentConfig → create_deep_agent kwargs 的映射（model/skills/subagents/白名单）。"""
    middleware = object()
    config = AgentConfig(
        model=AgentModel(name="deepseek-v4-flash"),
        system_prompt="be concise",
        tools=[AgentTool(name="t1", handler=lambda x: x)],
        skills=AgentSkillsConfig(sources=["skill_a"]),
        subagents=[AgentSubagent(name="s1", description="d", system_prompt="sp")],
        extra={"middleware": [middleware], "debug": True, "model": "preset-model"},
    )
    kwargs = build_create_agent_kwargs(config)
    assert kwargs["model"] == "preset-model"  # extra model 优先
    assert kwargs["system_prompt"] == "be concise"
    assert callable(kwargs["tools"][0])
    assert kwargs["skills"] == ["skill_a"]
    assert kwargs["subagents"][0]["name"] == "s1"
    assert kwargs["middleware"] == [middleware]
    assert kwargs["debug"] is True
