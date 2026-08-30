"""qcoder 翻译层（消息归一化 + options 映射）测试。"""

from qoder_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)

from agent_switch import (
    AgentConfig,
    AgentMcpConfig,
    AgentMcpServer,
    AgentMessage,
    AgentModel,
    AgentSkillsConfig,
    AgentTool,
    MessageRole,
    ToolCall,
    ToolResult,
)
from agent_switch.backends.qcoder.mapping import (
    agent_messages_to_qoder_wire,
    build_qoder_agent_options,
    qoder_message_to_agent_chunks,
    qoder_message_to_agent_message,
    qoder_message_to_agent_response,
)

# ---------------------------------------------------------------- 输入方向

def test_user_message_to_wire():
    """user 消息 → wire：{"type":"user","message":{"role":"user","content":...}}。"""
    wire = agent_messages_to_qoder_wire([AgentMessage(role=MessageRole.USER, content="hello")])
    assert len(wire) == 1
    assert wire[0]["type"] == "user"
    assert wire[0]["message"]["role"] == "user"
    assert wire[0]["message"]["content"] == "hello"
    assert wire[0]["parent_tool_use_id"] is None
    assert wire[0]["session_id"] is None


def test_system_message_not_in_wire():
    """system 消息不进 wire（由 QoderAgentOptions.system_prompt 承担）。"""
    wire = agent_messages_to_qoder_wire([AgentMessage(role=MessageRole.SYSTEM, content="sys")])
    assert wire == []


def test_assistant_tool_calls_to_wire():
    """assistant 文本 + 工具调用 → wire 的 text / tool_use content blocks。"""
    message = AgentMessage(
        role=MessageRole.ASSISTANT,
        content="answer",
        tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "x"})],
    )
    wire = agent_messages_to_qoder_wire([message])[0]
    blocks = wire["message"]["content"]
    assert blocks[0] == {"type": "text", "text": "answer"}
    assert blocks[1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "search",
        "input": {"q": "x"},
    }


def test_tool_message_to_wire():
    """tool 消息 → wire 的 tool_result content block。"""
    message = AgentMessage(
        role=MessageRole.TOOL,
        content="42",
        tool_result=ToolResult(tool_call_id="call_1", content="42"),
    )
    wire = agent_messages_to_qoder_wire([message])[0]
    block = wire["message"]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    assert block["content"] == "42"
    assert block["is_error"] is False


# ---------------------------------------------------------------- 输出方向

def test_assistant_message_to_agent():
    """AssistantMessage → assistant：ThinkingBlock→thinking、ToolUseBlock→tool_calls。"""
    sdk_message = AssistantMessage(
        content=[
            ThinkingBlock(thinking="t", signature="s"),
            TextBlock(text="hi"),
            ToolUseBlock(id="call_1", name="search", input={"q": "x"}),
        ],
        model="mock",
        stop_reason="tool_use",
    )
    agent = qoder_message_to_agent_message(sdk_message)
    assert agent is not None
    assert agent.role is MessageRole.ASSISTANT
    assert agent.content == "hi"
    assert agent.thinking == "t"
    assert agent.tool_calls[0].id == "call_1"
    assert agent.tool_calls[0].arguments == {"q": "x"}
    assert agent.meta["stop_reason"] == "tool_use"


def test_result_message_to_response():
    """ResultMessage → 终结 AgentResponse。"""
    sdk_message = ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        result="final answer",
        stop_reason="end_turn",
    )
    response = qoder_message_to_agent_response(sdk_message)
    assert response is not None
    assert response.content == "final answer"
    assert response.backend == "qcoder"


def test_non_result_message_returns_none():
    """非 ResultMessage 不产生终结响应。"""
    sdk_message = AssistantMessage(content=[TextBlock(text="x")], model="mock")
    assert qoder_message_to_agent_response(sdk_message) is None


def test_assistant_message_to_chunks():
    """AssistantMessage → 多个 AgentChunk（thinking / content / tool_call）。"""
    sdk_message = AssistantMessage(
        content=[
            ThinkingBlock(thinking="t", signature="s"),
            TextBlock(text="hi"),
            ToolUseBlock(id="call_1", name="search", input={}),
        ],
        model="mock",
    )
    chunks = qoder_message_to_agent_chunks(sdk_message)
    assert chunks[0].delta_thinking == "t"
    assert chunks[1].delta_content == "hi"
    assert chunks[2].delta_tool_call.name == "search"


# ---------------------------------------------------------------- options 映射

def test_build_options_maps_config():
    """AgentConfig → QoderAgentOptions：model/system_prompt/skills/tools/extra。"""
    config = AgentConfig(
        model=AgentModel(name="deepseek-v4-flash"),
        system_prompt="be terse",
        tools=[
            AgentTool(
                name="search",
                description="Search the web",
                handler=lambda args: {"content": [{"type": "text", "text": "ok"}]},
            )
        ],
        skills=AgentSkillsConfig(sources=["skill_a"]),
        extra={"permission_mode": "bypassPermissions"},
    )
    options = build_qoder_agent_options(config)
    assert options.model == "deepseek-v4-flash"
    assert options.system_prompt == "be terse"
    assert options.skills == ["skill_a"]
    assert options.permission_mode == "bypassPermissions"
    # 工具 handler → 进程内 SDK MCP server + allowed_tools
    assert "agent_tools" in options.mcp_servers
    assert options.allowed_tools == ["search"]
    # 默认 auth：复用本机 qodercli 登录态
    assert options.auth.type == "qodercli"


def test_build_options_extra_model_priority():
    """extra["model"]（str）优先于 AgentModel.name。"""
    config = AgentConfig(model=AgentModel(name="fallback"), extra={"model": "primary"})
    options = build_qoder_agent_options(config)
    assert options.model == "primary"


def test_build_options_mcp_config():
    """AgentMcpConfig → options.mcp_servers / allowed_mcp_server_names。"""
    config = AgentConfig(
        mcp=AgentMcpConfig(
            servers=[
                AgentMcpServer(
                    name="fs",
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-filesystem"],
                )
            ],
            allowed_server_names=["fs"],
        )
    )
    options = build_qoder_agent_options(config)
    assert options.mcp_servers["fs"]["type"] == "stdio"
    assert options.mcp_servers["fs"]["command"] == "npx"
    assert options.allowed_mcp_server_names == ["fs"]
