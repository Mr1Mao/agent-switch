"""qcoder 后端翻译层：agent_switch 类型 ↔ qoder-agent-sdk 类型。

负责两部分：
1. **消息格式归一化**（双向）：agent_switch ``AgentMessage`` ↔ qoder CLI wire 格式
   / SDK ``Message`` 对象（AssistantMessage / UserMessage / ResultMessage 等）；
2. **配置映射**：``AgentConfig`` → ``QoderAgentOptions``。

外部依赖（qoder-agent-sdk / mcp）全部延迟导入，未安装时抛 ``BackendDependencyError``。
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from typing import Any, AsyncIterator

from agent_switch.exceptions import BackendDependencyError
from agent_switch.types import (
    AgentChunk,
    AgentConfig,
    AgentMessage,
    AgentResponse,
    AgentTool,
    MessageRole,
    ToolCall,
)

_INSTALL_HINT = "pip install 'agent-switch[qcoder]'"

#: QoderAgentOptions 的 extra 白名单透传键
QODER_EXTRA_KEYS: tuple[str, ...] = (
    "permission_mode",
    "max_turns",
    "session_id",
    "cwd",
    "auth",
    "allowed_tools",
    "disallowed_tools",
    "can_use_tool",
    "include_partial_messages",
    "continue_conversation",
    "resume",
    "settings",
    "agents",
    "agent",
    "user",
    "env",
    "cli_path",
)


def _import_sdk() -> Any:
    """延迟导入 qoder-agent-sdk；未安装时抛 BackendDependencyError。"""
    try:
        sdk: Any = importlib.import_module("qoder_agent_sdk")
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise BackendDependencyError(backend="qcoder", install_hint=_INSTALL_HINT) from exc
    return sdk


# ---------------------------------------------------------------- 输入方向

def _arguments_to_qoder(arguments: dict[str, Any] | str) -> dict[str, Any]:
    """``ToolCall.arguments`` → Qoder ``ToolUseBlock.input``（必须是 dict）。"""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments) if arguments else {}
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def agent_messages_to_qoder_wire(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """agent_switch 消息 → qoder CLI wire 格式消息列表（输入方向）。

    Qoder CLI 的输入协议只有 ``user`` 类型消息：assistant 的文本与工具调用
    （``tool_use`` 块）以及 tool 结果（``tool_result`` 块）都以 content blocks
    的形式嵌在 user 消息里。``system`` 消息不进入 wire（由
    ``QoderAgentOptions.system_prompt`` 承担）；``thinking`` / ``meta`` 不发送。
    """
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role is MessageRole.SYSTEM:
            continue
        if message.role is MessageRole.TOOL:
            content: Any = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_result.tool_call_id if message.tool_result else "",
                    "content": message.content,
                    "is_error": False,
                }
            ]
        elif message.role is MessageRole.ASSISTANT:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": _arguments_to_qoder(call.arguments),
                    }
                )
            content = blocks if blocks else ""
        else:  # MessageRole.USER
            content = message.content
        wire.append(
            {
                "type": "user",
                "message": {"role": "user", "content": content},
                "parent_tool_use_id": None,
                "session_id": None,
            }
        )
    return wire


async def qoder_wire_iter(messages: list[AgentMessage]) -> AsyncIterator[dict[str, Any]]:
    """把 wire 消息列表包成异步迭代器（``query(prompt=...)`` 的输入形态）。"""
    for wire in agent_messages_to_qoder_wire(messages):
        yield wire


# ---------------------------------------------------------------- 输出方向

def _qoder_content_text(content: Any, sdk: Any) -> str:
    """提取 content 中的文本（str 或 TextBlock 列表）。"""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, sdk.TextBlock):
            parts.append(block.text)
        elif isinstance(block, sdk.ToolResultBlock) and isinstance(block.content, str):
            parts.append(block.content)
    return "".join(parts)


def qoder_message_to_agent_message(message: Any) -> AgentMessage | None:
    """qoder SDK ``Message`` → agent_switch 消息；无法归一的类型返回 None。

    映射：AssistantMessage → assistant（TextBlock→content、ThinkingBlock→thinking、
    ToolUseBlock→tool_calls）；UserMessage → user；SystemMessage → system（仅 meta）。
    """
    sdk = _import_sdk()
    if isinstance(message, sdk.AssistantMessage):
        content_parts: list[str] = []
        thinking: str | None = None
        tool_calls: list[ToolCall] = []
        for block in getattr(message, "content", None) or []:
            if isinstance(block, sdk.TextBlock):
                content_parts.append(block.text)
            elif isinstance(block, sdk.ThinkingBlock):
                thinking = block.thinking
            elif isinstance(block, sdk.ToolUseBlock):
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                )
        return AgentMessage(
            role=MessageRole.ASSISTANT,
            content="".join(content_parts),
            thinking=thinking,
            tool_calls=tool_calls,
            meta={
                "qoder_type": "assistant",
                "model": getattr(message, "model", None),
                "stop_reason": getattr(message, "stop_reason", None),
            },
            _raw=message,
        )
    if isinstance(message, sdk.UserMessage):
        return AgentMessage(
            role=MessageRole.USER,
            content=_qoder_content_text(message.content, sdk),
            meta={"qoder_type": "user"},
            _raw=message,
        )
    if isinstance(message, sdk.SystemMessage):
        return AgentMessage(
            role=MessageRole.SYSTEM,
            content="",
            meta={"qoder_type": "system", "subtype": message.subtype, "data": message.data},
            _raw=message,
        )
    return None


def qoder_message_to_agent_response(message: Any) -> AgentResponse | None:
    """``ResultMessage`` → 终结 AgentResponse；其它消息返回 None。

    ResultMessage 是 qoder 一次查询的终止消息，携带 ``result``（最终回复文本）、
    ``stop_reason``、``is_error``、``total_cost_usd`` 等元信息。
    """
    sdk = _import_sdk()
    if not isinstance(message, sdk.ResultMessage):
        return None
    content = message.result or ""
    agent_message: AgentMessage | None = None
    if content:
        agent_message = AgentMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            meta={
                "qoder_type": "result",
                "stop_reason": message.stop_reason,
                "is_error": message.is_error,
            },
            _raw=message,
        )
    return AgentResponse(content=content, message=agent_message, raw=message, backend="qcoder")


def qoder_message_to_agent_chunks(message: Any) -> list[AgentChunk]:
    """qoder ``Message`` → ``AgentChunk`` 列表。

    qoder 的 ``query()`` 默认按「每次模型回复」产出完整的 AssistantMessage
    （非 token 级流），因此一条消息会映射为 0~多个 AgentChunk：
    thinking → ``delta_thinking``、text → ``delta_content``、工具调用 → ``delta_tool_call``。
    """
    sdk = _import_sdk()
    if not isinstance(message, sdk.AssistantMessage):
        return []
    chunks: list[AgentChunk] = []
    thinking: str | None = None
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in getattr(message, "content", None) or []:
        if isinstance(block, sdk.TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, sdk.ThinkingBlock):
            thinking = block.thinking
        elif isinstance(block, sdk.ToolUseBlock):
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input or {}))
    if thinking:
        chunks.append(AgentChunk(delta_thinking=thinking))
    if text_parts:
        chunks.append(AgentChunk(delta_content="".join(text_parts)))
    for tool_call in tool_calls:
        chunks.append(AgentChunk(delta_tool_call=tool_call))
    return chunks


# ---------------------------------------------------------------- 配置映射

def _resolve_model(config: AgentConfig) -> str | None:
    """解析 Qoder 使用的模型名：``extra["model"]``（str）优先，其次 AgentModel.name。"""
    extra_model = config.extra.get("model")
    if isinstance(extra_model, str):
        return extra_model
    if config.model and config.model.name:
        return config.model.name
    return None


def _tool_schema(tool: AgentTool) -> dict[str, Any]:
    """AgentTool.parameters → JSON Schema（供 SdkMcpTool.input_schema）。"""
    params = tool.parameters or {}
    if isinstance(params, dict) and params.get("type") == "object":
        return params
    return {"type": "object", "properties": params}


def _ensure_async(handler: Any) -> Any:
    """保证工具 handler 是 async 函数（SDK 要求 ``await handler(args)``）。

    同步函数包装为 ``asyncio.to_thread`` 执行，避免阻塞事件循环；
    handler 的约定签名：``(args: dict) -> dict``。
    """
    if inspect.iscoroutinefunction(handler):
        return handler

    async def _wrapped(args: dict[str, Any]) -> Any:
        return await asyncio.to_thread(handler, args)

    return _wrapped


def build_qoder_agent_options(
    config: AgentConfig,
    hooks: dict[str, list[Any]] | None = None,
    *,
    session_id: str | None = None,
) -> Any:
    """``AgentConfig`` → ``QoderAgentOptions``。

    映射规则：
    - ``model``：extra["model"]（str）或 AgentModel.name；
    - ``system_prompt`` → options.system_prompt；
    - ``tools``：带 handler 的 AgentTool 注册为进程内 SDK MCP server
      （``create_sdk_mcp_server``），工具名进入 ``allowed_tools``；
    - ``skills`` → options.skills（enable_all → "all"）；
    - ``mcp``（AgentMcpConfig）→ options.mcp_servers / allowed_mcp_server_names；
    - ``extra`` 白名单键直接透传（permission_mode / max_turns / auth ...）；
    - ``auth`` 默认 ``qodercli_auth()``（复用本机登录态）。
    """
    sdk = _import_sdk()
    kwargs: dict[str, Any] = {}

    model = _resolve_model(config)
    if model:
        kwargs["model"] = model
    if config.system_prompt:
        kwargs["system_prompt"] = config.system_prompt

    # 自定义工具 handler → 进程内 SDK MCP server
    mcp_servers: dict[str, Any] = {}
    allowed_tools: list[str] = []
    handlers = [tool for tool in config.tools if tool.handler is not None]
    if handlers:
        server_name = "agent_tools"
        sdk_tools = [
            sdk.SdkMcpTool(
                name=tool.name,
                description=tool.description or tool.name,
                input_schema=_tool_schema(tool),
                handler=_ensure_async(tool.handler),
            )
            for tool in handlers
        ]
        mcp_servers[server_name] = sdk.create_sdk_mcp_server(name=server_name, tools=sdk_tools)
        allowed_tools.extend(tool.name for tool in handlers)

    # AgentSkillsConfig
    if config.skills:
        if config.skills.enable_all:
            kwargs["skills"] = "all"
        elif config.skills.sources:
            kwargs["skills"] = list(config.skills.sources)

    # AgentMcpConfig
    if config.mcp and config.mcp.servers:
        for server in config.mcp.servers:
            if server.name in mcp_servers:
                continue
            if server.url:
                # URL 形式的服务器 → Qoder http MCP server 配置
                mcp_servers[server.name] = {"type": "http", "url": server.url}
            elif server.command:
                # 本地命令形式 → Qoder stdio MCP server 配置
                mcp_servers[server.name] = {
                    "type": "stdio",
                    "command": server.command,
                    "args": list(server.args or []),
                }
        if config.mcp.allowed_server_names:
            kwargs["allowed_mcp_server_names"] = list(config.mcp.allowed_server_names)

    if mcp_servers:
        kwargs["mcp_servers"] = mcp_servers
    if allowed_tools:
        kwargs["allowed_tools"] = allowed_tools
    if session_id:
        kwargs["session_id"] = session_id

    # extra 白名单透传
    for key in QODER_EXTRA_KEYS:
        if key in config.extra:
            kwargs[key] = config.extra[key]

    if hooks:
        kwargs["hooks"] = hooks
    if "auth" not in kwargs:
        kwargs["auth"] = sdk.qodercli_auth()

    return sdk.QoderAgentOptions(**kwargs)
