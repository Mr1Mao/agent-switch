"""deepagents 适配器的翻译层：agent_core 类型 ↔ LangChain / deepagents 类型。

外部依赖（deepagents / langchain）全部延迟导入，未安装时抛 ``BackendDependencyError``。
"""

from __future__ import annotations

import importlib
import json
from typing import Any

from agent_core.exceptions import BackendDependencyError
from agent_core.types import (
    AgentConfig,
    AgentMessage,
    AgentSubagent,
    MessageRole,
    ToolCall,
    ToolResult,
)
from agent_core.types.response import AgentChunk

#: deepagents create_deep_agent 的 extra 白名单透传键
DEEPAGENTS_EXTRA_KEYS: tuple[str, ...] = (
    "middleware",
    "memory",
    "permissions",
    "backend",
    "interrupt_on",
    "response_format",
    "state_schema",
    "context_schema",
    "checkpointer",
    "store",
    "debug",
    "name",
    "cache",
)

#: LangChain 消息 type → agent_core 角色
_ROLE_BY_LC_TYPE: dict[str, MessageRole] = {
    "human": MessageRole.USER,
    "system": MessageRole.SYSTEM,
    "ai": MessageRole.ASSISTANT,
    "tool": MessageRole.TOOL,
}

_INSTALL_HINT = "pip install 'agent-core[deepagents]'"


def _import_module(module_name: str) -> Any:
    """延迟导入外部模块；失败时抛 BackendDependencyError。

    返回 ``Any`` 是因为 langchain / deepagents 未必带类型标注，
    显式标注为 Any 可让 mypy 把调用结果当 Any 处理（不做静态检查）。
    """
    try:
        module: Any = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise BackendDependencyError(backend="deepagents", install_hint=_INSTALL_HINT) from exc
    return module


def import_create_deep_agent() -> Any:
    """延迟导入 deepagents 的 ``create_deep_agent``。

    这是 deepagents 唯一在「真正使用时」才被导入的入口：
    ``import agent_core`` 本身不会加载 deepagents。
    """
    return _import_module("deepagents").create_deep_agent


# ---------------------------------------------------------------- model 解析

def resolve_chat_model(config: AgentConfig | None) -> Any | None:
    """解析 deepagents 使用的 model 参数（三级优先级，依次尝试）。

    优先级：
    1. ``config.extra["model"]`` —— 调用方已构建好的 LangChain ChatModel，
       直接透传（最高优先，适合复用连接 / 自定义配置的模型实例）；
    2. 仅 ``AgentModel.name``（无 api_key / base_url）→ 传模型名字符串
       （如 ``openai:gpt-4o-mini``），由 deepagents 自行解析厂商前缀；
    3. name + api_key / base_url → 调用 ``langchain.chat_models.init_chat_model``
       按名称 + 凭据构建 ChatModel。

    返回 ``None`` 表示没有可用的模型配置（交给 deepagents 默认值）。
    """
    if config is None:
        return None
    extra_model = config.extra.get("model")
    if extra_model is not None:
        return extra_model
    if config.model is None or config.model.name is None:
        return None
    if not config.model.api_key and not config.model.base_url:
        # 既没有 key 也没有 url：deepagents 支持直接传模型名字符串
        return config.model.name
    # 有凭据：用 init_chat_model 按厂商前缀构建（如 "deepseek:deepseek-v4-flash"）
    lc_chat_models = _import_module("langchain.chat_models")
    init_chat_model = lc_chat_models.init_chat_model
    init_kwargs: dict[str, Any] = {"model": config.model.name}
    if config.model.api_key:
        init_kwargs["api_key"] = config.model.api_key
    if config.model.base_url:
        init_kwargs["base_url"] = config.model.base_url
    return init_chat_model(**init_kwargs)


# ---------------------------------------------------------------- kwargs 构建

def _map_tools(config: AgentConfig) -> list[Any]:
    """tools 映射：优先 AgentTool.handler，其次经 ``extra["tools"]`` 解析。

    ``AgentTool.handler`` 是推荐方式（工具自带可调用对象）；
    也兼容 ``extra["tools"]`` 传入 ``{工具名: 可调用对象}`` 字典
    或直接传工具列表（list 会整体追加）。
    """
    extra_tools: Any = config.extra.get("tools")
    tools: list[Any] = []
    for tool in config.tools:
        handler = tool.handler
        # handler 缺失时，尝试从 extra["tools"] 字典按 name 查
        if handler is None and isinstance(extra_tools, dict):
            handler = extra_tools.get(tool.name)
        if handler is not None:
            tools.append(handler)
    if isinstance(extra_tools, list):
        tools.extend(extra_tools)
    return tools


def _map_subagent(subagent: AgentSubagent) -> dict[str, Any]:
    """AgentSubagent → deepagents subagent 字典。

    只映射 deepagents 认识的键（name / description / system_prompt /
    model / tools / skills），并把 ``extra`` 里额外的键透传进去，
    让用户能按需补充 deepagents 特有配置。
    """
    mapped: dict[str, Any] = {
        "name": subagent.name,
        "description": subagent.description,
    }
    if subagent.system_prompt:
        mapped["system_prompt"] = subagent.system_prompt
    if subagent.model and subagent.model.name:
        mapped["model"] = subagent.model.name
    handlers = [tool.handler for tool in subagent.tools if tool.handler is not None]
    if handlers:
        mapped["tools"] = handlers
    if subagent.skills and subagent.skills.sources:
        mapped["skills"] = list(subagent.skills.sources)
    mapped.update(subagent.extra)
    return mapped


def build_create_agent_kwargs(config: AgentConfig) -> dict[str, Any]:
    """将 AgentConfig 翻译为 ``create_deep_agent(**kwargs)`` 的关键字参数。

    映射规则：
    - model / system_prompt / tools / skills / subagents 走各自的转换函数；
    - ``config.extra`` 中命中 ``DEEPAGENTS_EXTRA_KEYS`` 白名单的键直接透传
      （middleware / memory / permissions / backend / interrupt_on ...），
      不在白名单里的键（如 tools / model）已被专门处理，不会重复透传。
    """
    kwargs: dict[str, Any] = {}

    model = resolve_chat_model(config)
    if model is not None:
        kwargs["model"] = model

    if config.system_prompt:
        kwargs["system_prompt"] = config.system_prompt

    tools = _map_tools(config)
    if tools:
        kwargs["tools"] = tools

    if config.skills and config.skills.sources:
        kwargs["skills"] = list(config.skills.sources)

    if config.subagents:
        kwargs["subagents"] = [_map_subagent(subagent) for subagent in config.subagents]

    # 白名单透传：只放行 deepagents 认识的高级能力键
    for key in DEEPAGENTS_EXTRA_KEYS:
        if key in config.extra:
            kwargs[key] = config.extra[key]

    return kwargs


# ---------------------------------------------------------------- 输入方向

def _arguments_to_langchain(arguments: dict[str, Any] | str) -> dict[str, Any]:
    """ToolCall.arguments → LangChain args（必须是 dict）。

    arguments 若是 JSON 字符串则解析；解析失败或不是 dict 时回退为空 dict，
    避免把非法参数传给 SDK。
    """
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments) if arguments else {}
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def agent_messages_to_langchain(messages: list[AgentMessage]) -> list[Any]:
    """将 agent_core 消息翻译为 LangChain 消息列表（输入方向）。

    角色映射：user → HumanMessage、system → SystemMessage、
    assistant → AIMessage（带 tool_calls）、tool → ToolMessage。
    注意：``thinking`` 与 ``meta`` 不会发送给后端 —— 它们是回程时才提取的信息。
    """
    lc_messages = _import_module("langchain_core.messages")
    result: list[Any] = []
    for message in messages:
        if message.role is MessageRole.USER:
            result.append(lc_messages.HumanMessage(content=message.content))
        elif message.role is MessageRole.SYSTEM:
            result.append(lc_messages.SystemMessage(content=message.content))
        elif message.role is MessageRole.ASSISTANT:
            # LangChain 的 tool_calls 结构：{"id", "name", "args"}
            tool_calls = [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "args": _arguments_to_langchain(tool_call.arguments),
                }
                for tool_call in message.tool_calls
            ]
            result.append(
                lc_messages.AIMessage(
                    content=message.content,
                    tool_calls=tool_calls or None,
                )
            )
        elif message.role is MessageRole.TOOL:
            tool_call_id = message.tool_result.tool_call_id if message.tool_result else ""
            result.append(
                lc_messages.ToolMessage(content=message.content, tool_call_id=tool_call_id)
            )
    return result


# ---------------------------------------------------------------- 输出方向

def _extract_text_content(message: Any) -> str:
    """提取文本内容（输出方向）。

    LangChain 消息的 content 可能是：
    - 纯字符串 → 直接返回；
    - 内容块列表（如 ``[{"type": "text", "text": "..."}]``）→ 拼接所有
      ``type=text`` 块，忽略图片 / 工具调用等其它块。
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_thinking(message: Any) -> str | None:
    """提取思考 / 推理内容（输出方向），按优先级尝试三个来源：

    1. ``additional_kwargs["reasoning_content"]`` —— DeepSeek 等模型的推理字段；
    2. ``additional_kwargs["thinking"]`` —— 部分模型 / 中间件写入的思考字段；
    3. ``content_blocks`` 中 ``type`` 为 ``reasoning`` / ``thinking`` 的块
       （较新的消息格式把推理作为独立内容块）。

    返回 None 表示该消息没有思考内容。
    """
    additional = getattr(message, "additional_kwargs", None)
    if isinstance(additional, dict):
        for key in ("reasoning_content", "thinking"):
            value = additional.get(key)
            if isinstance(value, str) and value:
                return value
    content_blocks = getattr(message, "content_blocks", None)
    parts: list[str] = []
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") in ("reasoning", "thinking"):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
    return "".join(parts) if parts else None


def _normalize_arguments(args: Any) -> dict[str, Any] | str:
    """LangChain tool_calls 的 args → ToolCall.arguments。

    args 通常已是 dict；保持 str 原样（部分 SDK 输出 JSON 字符串）；
    其它形态（None / 非法值）归一化为空 dict。
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        return args
    return {}


def langchain_message_to_agent_message(message: Any) -> AgentMessage:
    """LangChain 消息 → agent_core 消息（输出方向）。

    - ``type`` 映射角色：human→user、system→system、ai→assistant、tool→tool，
      未知类型兜底为 user；
    - content / thinking 分别走 ``_extract_text_content`` / ``_extract_thinking``；
    - tool 消息额外生成 ``tool_result``（tool_call_id + content）；
    - 原始对象存入 ``_raw``（PrivateAttr，不参与序列化），并记录
      ``meta.langchain_type`` 供调试 / 回程识别。
    """
    msg_type = getattr(message, "type", "") or ""
    role = _ROLE_BY_LC_TYPE.get(msg_type, MessageRole.USER)
    content = _extract_text_content(message)
    thinking = _extract_thinking(message)
    tool_calls: list[ToolCall] = []
    raw_tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in raw_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_calls.append(
            ToolCall(
                id=tool_call.get("id", "") or "",
                name=tool_call.get("name", "") or "",
                arguments=_normalize_arguments(tool_call.get("args", {})),
            )
        )
    tool_result: ToolResult | None = None
    if role is MessageRole.TOOL:
        tool_result = ToolResult(
            tool_call_id=getattr(message, "tool_call_id", "") or "",
            content=content,
        )
    return AgentMessage(
        role=role,
        content=content,
        thinking=thinking,
        tool_calls=tool_calls,
        tool_result=tool_result,
        meta={"langchain_type": msg_type},
        _raw=message,
    )


# ---------------------------------------------------------------- 流式

def unpack_stream_event(event: Any) -> Any:
    """解包流式事件，取出真正的 chunk。

    ``graph.astream(..., stream_mode="messages")`` 的产出有两种形态：
    - ``(namespace, chunk)`` 元组（namespace 是 tuple/list，如 ``(("messages", 0), AIMessageChunk)``）；
    - 裸 chunk（如 AIMessageChunk 本身）。
    这里只解包第一种，其余原样返回。
    """
    if isinstance(event, tuple) and len(event) == 2:
        first, second = event
        if isinstance(first, (tuple, list, dict)):
            return second
    return event


def _join_tool_chunk_field(tool_call_chunks: Any, field: str) -> str:
    """聚合 tool_call_chunks 上某个字段的分片字符串。

    流式工具调用（tool_call_chunks）会把 id / name / args 拆成多个分片，
    这里把它们按顺序拼接成完整字符串（兼容 dict 与对象两种形态）。
    """
    parts: list[str] = []
    for tool_call_chunk in tool_call_chunks:
        if isinstance(tool_call_chunk, dict):
            value = tool_call_chunk.get(field)
        else:
            value = getattr(tool_call_chunk, field, None)
        if value:
            parts.append(str(value))
    return "".join(parts)


def map_message_chunk_to_agent_chunks(chunk: Any) -> list[AgentChunk]:
    """一条 LangChain chunk 可能映射为多个 AgentChunk。

    流式输出中模型可能同时产生多个维度（思考、正文、工具调用分片），
    因此按以下顺序逐个产出 AgentChunk：
    1. ``delta_thinking``（reasoning_content / thinking 等来源）；
    2. ``delta_content``（文本增量）；
    3. ``delta_tool_call``（工具调用分片聚合，id / name / args）。
    没有内容时返回空列表（调用方直接忽略）。
    """
    message = unpack_stream_event(chunk)
    msg_type = getattr(message, "type", "") or ""
    meta: dict[str, Any] = {"langchain_type": msg_type} if msg_type else {}
    agent_chunks: list[AgentChunk] = []

    thinking = _extract_thinking(message)
    if thinking:
        agent_chunks.append(AgentChunk(delta_thinking=thinking, meta=meta))

    content = _extract_text_content(message)
    if content:
        agent_chunks.append(AgentChunk(delta_content=content, meta=meta))

    tool_call_chunks = getattr(message, "tool_call_chunks", None)
    if tool_call_chunks:
        tool_id = _join_tool_chunk_field(tool_call_chunks, "id")
        tool_name = _join_tool_chunk_field(tool_call_chunks, "name")
        tool_args = _join_tool_chunk_field(tool_call_chunks, "args")
        if tool_id or tool_name or tool_args:
            agent_chunks.append(
                AgentChunk(
                    delta_tool_call=ToolCall(id=tool_id, name=tool_name, arguments=tool_args),
                    meta=meta,
                )
            )

    return agent_chunks
