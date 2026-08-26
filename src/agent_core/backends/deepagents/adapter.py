"""DeepAgents 后端适配器（真实实现，延迟导入 deepagents）。"""

from __future__ import annotations

from typing import Any, AsyncIterator

from agent_core.adapter_base import BaseAgentAdapter
from agent_core.backends.deepagents.mapping import (
    agent_messages_to_langchain,
    build_create_agent_kwargs,
    import_create_deep_agent,
    langchain_message_to_agent_message,
    map_message_chunk_to_agent_chunks,
)
from agent_core.logging import get_logger
from agent_core.types import AgentChunk, AgentConfig, AgentMessage, AgentResponse

_logger = get_logger("agent_core.backends.deepagents")


class DeepAgentsAdapter(BaseAgentAdapter):
    """基于 langchain-deepseek / deepagents 的真实适配器。"""

    backend_name = "deepagents"

    #: beforeLLM / afterLLM 由注入的 AgentHooksMiddleware 按调用触发，adapter 层不重复触发
    call_hooks_via_middleware = True
    #: beforeAgent / beforePrompt / afterAgent 由中间件的 before_agent / after_agent
    #: 节点钩子触发（错误路径的 afterAgent 仍在 adapter 补发）
    agent_hooks_via_middleware = True

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config)
        self._graph: Any = None
        self._graph_cache_key: str | None = None

    def _build_agent(self, config: AgentConfig | None) -> Any:
        """构建（并缓存）``create_deep_agent`` 返回的图。

        缓存键为 kwargs 排序 repr + hooks 指纹；键相同则复用已构建的图。
        配置了 hooks 时，额外注入 ``AgentHooksMiddleware`` 桥接 agent 级与调用级事件
        （beforeAgent / beforePrompt / beforeLLM / afterLLM / beforeTool /
        afterTool / afterToolError / afterAgent）。

        时序细节：缓存键在注入中间件**之前**计算（中间件内含 dispatcher，
        repr 不稳定，不能进缓存键）；hooks 指纹单独参与缓存键，
        保证「更换 hooks 配置」时不会命中旧图。
        """
        kwargs = build_create_agent_kwargs(config) if config is not None else {}
        cache_key = _build_cache_key(kwargs, config)
        if self._graph is not None and self._graph_cache_key == cache_key:
            # 同一配置重复调用：直接复用已编译的图（编译开销大，值得缓存）
            return self._graph
        create_deep_agent = import_create_deep_agent()
        if config is not None and config.hooks:
            # 延迟导入：本模块只在 deepagents 后端真正使用时加载
            from agent_core.backends.deepagents.hooks_middleware import AgentHooksMiddleware

            dispatcher = self._resolve_hooks_dispatcher(config)
            assert dispatcher is not None  # config.hooks 非空时必有 dispatcher
            # 与用户通过 config.extra["middleware"] 传入的中间件共存
            middleware = list(kwargs.get("middleware") or [])
            middleware.append(
                AgentHooksMiddleware(
                    dispatcher,
                    backend=self.backend_name,
                    config=config,
                    # 闭包实时读取 adapter 当前 run 的会话 ID，
                    # 保证中间件内触发的事件与 adapter 层同一会话
                    session_provider=lambda: (self._session_id, self._correlation_id),
                )
            )
            kwargs["middleware"] = middleware
        self._graph = create_deep_agent(**kwargs)
        self._graph_cache_key = cache_key
        return self._graph

    def run(self, input: str | list[AgentMessage], config: AgentConfig | None = None) -> AgentResponse:
        """run 主流程：hooks → graph.invoke → build_agent_response。"""
        resolved = self._resolve_config(config)
        try:
            messages = self._prepare_messages_sync(input, resolved)
            messages = self._emit_before_llm_sync(messages, resolved)
            graph = self._build_agent(resolved)
            langchain_messages = agent_messages_to_langchain(messages)
            result = graph.invoke({"messages": langchain_messages})
            response = build_agent_response(result, backend=self.backend_name)
            self._finalize_run_success_sync(resolved, response)
            return response
        except Exception as exc:
            self._finalize_run_error_sync(resolved, exc)
            raise

    async def stream(self, input: str | list[AgentMessage], config: AgentConfig | None = None) -> AsyncIterator[AgentChunk]:
        """stream 主流程：graph.astream(stream_mode="messages") → map chunks → is_finish。"""
        resolved = self._resolve_config(config)
        messages = await self._prepare_messages_async(input, resolved)
        messages = await self._emit_before_llm_async(messages, resolved)
        graph = self._build_agent(resolved)
        langchain_messages = agent_messages_to_langchain(messages)
        final_content = ""
        async for raw_chunk in graph.astream(
            {"messages": langchain_messages}, stream_mode="messages"
        ):
            for agent_chunk in map_message_chunk_to_agent_chunks(raw_chunk):
                final_content += agent_chunk.delta_content
                yield agent_chunk
        yield AgentChunk(is_finish=True, meta={"final_content": final_content})
        response = self._build_stream_response(final_content)
        await self._finalize_stream_success_async(resolved, response)


def _build_cache_key(kwargs: dict[str, Any], config: AgentConfig | None) -> str:
    """kwargs 排序 repr + hooks 指纹（hooks 参与缓存，避免更换 hooks 后复用旧图）。"""
    base = repr(sorted(kwargs.items()))
    if config is None or not config.hooks:
        return base
    fingerprint = ",".join(str(id(hook)) for hook in config.hooks)
    return f"{base}|hooks={fingerprint}"


def build_agent_response(result: Any, backend: str) -> AgentResponse:
    """将 deepagents 图的 invoke 结果转换为 AgentResponse。"""
    raw_messages: Any = None
    if isinstance(result, dict):
        raw_messages = result.get("messages")
    elif result is not None:
        raw_messages = getattr(result, "messages", None)
    agent_message: AgentMessage | None = None
    content = ""
    if isinstance(raw_messages, list) and raw_messages:
        agent_message = langchain_message_to_agent_message(raw_messages[-1])
        content = agent_message.content or ""
    return AgentResponse(content=content, message=agent_message, raw=result, backend=backend)
