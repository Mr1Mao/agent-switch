"""构建 hooks Context 与处理 MODIFY 结果的工具函数。

注意：``build_before_agent_context`` / ``build_before_llm_context`` 使用
``summarize_config`` 的 ``model_name`` key（与 summarize 返回的 ``model``
字段不一致——保持此实现）。
"""

from __future__ import annotations

from pydantic import ValidationError

from agent_switch.hooks.context import (
    AfterAgentHookContext,
    AfterLLMHookContext,
    AfterStopHookContext,
    BeforeAgentHookContext,
    BeforeLLMHookContext,
    BeforePromptHookContext,
)
from agent_switch.hooks.result import HookOutcome, HookResult
from agent_switch.logging import summarize_config
from agent_switch.types import AgentConfig, AgentMessage, AgentResponse


def build_before_agent_context(
    backend: str,
    messages: list[AgentMessage],
    config: AgentConfig | None = None,
    *,
    session_id: str | None = None,
    correlation_id: str | None = None,
) -> BeforeAgentHookContext:
    """构建 beforeAgent Context（model 取 summarize_config 的 model_name key）。"""
    model_name = summarize_config(config).get("model_name") if config is not None else None
    return BeforeAgentHookContext(
        backend=backend,
        session_id=session_id,
        correlation_id=correlation_id,
        model=model_name,
        input_messages=messages,
        config=config,
    )


def build_before_prompt_context(
    backend: str,
    messages: list[AgentMessage],
    config: AgentConfig | None = None,
    *,
    session_id: str | None = None,
    correlation_id: str | None = None,
) -> BeforePromptHookContext:
    """构建 beforePrompt Context。"""
    system_prompt = config.system_prompt if config is not None else None
    return BeforePromptHookContext(
        backend=backend,
        session_id=session_id,
        correlation_id=correlation_id,
        messages=messages,
        system_prompt=system_prompt,
    )


def build_before_llm_context(
    backend: str,
    messages: list[AgentMessage],
    config: AgentConfig | None = None,
    *,
    session_id: str | None = None,
    correlation_id: str | None = None,
) -> BeforeLLMHookContext:
    """构建 beforeLLM Context（model 取 summarize_config 的 model_name key）。"""
    model_name = summarize_config(config).get("model_name") if config is not None else None
    return BeforeLLMHookContext(
        backend=backend,
        session_id=session_id,
        correlation_id=correlation_id,
        model=model_name,
        messages=messages,
    )


def build_after_llm_context(
    backend: str,
    response: AgentResponse | None = None,
    *,
    config: AgentConfig | None = None,
    session_id: str | None = None,
    correlation_id: str | None = None,
    content: str = "",
    error: BaseException | None = None,
) -> AfterLLMHookContext:
    """构建 afterLLM Context。"""
    return AfterLLMHookContext(
        backend=backend,
        session_id=session_id,
        correlation_id=correlation_id,
        response=response,
        content=content,
        error=error,
    )


def build_after_agent_context(
    backend: str,
    response: AgentResponse | None = None,
    *,
    session_id: str | None = None,
    correlation_id: str | None = None,
    error: BaseException | None = None,
) -> AfterAgentHookContext:
    """构建 afterAgent Context。"""
    return AfterAgentHookContext(
        backend=backend,
        session_id=session_id,
        correlation_id=correlation_id,
        response=response,
        error=error,
    )


def build_after_stop_context(
    backend: str,
    reason: str = "complete",
    *,
    response: AgentResponse | None = None,
    session_id: str | None = None,
    correlation_id: str | None = None,
    error: BaseException | None = None,
) -> AfterStopHookContext:
    """构建 afterStop Context。"""
    return AfterStopHookContext(
        backend=backend,
        session_id=session_id,
        correlation_id=correlation_id,
        reason=reason,
        response=response,
        error=error,
    )


def apply_messages_modify(messages: list[AgentMessage], hook_result: HookResult | None) -> list[AgentMessage]:
    """当 MODIFY 且 data 含 ``messages`` 列表时，解析为 AgentMessage 列表替换。"""
    if hook_result is None or hook_result.outcome is not HookOutcome.MODIFY:
        return messages
    data = hook_result.data or {}
    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list):
        return messages
    parsed: list[AgentMessage] = []
    for item in raw_messages:
        if isinstance(item, AgentMessage):
            parsed.append(item)
        elif isinstance(item, dict):
            try:
                parsed.append(AgentMessage.model_validate(item))
            except ValidationError:
                return messages
        else:
            return messages
    return parsed
