"""Agent 适配器抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from agent_switch.types import AgentChunk, AgentConfig, AgentMessage, AgentResponse


class AgentAdapter(ABC):
    """Agent SDK 统一抽象：业务代码只依赖本接口，不直接依赖底层框架。"""

    backend_name: str = ""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self._default_config = config

    @abstractmethod
    def run(self, input: str | list[AgentMessage], config: AgentConfig | None = None) -> AgentResponse:
        """同步运行一次 Agent 会话，返回完整响应。"""
        raise NotImplementedError

    @abstractmethod
    def stream(self, input: str | list[AgentMessage], config: AgentConfig | None = None) -> AsyncIterator[AgentChunk]:
        """流式运行，逐块返回 AgentChunk。"""
        raise NotImplementedError
