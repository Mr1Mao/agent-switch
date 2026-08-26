"""MCP 相关配置模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class AgentMcpServer(BaseModel):
    """单个 MCP 服务器描述。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    command: str | None = None
    url: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    enabled: bool = True
    extra: dict[str, Any] = {}


class AgentMcpConfig(BaseModel):
    """MCP 配置：服务器列表、允许的服务器名与严格模式。"""

    model_config = ConfigDict(extra="forbid")

    servers: list[AgentMcpServer] = []
    allowed_server_names: list[str] = []
    strict: bool = False
