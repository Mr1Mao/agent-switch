"""AgentTool：统一的工具描述模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class AgentTool(BaseModel):
    """工具描述：名称、描述、参数 schema 与可选的 handler 可调用对象。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    parameters: dict[str, Any] = {}
    handler: Any = None
