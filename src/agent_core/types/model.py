"""AgentModel：统一的模型配置模型。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AgentModel(BaseModel):
    """模型引用：名称 + 可选的 api_key / base_url。

    - 仅提供 name 时，deepagents 后端会直接透传模型名字符串（如 ``openai:gpt-4o-mini``）。
    - 同时提供 api_key / base_url 时，由后端负责构建对应的 ChatModel。
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    api_key: str | None = None
    base_url: str | None = None
