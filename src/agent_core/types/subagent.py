"""AgentSubagent：子代理配置模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_core.types.mcp import AgentMcpConfig
from agent_core.types.model import AgentModel
from agent_core.types.skill import AgentSkillsConfig
from agent_core.types.tool import AgentTool


class AgentSubagent(BaseModel):
    """子代理描述：名称、说明、系统提示词、工具、模型、技能、MCP 与透传 extra。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    system_prompt: str | None = None
    tools: list[AgentTool] = []
    model: AgentModel | None = None
    skills: AgentSkillsConfig | None = None
    mcp: AgentMcpConfig | None = None
    extra: dict[str, Any] = {}
