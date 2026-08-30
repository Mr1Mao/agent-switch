"""AgentSkillsConfig：技能（skills）配置模型。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AgentSkillsConfig(BaseModel):
    """技能配置：技能来源列表 + 是否启用全部技能。"""

    model_config = ConfigDict(extra="forbid")

    sources: list[str] = []
    enable_all: bool = False
