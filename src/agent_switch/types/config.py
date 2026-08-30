"""AgentConfig：统一 Agent 配置模型。

循环导入处理（重要）：
``hooks`` 字段的类型是 ``BaseAgentHooks``，而 ``agent_switch.hooks`` 包又依赖
本模块（Context 模型引用 AgentConfig）——直接相互 import 会成环。
解决方案：
1. 本模块用 ``from __future__ import annotations`` 让注解变成字符串，运行时不求值；
2. ``BaseAgentHooks`` 只在 ``TYPE_CHECKING`` 下导入（供 mypy 检查，不产生运行时依赖）；
3. 真正的运行时类型由 ``agent_switch/hooks/__init__.py`` 末尾注入：
   ``setattr(config_module, "BaseAgentHooks", BaseAgentHooks)`` 后调用
   ``AgentConfig.model_rebuild()``，pydantic 才能解析字符串注解并构建校验 schema。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, field_validator

from agent_switch.types.mcp import AgentMcpConfig
from agent_switch.types.model import AgentModel
from agent_switch.types.skill import AgentSkillsConfig
from agent_switch.types.subagent import AgentSubagent
from agent_switch.types.tool import AgentTool

if TYPE_CHECKING:
    from agent_switch.hooks.base import BaseAgentHooks


class AgentConfig(BaseModel):
    """统一 Agent 配置，业务代码通过它切换底层框架。

    - ``model`` / ``system_prompt`` / ``tools`` / ``skills`` / ``mcp`` / ``subagents``：
      各后端通用的配置项；
    - ``hooks``：hooks 实例列表（见 hooks 章节），默认 ``[]``；
    - ``extra``：透传字典，后端自定义能力（如 deepagents 的 ``extra["model"]``、
      ``extra["middleware"]``）都从这里取，避免为每个新能力扩字段。
    """

    # extra="forbid"：拼写错误的字段名会立刻报错，防止静默忽略；
    # arbitrary_types_allowed=True：允许 hooks 字段存放任意 BaseAgentHooks 实例
    # （它不是 pydantic 模型，而是普通类）。
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model: AgentModel | None = None
    system_prompt: str | None = None
    tools: list[AgentTool] = []
    skills: AgentSkillsConfig | None = None
    mcp: AgentMcpConfig | None = None
    subagents: list[AgentSubagent] = []
    hooks: list[BaseAgentHooks] = []
    extra: dict[str, Any] = {}

    @field_validator("hooks", mode="before")
    @classmethod
    def _normalize_hooks(cls, value: Any) -> Any:
        """把 ``hooks`` 入参归一化为列表（校验之前执行）。

        - ``None`` → ``[]``（显式传 None 等价于不传）；
        - 单个 ``BaseAgentHooks`` 实例 → 包装成单元素列表（``AgentConfig(hooks=AuditHooks())`` 可用）；
        - ``list`` → 原样透传；
        - 其他类型（str / dict / 数字等）→ 抛 TypeError。
        """
        if value is None:
            return []
        if isinstance(value, BaseAgentHooks):
            return [value]
        if isinstance(value, list):
            return value
        raise TypeError(
            "AgentConfig.hooks 必须是 BaseAgentHooks 实例、BaseAgentHooks 列表或 None"
        )

    @field_validator("hooks", mode="after")
    @classmethod
    def _validate_hooks_items(cls, value: list[BaseAgentHooks]) -> list[BaseAgentHooks]:
        """校验（之后执行）：列表里每一项都必须是 BaseAgentHooks 实例。

        正常情况下 pydantic 的 arbitrary-type 校验已保证 isinstance，
        这里兜底再查一遍，错误信息更友好。
        """
        for item in value:
            if not isinstance(item, BaseAgentHooks):
                raise TypeError(
                    f"AgentConfig.hooks 包含非 BaseAgentHooks 项: {type(item).__name__}"
                )
        return value
