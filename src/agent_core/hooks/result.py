"""Hook 执行结果模型。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_core.hooks.enums import AgentHookEvent


class HookOutcome(str, Enum):
    """Hook 执行结果类型。"""

    # 放行：什么都不做，继续正常流程
    CONTINUE = "continue"
    # 拦截：终止本次调用（dispatcher 短路返回，adapter 抛出 HookBlockedError）
    BLOCK = "block"
    # 拦截并修改数据（如 data={"messages": [...]} 改写 prompt 消息），改完继续
    MODIFY = "modify"


class HookResult(BaseModel):
    """Hook 返回值：outcome + reason + data。"""

    model_config = ConfigDict(extra="forbid")

    outcome: HookOutcome = HookOutcome.CONTINUE
    reason: str = ""
    data: dict[str, Any] = {}


#: 允许拦截（BLOCK / MODIFY）的事件集合
INTERCEPTABLE_HOOK_EVENTS: frozenset[AgentHookEvent] = frozenset(
    {
        AgentHookEvent.BEFORE_PROMPT,
        AgentHookEvent.BEFORE_TOOL,
        AgentHookEvent.BEFORE_PERMISSION,
    }
)
