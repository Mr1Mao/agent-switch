"""AgentMessage：统一消息模型（含 tool_call / tool_result 与私有 _raw 调试字段）。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from agent_switch.types.enums import MessageRole


class ToolCall(BaseModel):
    """工具调用：id / name / arguments。

    ``arguments`` 与 LangChain 的 ``args`` 字段对应：既可以是 dict，
    也可以是 JSON 字符串（部分 SDK 输出形态如此）。
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] | str = {}


class ToolResult(BaseModel):
    """工具执行结果：tool_call_id / content。"""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    content: str


class AgentMessage(BaseModel):
    """统一消息模型 —— 各后端消息的中立表示。

    - ``role``：user / assistant / system / tool；
    - ``content``：文本内容；``thinking``：推理内容（模型相关，可为 None）；
    - ``tool_calls`` / ``tool_result``：工具调用与结果（tool 角色消息）；
    - ``meta``：后端元信息（如 ``langchain_type``），透传不参与转换；
    - ``_raw``：PrivateAttr，仅用于 Adapter 调试（保留后端原始对象），
      不会出现在 ``model_dump()`` / ``to_dict()`` 的结果中。
    """

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = ""
    thinking: str | None = None
    tool_calls: list[ToolCall] = []
    tool_result: ToolResult | None = None
    meta: dict[str, Any] = {}
    _raw: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        """自定义 __init__：支持通过 ``_raw`` 关键字注入调试信息。

        背景：较新的 pydantic（>=2.12）不再允许通过构造参数设置 PrivateAttr
        （官方示例也会静默丢弃），因此这里在进入标准校验前把 ``_raw`` 从
        入参中取出，校验完成后再赋值给私有属性。
        这样 ``AgentMessage(..., _raw=langchain_message)`` 与
        ``message._raw = x`` 两种写法都能工作，且 ``_raw`` 始终不参与序列化。
        """
        raw = data.pop("_raw", None)
        super().__init__(**data)
        if raw is not None:
            self._raw = raw

    def to_dict(self) -> dict[str, Any]:
        """转 JSON 可序列化 dict。

        规则：``mode="json"``（枚举→字符串、datetime→ISO 等），
        并排除 None 与默认值，让序列化结果尽量精简（``_raw`` 天然不在其中）。
        """
        return self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
