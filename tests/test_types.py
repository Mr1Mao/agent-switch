"""types 类型系统测试。"""

import pytest
from pydantic import ValidationError

from agent_switch import AgentConfig, AgentMessage, MessageRole


def test_agent_config_default_hooks_is_empty_list():
    """默认 hooks 为 []（不是 None）。"""
    config = AgentConfig()
    assert config.hooks == []
    assert isinstance(config.hooks, list)


def test_agent_message_extra_fields_forbidden():
    """AgentMessage 禁止未知字段。"""
    with pytest.raises(ValidationError):
        AgentMessage(role=MessageRole.USER, content="hi", unexpected_field="x")


def test_raw_not_in_model_dump():
    """_raw 是 PrivateAttr，不出现在 model_dump / to_dict。"""
    message = AgentMessage(
        role=MessageRole.USER,
        content="hi",
        _raw={"langchain": "HumanMessage"},
    )
    dumped = message.model_dump()
    assert "_raw" not in dumped
    assert message._raw == {"langchain": "HumanMessage"}  # 调试信息仍可通过属性访问
    as_dict = message.to_dict()
    assert "_raw" not in as_dict
    assert as_dict["role"] == "user"
    assert as_dict["content"] == "hi"
