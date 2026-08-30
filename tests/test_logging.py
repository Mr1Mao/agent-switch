"""logging 工具测试。"""

import json
import logging
import re

from agent_switch import (
    AgentConfig,
    AgentMcpConfig,
    AgentMcpServer,
    AgentMessage,
    AgentModel,
    AgentSkillsConfig,
    AgentSubagent,
    AgentTool,
    MessageRole,
)
from agent_switch.logging import (
    AgentCoreDevFormatter,
    AgentCoreJsonFormatter,
    configure_logging,
    get_logger,
    log_fields,
    summarize_config,
    summarize_input,
)


def test_get_logger_returns_namespaced_logger():
    """get_logger 返回 agent_switch 命名空间下的 Logger。"""
    logger = get_logger("agent_switch.factory")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "agent_switch.factory"


def test_log_fields_redacts_sensitive_keys():
    """敏感 key（api_key / token 等）递归脱敏为 <redacted>。"""
    fields = log_fields(
        backend="deepagents",
        api_key="sk-123",
        nested={"token": "abc", "ok": 1},
    )
    payload = fields["agent_switch"]
    assert payload["backend"] == "deepagents"
    assert payload["api_key"] == "<redacted>"
    assert payload["nested"]["token"] == "<redacted>"
    assert payload["nested"]["ok"] == 1


def test_summarize_config_returns_expected_keys():
    """summarize_config 返回规格要求的全部摘要字段。"""
    config = AgentConfig(
        model=AgentModel(name="deepseek-v4-flash", api_key="sk-1", base_url="https://x"),
        system_prompt="be concise",
        tools=[AgentTool(name="search")],
        skills=AgentSkillsConfig(sources=["skill_a"]),
        mcp=AgentMcpConfig(servers=[AgentMcpServer(name="server_a")]),
        subagents=[AgentSubagent(name="sub_a")],
        extra={"model": object()},
    )
    summary = summarize_config(config)
    assert summary["model"] == "deepseek-v4-flash"
    assert summary["has_api_key"] is True
    assert summary["has_base_url"] is True
    assert summary["tools_count"] == 1
    assert summary["skills_count"] == 1
    assert summary["mcp_server_count"] == 1
    assert summary["subagents_count"] == 1
    assert summary["has_system_prompt"] is True
    assert summary["extra_model_override"] is True


def test_summarize_config_uses_model_name_key():
    """summarize_config 同时提供 model 与 model_name（emitter 依赖 model_name）。"""
    config = AgentConfig(model=AgentModel(name="deepseek-v4-flash"))
    summary = summarize_config(config)
    assert summary["model_name"] == "deepseek-v4-flash"
    assert summary["model"] == "deepseek-v4-flash"


def test_summarize_input_preview_truncated_to_120():
    """summarize_input：取最后一条消息，preview 截断到 120 字符。"""
    long_text = "x" * 300
    messages = [
        AgentMessage(role=MessageRole.USER, content="first"),
        AgentMessage(role=MessageRole.USER, content=long_text),
    ]
    summary = summarize_input(messages)
    assert summary["message_count"] == 2
    assert summary["input_preview_len"] == 300
    assert len(summary["input_preview"]) == 120
    assert summary["input_preview"] == long_text[:120]


def test_configure_logging_dev_and_json_formatters():
    """configure_logging 配置 agent_switch logger：级别、propagate、Dev/JSON Formatter。"""
    logger = configure_logging(level=logging.DEBUG, json=False)
    assert logger.name == "agent_switch"
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert isinstance(logger.handlers[-1].formatter, AgentCoreDevFormatter)

    record = logging.LogRecord(
        "agent_switch.factory", logging.INFO, "factory.py", 1, "agent.create.start", None, None
    )
    setattr(record, "agent_switch", {"backend": "deepagents"})
    line = logger.handlers[-1].formatter.format(record)
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z "
        r"INFO \[agent_switch\.factory\] agent\.create\.start backend=deepagents$",
        line,
    )

    configure_logging(level=logging.INFO, json=True)
    assert logger.level == logging.INFO
    assert isinstance(logger.handlers[-1].formatter, AgentCoreJsonFormatter)
    json_line = logger.handlers[-1].formatter.format(record)
    parsed = json.loads(json_line)
    assert parsed["severity"] == "INFO"
    assert parsed["logger"] == "agent_switch.factory"
    assert parsed["message"] == "agent.create.start"
    assert parsed["backend"] == "deepagents"
