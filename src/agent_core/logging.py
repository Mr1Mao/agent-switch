"""agent_core 日志工具。

原则：
- 库不打日志 handler：``configure_logging()`` 才配置日志，``import`` 时不自动配置；
- 仅配置 ``agent_core`` 命名空间 logger，``propagate=False``；
- 敏感字段（api_key / token / password / secret / authorization / credential）自动脱敏。
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from agent_core.types import AgentConfig, AgentMessage
from agent_core.utils.input import format_input_preview

AGENT_CORE_LOGGER_NAME = "agent_core"

#: 敏感 key 子串，命中即脱敏
SENSITIVE_KEYS: tuple[str, ...] = (
    "api_key",
    "token",
    "password",
    "secret",
    "authorization",
    "credential",
)


def get_logger(name: str) -> logging.Logger:
    """返回 ``agent_core`` 命名空间下的 logger。"""
    return logging.getLogger(name)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(sensitive in lowered for sensitive in SENSITIVE_KEYS)


def sanitize_dict(data: dict[str, Any]) -> dict[str, Any]:
    """递归脱敏敏感 key，返回新 dict。"""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if _is_sensitive_key(key):
            result[key] = "<redacted>"
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value)
        else:
            result[key] = value
    return result


def log_fields(**kwargs: Any) -> dict[str, Any]:
    """构造 logging ``extra=`` 使用的字段：``{"agent_core": sanitized}``。

    Python logging 的 ``extra`` 只能带一层 key，这里把全部字段包进
    ``agent_core`` 命名空间，Formatter 再从 ``record.agent_core`` 读取，
    避免污染标准字段（如 message / levelname 等）。
    """
    return {"agent_core": sanitize_dict(kwargs)}


def summarize_config(config: AgentConfig | None) -> dict[str, Any]:
    """汇总 AgentConfig 的关键信息用于日志（只记录概要，不记录密钥本身）。

    注意：同时提供 ``model`` 与 ``model_name`` 两个 key ——
    ``hooks/emitter.py`` 依赖 ``model_name``（与 ``model`` 不一致是有意保留的实现）。
    """
    if config is None:
        return {
            "model": None,
            "model_name": None,
            "has_api_key": False,
            "has_base_url": False,
            "tools_count": 0,
            "skills_count": 0,
            "mcp_server_count": 0,
            "subagents_count": 0,
            "has_system_prompt": False,
            "extra_model_override": False,
        }
    model = config.model
    return {
        # model 与 model_name 同值：前者供日志展示，后者供 hooks/emitter 读取
        "model": model.name if model else None,
        "model_name": model.name if model else None,
        # 只记录「是否有」密钥，绝不记录密钥值本身
        "has_api_key": bool(model and model.api_key),
        "has_base_url": bool(model and model.base_url),
        "tools_count": len(config.tools),
        "skills_count": len(config.skills.sources) if config.skills else 0,
        "mcp_server_count": len(config.mcp.servers) if config.mcp else 0,
        "subagents_count": len(config.subagents),
        "has_system_prompt": bool(config.system_prompt),
        # extra 里是否覆盖了 model（日志里不输出模型对象本身）
        "extra_model_override": config.extra.get("model") is not None,
    }


def summarize_input(messages: list[AgentMessage]) -> dict[str, Any]:
    """汇总输入消息信息；``input_preview`` 截断到 120 字符。

    只记录最后一条消息的 content 前缀，避免把整段对话写进日志。
    """
    preview = format_input_preview(messages)
    return {
        "message_count": len(messages),
        "input_preview_len": len(preview),
        "input_preview": preview[:120],
    }


def _format_timestamp(timestamp: datetime) -> str:
    """格式化 UTC 时间戳：``2026-08-25T10:04:00.123Z``（毫秒精度 + Z 后缀）。"""
    base = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.")
    return f"{base}{timestamp.microsecond // 1000:03d}Z"


def _extra_payload(record: logging.LogRecord) -> dict[str, Any]:
    """读取 ``record.agent_core``（由 ``log_fields`` 写入），不存在时返回空 dict。"""
    payload = getattr(record, "agent_core", None)
    if isinstance(payload, dict):
        return payload
    return {}


class AgentCoreDevFormatter(logging.Formatter):
    """开发格式：``2026-08-25T10:04:00.123Z INFO [logger] message key=value``。

    agent_core 字段按 key 排序输出，保证同一事件的行格式稳定、易于 grep。
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        extra_fields = _extra_payload(record)
        suffix_parts: list[str] = []
        for key in sorted(extra_fields):
            value = extra_fields[key]
            # 字符串原样输出（如 backend=deepagents），其它类型走 repr
            rendered = value if isinstance(value, str) else repr(value)
            suffix_parts.append(f"{key}={rendered}")
        timestamp = _format_timestamp(datetime.fromtimestamp(record.created, tz=timezone.utc))
        line = f"{timestamp} {record.levelname} [{record.name}] {message}"
        if suffix_parts:
            line += " " + " ".join(suffix_parts)
        return line


class AgentCoreJsonFormatter(logging.Formatter):
    """JSON 格式：单行 ``{timestamp, severity, logger, message, ...agent_core 字段}``。

    便于日志采集系统（如 ELK / Loki）直接解析；非 JSON 序列化的值
    通过 ``default=str`` 兜底转换。
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _format_timestamp(
                datetime.fromtimestamp(record.created, tz=timezone.utc)
            ),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # agent_core 字段平铺进顶层，与 timestamp/severity 并列
        payload.update(_extra_payload(record))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO, json: bool = False, stream: Any = sys.stderr) -> logging.Logger:
    """配置 ``agent_core`` logger；重复调用会先清空已有 handler。

    :param level: 日志级别。
    :param json: True 使用 JSON Formatter，False 使用 Dev Formatter。
    :param stream: 输出流，默认 stderr。
    """
    logger = logging.getLogger(AGENT_CORE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.StreamHandler(stream)
    formatter: logging.Formatter
    if json:
        formatter = AgentCoreJsonFormatter()
    else:
        formatter = AgentCoreDevFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
