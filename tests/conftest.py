"""共享测试夹具：mock QcoderAdapter 的 SDK 调用点（不依赖真实 qodercli）。"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
from qoder_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

from agent_core.backends.qcoder.adapter import QcoderAdapter


async def _drain_wires(prompt: Any) -> list[dict[str, Any]]:
    """消费 qoder wire 异步迭代器，取回全部 wire 消息。"""
    if prompt is None:
        return []
    return [wire async for wire in prompt]


async def _fake_query(
    *,
    prompt: Any = None,
    options: Any = None,
    transport: Any = None,
) -> AsyncIterator[Any]:
    """模拟 ``qoder_agent_sdk.query``：回显最后一条用户文本，随后输出终结消息。

    行为：
    - 从 prompt（wire 格式）中提取最后一段 text 内容；
    - 先 yield 一条 AssistantMessage（回显文本），再 yield ResultMessage 终结；
    - 这样 adapter 的 run/stream 收尾逻辑（收集文本、终结判定、hooks 顺序）
      都能被真实路径覆盖。
    """
    last_content = ""
    for wire in await _drain_wires(prompt):
        content = (wire.get("message") or {}).get("content")
        if isinstance(content, str):
            last_content = content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    last_content = block.get("text", "")
    echo = f"[stub] {last_content}" if last_content else "[stub]"
    yield AssistantMessage(content=[TextBlock(text=echo)], model="mock-model")
    yield ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="mock-session",
        result=echo,
        stop_reason="end_turn",
    )


@pytest.fixture
def qcoder_fake_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 QcoderAdapter._query_impl 替换为回显式 fake query。"""
    monkeypatch.setattr(QcoderAdapter, "_query_impl", lambda self: _fake_query)
