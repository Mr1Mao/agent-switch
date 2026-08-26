"""可复用的真实 hooks 实现（生产风格，与后端无关）。

这四个实现覆盖了 hooks 的三种返回语义：
- 返回 ``None``：放行（观察类 hooks，如审计日志）；
- 返回 ``BLOCK``：拦截并终止调用（限流、敏感词）；
- 返回 ``MODIFY``：改写输入后继续（注入上下文）。

在你自己的 entry 入口中真实调用示例：

    import logging
    from agent_core import AgentBackend, AgentConfig, create_agent
    from examples.hooks import AuditLogHooks, RateLimitHooks, SensitiveWordHooks

    logging.basicConfig(level=logging.INFO)  # 让 AuditLogHooks 的日志可见

    config = AgentConfig(
        hooks=[
            AuditLogHooks(),
            RateLimitHooks(max_calls=30, window_seconds=60),
            SensitiveWordHooks(words=["机密", "内部资料"]),
        ]
    )
    agent = create_agent(AgentBackend.QCODER, config)  # 或 AgentBackend.DEEPAGENTS
    response = agent.run("你好")
"""

from __future__ import annotations

import logging
import time

from agent_core import (
    AgentMessage,
    BaseAgentHooks,
    HookOutcome,
    HookResult,
    MessageRole,
)

_audit_logger = logging.getLogger("examples.hooks.audit")
_rate_logger = logging.getLogger("examples.hooks.rate_limit")
_policy_logger = logging.getLogger("examples.hooks.policy")


class AuditLogHooks(BaseAgentHooks):
    """审计日志 hooks：把关键事件写成结构化日志，用于审计与排障。

    触发时机（均为放行语义，返回 None）：
    - ``before_agent``：agent 执行开始（含 session_id，可关联同一次调用的所有事件）；
    - ``before_llm`` / ``after_llm``：每次 LLM 调用前后（模型名 / 消息数 / 输出长度）；
    - ``after_stop``：调用结束（reason 为 complete 或 error）。
    """

    def __init__(self, logger_name: str = "examples.hooks.audit") -> None:
        self.logger = logging.getLogger(logger_name)

    async def before_agent(self, context) -> None:
        self.logger.info(
            "before_agent backend=%s session=%s input_messages=%d",
            context.backend,
            context.session_id,
            len(context.input_messages),
        )

    async def before_llm(self, context) -> None:
        self.logger.info(
            "before_llm session=%s model=%s messages=%d",
            context.session_id,
            context.model,
            len(context.messages),
        )

    async def after_llm(self, context) -> None:
        content_len = len(context.response.content) if context.response else 0
        self.logger.info("after_llm session=%s content_len=%d", context.session_id, content_len)

    async def after_stop(self, context) -> None:
        self.logger.info(
            "after_stop session=%s reason=%s", context.session_id, context.reason
        )


class RateLimitHooks(BaseAgentHooks):
    """限流 hooks：滑动时间窗口内调用次数超过上限时 BLOCK。

    用 ``time.monotonic()`` 记录调用时刻（不受系统时钟回拨影响），
    每次 ``before_prompt`` 先剔除窗口外的旧记录，再判断是否超限。
    注意：hooks 实例是进程内共享的，多线程调用需自行加锁。
    """

    def __init__(self, max_calls: int = 30, window_seconds: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._call_times: list[float] = []

    async def before_prompt(self, context) -> HookResult | None:
        now = time.monotonic()
        # 只保留仍在窗口内的调用记录，防止列表无限增长
        self._call_times = [t for t in self._call_times if now - t < self.window_seconds]
        if len(self._call_times) >= self.max_calls:
            _rate_logger.warning(
                "rate limit exceeded: %d calls in %.0fs window", self.max_calls, self.window_seconds
            )
            return HookResult(
                outcome=HookOutcome.BLOCK,
                reason=f"调用频率超限：{self.window_seconds:.0f}s 内最多 {self.max_calls} 次",
            )
        self._call_times.append(now)
        return None  # 未超限，放行


class SensitiveWordHooks(BaseAgentHooks):
    """敏感词拦截 hooks：输入包含敏感词时 BLOCK。

    在 ``before_prompt``（可拦截事件）检查全部消息的 content，
    命中任意敏感词即返回 BLOCK 终止调用。
    """

    def __init__(self, words: list[str] | None = None) -> None:
        self.words = words or ["机密", "内部资料", "密码"]

    async def before_prompt(self, context) -> HookResult | None:
        text = "\n".join(message.content for message in context.messages)
        for word in self.words:
            if word in text:
                _policy_logger.warning("sensitive word blocked: %r", word)
                return HookResult(
                    outcome=HookOutcome.BLOCK,
                    reason=f"输入包含敏感词：{word}",
                )
        return None


class ContextInjectHooks(BaseAgentHooks):
    """上下文注入 hooks：MODIFY 在提示词前插入一条 system 上下文消息。

    例如注入当前时间 / 用户身份 / 业务规则，让每次调用都带上统一上下文。
    用 ``before_prompt``（每次 run 一次）而不是 ``before_llm``（每次模型调用一次），
    避免 agent 循环多轮时重复注入。
    """

    def __init__(self, context_text: str) -> None:
        self.context_text = context_text

    async def before_prompt(self, context) -> HookResult | None:
        return HookResult(
            outcome=HookOutcome.MODIFY,
            data={
                "messages": [
                    AgentMessage(role=MessageRole.SYSTEM, content=self.context_text),
                    *context.messages,
                ]
            },
        )
