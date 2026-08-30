"""agent-core 统一 API 调用入口演示。

覆盖所有调用入口：backend 枚举/字符串、str/list 输入、构造时/调用时配置、
同步 run、异步 stream、hooks 生命周期、未知后端异常、deepagents 真实后端。

运行方式（项目根目录下）：
    python -m examples
    python examples/entry_demo.py
    python -m examples.entry_demo
"""

from __future__ import annotations

import asyncio

from agent_core import (
    AgentBackend,
    AgentConfig,
    AgentMessage,
    AgentModel,
    BackendNotFoundError,
    BaseAgentHooks,
    HookOutcome,
    HookResult,
    MessageRole,
    create_agent,
)


class AuditHooks(BaseAgentHooks):
    """审计 hook：LLM 调用前打印模型名。"""

    async def before_llm(self, context) -> None:
        print(f"    [hook] beforeLLM model={context.model}")


class ModifyHooks(BaseAgentHooks):
    """改写 hook：beforePrompt 时替换用户消息。"""

    async def before_prompt(self, context) -> HookResult | None:
        return HookResult(
            outcome=HookOutcome.MODIFY,
            data={
                "messages": [
                    AgentMessage(role=MessageRole.USER, content="(已被 beforePrompt hook 改写)")
                ]
            },
        )


def demo_backend_enum_and_string() -> None:
    """入口 1：create_agent 接受 AgentBackend 枚举或字符串。"""
    agent_a = create_agent(AgentBackend.QCODER)
    agent_b = create_agent("qcoder")
    print(f"[入口1] 枚举创建 -> {type(agent_a).__name__}；字符串创建 -> {type(agent_b).__name__}")


def demo_run_inputs() -> None:
    """入口 2：run 接受 str 或 list[AgentMessage] 两种输入。"""
    agent = create_agent(AgentBackend.QCODER)
    try:
        response = agent.run("字符串输入")
        print(f"[入口2] run(str)   -> {response.content}")
        response = agent.run([AgentMessage(role=MessageRole.USER, content="消息列表输入")])
        print(f"[入口2] run(list)  -> {response.content}")
    except Exception as exc:  # noqa: BLE001 - 未安装 qodercli / 未登录时提示
        print(f"[入口2] 跳过 qcoder 真实调用（需 qodercli 并登录）: {type(exc).__name__}")


def demo_run_with_config() -> None:
    """入口 3：构造时传入 AgentConfig，或调用时按次传入（不影响构造时默认配置）。"""
    agent = create_agent(
        AgentBackend.QCODER,
        AgentConfig(
            system_prompt="You are terse.",
            model=AgentModel(name="deepseek-v4-flash"),
            hooks=AuditHooks(),
        ),
    )
    print("[入口3] 构造时配置 -> run() 使用默认 config:")
    try:
        print(f"    -> {agent.run('hello').content}")

        override = AgentConfig(
            system_prompt="Override.",
            hooks=[AuditHooks(), ModifyHooks()],
        )
        print("[入口3] 调用时配置 -> run(input, AgentConfig(...)) 按次覆盖:")
        print(f"    -> {agent.run('原始 prompt', override).content}")
    except Exception as exc:  # noqa: BLE001 - 未安装 qodercli / 未登录时提示
        print(f"[入口3] 跳过 qcoder 真实调用（需 qodercli 并登录）: {type(exc).__name__}")


async def demo_stream() -> None:
    """入口 4：异步流式 stream 逐块消费。"""
    agent = create_agent(AgentBackend.QCODER)
    print("[入口4] stream 流式调用:")
    try:
        async for chunk in agent.stream("流式调用"):
            if chunk.delta_content:
                print(chunk.delta_content, end="")
            if chunk.is_finish:
                print(" [finish]")
    except Exception as exc:  # noqa: BLE001 - 未安装 qodercli / 未登录时提示
        print(f"跳过 qcoder 真实调用（需 qodercli 并登录）: {type(exc).__name__}")


def demo_hooks_lifecycle() -> None:
    """入口 5：hooks 生命周期（run 触发 6 个阶段事件）。"""
    events: list[str] = []

    class TrackingHooks(BaseAgentHooks):
        async def before_agent(self, context):
            events.append("beforeAgent")

        async def before_prompt(self, context):
            events.append("beforePrompt")

        async def before_llm(self, context):
            events.append("beforeLLM")

        async def after_llm(self, context):
            events.append("afterLLM")

        async def after_agent(self, context):
            events.append("afterAgent")

        async def after_stop(self, context):
            events.append("afterStop")

    agent = create_agent(AgentBackend.QCODER, AgentConfig(hooks=TrackingHooks()))
    try:
        agent.run("hello")
        print(f"[入口5] hooks 事件顺序 -> {' → '.join(events)}")
    except Exception as exc:  # noqa: BLE001 - 未安装 qodercli / 未登录时提示
        print(f"[入口5] 跳过 qcoder 真实调用（需 qodercli 并登录）: {type(exc).__name__}")


def demo_unknown_backend() -> None:
    """入口 6：未知后端抛 BackendNotFoundError。"""
    try:
        create_agent("not-a-backend")
    except BackendNotFoundError as exc:
        print(f"[入口6] BackendNotFoundError: {exc}")


def demo_deepagents_guarded() -> None:
    """入口 7：deepagents 真实后端（未配置模型/凭据时跳过并给出提示）。"""
    agent = create_agent(AgentBackend.DEEPAGENTS)
    try:
        response = agent.run("你好，介绍一下你自己")
        print(f"[入口7] deepagents -> {response.content}")
    except Exception as exc:  # noqa: BLE001 - 演示脚本中吞掉凭据缺失类异常
        print(f"[入口7] 跳过真实调用（需配置模型与凭据）: {type(exc).__name__}")


def main() -> None:
    """统一 API 调用入口演示的主入口。"""
    print("== agent-core 统一 API 调用入口演示 ==\n")
    demo_backend_enum_and_string()
    demo_run_inputs()
    demo_run_with_config()
    asyncio.run(demo_stream())
    demo_hooks_lifecycle()
    demo_unknown_backend()
    demo_deepagents_guarded()
    print("\n== 演示结束 ==")


if __name__ == "__main__":
    main()
