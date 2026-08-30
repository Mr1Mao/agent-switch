"""agent-switch 基础用法示例：DEEPAGENTS + QCODER sync run 与 QCODER stream。

运行方式（项目根目录下）：
    python -m examples.basic_usage
    python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio

from agent_switch import AgentBackend, AgentMessage, MessageRole, create_agent


def demo_sync_run() -> None:
    """同步 run：DEEPAGENTS（真实 SDK）与 QCODER（真实 qoder-agent-sdk）。"""
    # DEEPAGENTS：需要已安装 deepagents 且配置了模型/凭据
    # （pip install 'agent-switch[deepagents]'；通过 AgentConfig.extra["model"]
    #   传入已构建的 ChatModel，或用 config/deepseek_flash.py 从环境变量构建）。
    deep_agent = create_agent(AgentBackend.DEEPAGENTS)
    try:
        response = deep_agent.run("你好，介绍一下你自己")
        print("[deepagents]", response.content)
    except Exception as exc:  # noqa: BLE001 - 未配置模型/凭据时给出提示，不影响其余演示
        print(f"[deepagents] 跳过真实调用（需要配置模型与凭据）: {type(exc).__name__}")

    # QCODER：真实 qoder-agent-sdk（需安装 qodercli 并登录，pip install 'agent-switch[qcoder]'）
    qcoder_agent = create_agent(AgentBackend.QCODER)
    try:
        response = qcoder_agent.run([AgentMessage(role=MessageRole.USER, content="讲个笑话")])
        print("[qcoder]", response.content)
    except Exception as exc:  # noqa: BLE001 - 未安装 qodercli / 未登录时给出提示
        print(f"[qcoder] 跳过真实调用（需要 qodercli 并登录）: {type(exc).__name__}")


async def demo_stream() -> None:
    """流式 stream：逐块打印内容。"""
    agent = create_agent(AgentBackend.QCODER)
    async for chunk in agent.stream("流式输出测试"):
        if chunk.delta_thinking:
            print(f"[thinking] {chunk.delta_thinking}")
        if chunk.delta_content:
            print(chunk.delta_content, end="")
        if chunk.is_finish:
            print("\n[finish]")


def main() -> None:
    """基础用法示例主入口。"""
    demo_sync_run()
    asyncio.run(demo_stream())


if __name__ == "__main__":
    main()
