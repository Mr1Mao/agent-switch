"""DeepSeek Flash 配置使用示例（deepagents 后端）。

运行方式（项目根目录下，需先配置凭据）：
    cp .env.example .env   # 填入 DEEPSEEK_API_KEY
    python -m examples.deepseek_flash_usage
    python examples/deepseek_flash_usage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接运行本脚本时导入项目根目录的 config/ 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_core import AgentBackend, create_agent
from config.deepseek_flash import build_deepseek_flash_config


def main() -> None:
    """从环境变量构建 DeepSeek Flash 配置并运行 deepagents 后端。"""
    config = build_deepseek_flash_config(
        system_prompt="你是 DeepSeek 助手，回答尽量简洁。"
    )
    if not config.model or not config.model.api_key:
        print(
            "未检测到 DEEPSEEK_API_KEY：请先 `cp .env.example .env` 并填入真实 Key，"
            "或调用 build_deepseek_flash_config(api_key=...) 显式传入。"
        )
        return

    print(f"使用模型: {config.model.name}")
    agent = create_agent(AgentBackend.DEEPAGENTS, config)
    response = agent.run("用一句话介绍 agent-core")
    print(response.content)


if __name__ == "__main__":
    main()
