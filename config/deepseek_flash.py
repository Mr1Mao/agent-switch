"""DeepSeek Flash 配置工厂：从环境变量构建 AgentConfig。

支持 ``python-dotenv``（可选）：若已安装则自动加载项目根目录的 ``.env``。
"""

from __future__ import annotations

import os

try:  # python-dotenv 为可选依赖
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - 未安装时静默跳过
    pass

from agent_core import AgentConfig, AgentModel

#: DeepSeek Flash 默认模型引用（deepagents 可直接透传的模型名字符串）
DEFAULT_DEEPSEEK_MODEL_REF = "deepseek:deepseek-v4-flash"


def build_deepseek_flash_config(
    system_prompt: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> AgentConfig:
    """构建 DeepSeek Flash 的 AgentConfig。

    未显式传入的参数从环境变量读取：
    ``DEEPSEEK_API_KEY`` / ``DEEPSEEK_BASE_URL`` / ``DEEPSEEK_MODEL``。
    """
    api_key_value = api_key or os.getenv("DEEPSEEK_API_KEY")
    base_url_value = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model_value = model or os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL_REF)
    return AgentConfig(
        model=AgentModel(
            name=model_value,
            api_key=api_key_value,
            base_url=base_url_value,
        ),
        system_prompt=system_prompt,
    )
