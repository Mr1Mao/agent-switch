"""agent-switch 示例包。

可直接运行（项目根目录下）：
- ``python -m examples``                  统一 API 调用入口演示
- ``python -m examples.basic_usage``      DEEPAGENTS + QCODER 基础用法
- ``python -m examples.deepseek_flash_usage``  DeepSeek Flash 配置示例

可复用的实现：
- ``examples.hooks``                      真实 hooks 实现（审计日志 / 限流 / 敏感词 / 上下文注入），
                                          可在你自己的 entry 入口中 import 并配置到 AgentConfig.hooks。
"""
