# agent-core

**Agent SDK 统一抽象层（deepagents、Qcoder SDK 等）**

`agent-core` 为业务代码提供一套稳定的统一 API（`create_agent` + `run` / `stream`），
上层类型与调用方式不变，即可切换底层 Agent 框架（deepagents、qcoder 等）。

## 特性

- **一套 API，多个后端**：`create_agent(AgentBackend.DEEPAGENTS | "qcoder", config)` 返回适配器，
  统一暴露 `run(input) -> AgentResponse` 与 `stream(input) -> AsyncIterator[AgentChunk]`。
- **完备的类型系统**：`AgentConfig`、`AgentMessage`、`AgentTool`、`AgentSkillsConfig`、
  `AgentMcpConfig`、`AgentSubagent`、`AgentChunk`、`AgentResponse` 等（Pydantic v2，`extra="forbid"`）。
- **Hooks 生命周期**：12 个异步 hook 事件（`beforeAgent` … `afterStop`），支持 `BLOCK` / `MODIFY`
  结果与按 `AgentConfig` 配置的 hooks 列表。
- **结构化日志**：自动脱敏密钥、汇总 config/input 摘要、Dev 与 JSON 两种 Formatter，
  仅配置 `agent_core` 命名空间（不添加全局 handler）。
- **延迟依赖**：`deepagents` 仅在真正使用 deepagents 后端时才导入；`import agent_core` 不强制要求安装。

## 安装

```bash
pip install agent-core
# 或按需安装额外依赖
pip install "agent-core[deepagents]"    # deepagents 后端
pip install "agent-core[qcoder]"        # qcoder 后端（当前为 stub）
pip install "agent-core[all]"
```

## Quick start

```python
from agent_core import AgentConfig, AgentMessage, MessageRole, create_agent, AgentBackend

# qcoder 是 stub 后端——无需 SDK 即可测试 hooks 与流式行为
config = AgentConfig(system_prompt="Be concise.")
agent = create_agent(AgentBackend.QCODER, config)

response = agent.run("Tell me a joke")
print(response.content)  # [stub] Tell me a joke

async def demo_stream() -> None:
    async for chunk in agent.stream("Hello"):
        print(chunk.delta_content, end="")
```

## 运行示例

```bash
python -m examples                         # 统一 API 调用入口演示（覆盖全部调用方式）
python -m examples.basic_usage             # DEEPAGENTS + QCODER sync run 与 QCODER stream
python -m examples.deepseek_flash_usage    # 通过环境变量配置 DeepSeek Flash（需 DEEPSEEK_API_KEY）
```

可复用的 hooks 实现位于 `examples/hooks.py`（审计日志 / 限流 / 敏感词拦截 /
上下文注入）——可在你自己的 entry 入口中 import，并配置到 `AgentConfig(hooks=[...])`。
```

## DeepAgents + `extra["model"]`

真实 `deepagents` 后端可通过 `AgentConfig.extra["model"]` 传入已构建好的 LangChain `ChatModel`，
它拥有最高优先级：

```python
from langchain_deepseek import ChatDeepSeek
from agent_core import AgentConfig, create_agent, AgentBackend

model = ChatDeepSeek(model="deepseek-v4-flash", api_key="sk-...")
config = AgentConfig(
    system_prompt="You are a helpful assistant.",
    tools=[AgentTool(name="search", handler=my_search_tool)],
    extra={"model": model},  # ← 已构建的 ChatModel 优先
)
agent = create_agent(AgentBackend.DEEPAGENTS, config)
response = agent.run("What is the weather in Paris?")
```

也可以让 agent-core 自行构建模型：`AgentModel(name=..., api_key=..., base_url=...)`
会映射到 `langchain.chat_models.init_chat_model`；只有 `AgentModel(name="openai:gpt-4o-mini")`
（无 key/url）时则直接把模型名字符串透传给 deepagents。

## Hooks

```python
from agent_core import (
    AgentConfig, AgentHookEvent, BaseAgentHooks, HookOutcome, HookResult, create_agent,
)

class AuditHooks(BaseAgentHooks):
    async def before_llm(self, context) -> None:
        print(f"[audit] beforeLLM model={context.model}")

class RateLimitHooks(BaseAgentHooks):
    async def before_prompt(self, context):
        if len(context.messages) > 10:
            return HookResult(outcome=HookOutcome.BLOCK, reason="rate limit exceeded")

# 单个实例或列表均可（自动归一化）
config = AgentConfig(hooks=[AuditHooks(), RateLimitHooks()])
# 或 AgentConfig(hooks=AuditHooks())
agent = create_agent(AgentBackend.QCODER, config)
response = agent.run("hello")
```

run/stream 按顺序触发 6 个生命周期事件：
`beforeAgent → beforePrompt → beforeLLM → [SDK] → afterLLM → afterAgent → afterStop`。
hook 可返回 `HookResult(outcome=BLOCK, reason=...)`（抛出 `HookBlockedError`）
或 `HookResult(outcome=MODIFY, data={"messages": [...]})`（改写 prompt 消息）。

hooks 分两层触发：

- **agent 级**（每次 agent 执行一次）：`beforeAgent`、`beforePrompt`、`afterAgent` ——
  deepagents 后端在 SDK 内部的 `before_agent` / `after_agent`（图的 entry / exit 节点）触发；
- **调用级**（agent 循环内每次 LLM / 工具调用一次）：`beforeLLM`、`afterLLM`、`beforeTool`、
  `afterTool`、`afterToolError` —— 通过注入的 `AgentHooksMiddleware`
  （`wrap_model_call` / `wrap_tool_call`）桥接，在真实的模型/工具调用点触发
  （例如 agent 循环多次调用工具时会触发多次）；
- `afterStop`（reason 为 `complete` / `error`）由 adapter 在 `run` / `stream` 边界触发 ——
  `after_agent` 只在成功路径执行，因此运行失败时 adapter 会补发
  `afterAgent(error)` + `afterStop(error)`。

stub 后端（`qcoder`）的全部 6 个事件在 adapter 层每次 run 触发一次。
`beforePermission / beforeSubagent / afterSubagent` 已声明但尚未桥接。

### Hooks ↔ deepagents 实现映射

| agent-core hook          | deepagents 实现方式                                          | 层次 / 时机                                    |
| ------------------------ | ------------------------------------------------------------ | ---------------------------------------------- |
| `beforeAgent`          | `AgentHooksMiddleware.before_agent` / `abefore_agent`（entry 节点） | agent 级，每次 agent 执行一次 |
| `beforePrompt`           | `AgentHooksMiddleware.before_agent` / `abefore_agent`（entry 节点） | agent 级，每次 agent 执行一次 |
| `beforeLLM`              | `AgentHooksMiddleware.wrap_model_call` / `awrap_model_call`，`handler(request)` 之前 | 调用级，每次 LLM 调用一次 |
| `afterLLM`               | `AgentHooksMiddleware.wrap_model_call` / `awrap_model_call`，`handler(request)` 之后 | 调用级，每次 LLM 调用一次 |
| `beforeTool`             | `AgentHooksMiddleware.wrap_tool_call` / `awrap_tool_call`，执行工具前 | 调用级，每次工具调用一次 |
| `afterTool`              | `AgentHooksMiddleware.wrap_tool_call` / `awrap_tool_call`，工具返回后 | 调用级，每次工具调用一次 |
| `afterToolError`         | `AgentHooksMiddleware.wrap_tool_call` 异常分支，随后重抛 | 调用级，每次失败的工具调用一次 |
| `afterAgent`           | `AgentHooksMiddleware.after_agent` / `aafter_agent`（exit 节点）；失败时 adapter 补发 `afterAgent(error)` | agent 级，每次成功执行一次 |
| `afterStop`              | adapter 层（`_finalize_run_success_*` / `_finalize_run_error_*`，reason 为 `complete` / `error`） | agent 级，每次 run / stream 一次 |
| `beforePermission` / `beforeSubagent` / `afterSubagent` | 尚未桥接 | — |

deepagents 后端实现细节：

- `DeepAgentsAdapter._build_agent()` 在 `AgentConfig.hooks` 非空时，把 `AgentHooksMiddleware`
  实例追加到 `create_deep_agent(middleware=[...])`，与 `config.extra["middleware"]`
  传入的用户中间件共存；
- 中间件通过 `session_provider` 闭包（绑定 adapter 的 `_session_id` / `_correlation_id`）
  读取当前会话标识，保证所有 Context 共用同一会话；
- `before_agent` / `after_agent` 是图的 entry / exit 节点：每次 agent 执行恰好各执行一次
  （子代理是独立编译的图，不会触发）。`beforePrompt` 返回 `MODIFY` 时通过
  state 更新 `{"messages": [...]}` 改写初始消息；返回 `BLOCK` 时在 SDK 内部抛
  `HookBlockedError`，中断整个 agent 运行；
- `beforeLLM` 返回 `MODIFY` 时通过 `request.override(messages=...)` 改写真实请求；
- 由于这些事件已在 SDK 内部触发，`DeepAgentsAdapter` 设置 `call_hooks_via_middleware = True`
  与 `agent_hooks_via_middleware = True`，避免 adapter 层重复触发；`afterStop`
  （以及错误路径的 `afterAgent`）仍留在 adapter —— 因为图抛异常时 `after_agent`
  不会执行；
- 构建的图带缓存，缓存键包含配置的 hooks 指纹：更换 hooks 会重建 agent，
  而不是复用旧图；
- `beforeLLM` / `afterLLM` 选用 `wrap_model_call` 而非 `before_model` / `after_model`
  节点钩子，是因为它能改写真实请求（`MODIFY` 需要）且可不调用模型直接短路（`BLOCK` 需要）。

## Streaming

```python
async for chunk in agent.stream("hello"):
    if chunk.delta_content:
        print(chunk.delta_content, end="")
    if chunk.delta_thinking:
        print(f"\n[thinking] {chunk.delta_thinking}")
```

`AgentChunk` 字段：`delta_content`、`delta_thinking`、`delta_tool_call`、`is_finish`、`meta`。
流结束前总会 yield 一条 `is_finish=True` 的分片。

## Message model

| 字段         | 类型                   | 说明                                        |
| ------------ | ---------------------- | ------------------------------------------- |
| `role`       | `MessageRole`          | `user` / `assistant` / `system` / `tool`    |
| `content`    | `str`                  | 文本内容                                    |
| `thinking`   | `str \| None`          | 推理内容（取决于模型）                      |
| `tool_calls` | `list[ToolCall]`       | `{id, name, arguments}`                     |
| `tool_result`| `ToolResult \| None`   | `{tool_call_id, content}`                   |
| `meta`       | `dict`                 | 后端元信息（`langchain_type` 等）           |
| `_raw`       | `PrivateAttr`          | 仅供适配器调试，永不参与序列化              |

## DeepAgents ↔ agent-core 映射表

| agent-core               | deepagents / LangChain                                 |
| ------------------------ | ------------------------------------------------------ |
| `MessageRole.USER`       | `HumanMessage`                                         |
| `MessageRole.SYSTEM`     | `SystemMessage`                                        |
| `MessageRole.ASSISTANT`  | `AIMessage`（带 `tool_calls: [{id, name, args}]`）     |
| `MessageRole.TOOL`       | `ToolMessage`（`tool_call_id`）                        |
| `AgentMessage.thinking`  | 提取优先级：`additional_kwargs.reasoning_content` → `additional_kwargs.thinking` → `content_blocks` 中 `reasoning` / `thinking` 类型块 |
| `AgentTool.handler`      | deepagents `tools`（或经 `extra["tools"]` 解析）        |
| `AgentSkillsConfig.sources` | deepagents `skills`                                  |
| `AgentSubagent`          | deepagents `subagents` 字典                             |
| `AgentConfig.extra`      | 白名单透传：`middleware, memory, permissions, backend, interrupt_on, response_format, state_schema, context_schema, checkpointer, store, debug, name, cache` |
| `AgentResponse.raw`      | 图 `invoke` / `astream` 的原始结果                     |
| 流式分片                 | `graph.astream(stream_mode="messages")` → 一条 LangChain chunk 可能产生多个 `AgentChunk`（`delta_content` / `delta_thinking` / `delta_tool_call`） |

`thinking` / `meta` **不会**发送给后端（输入方向）；仅在回程时提取。

## Current status

- [x] 类型系统与统一 API（`create_agent` / `run` / `stream`）
- [x] `deepagents` 后端（真实实现，延迟导入）
- [x] `qcoder` 后端（stub 适配器，hooks + 流式演示）
- [x] Hooks 生命周期（12 事件，BLOCK / MODIFY）
- [x] deepagents 调用级事件中间件桥接（`beforeLLM` / `afterLLM` / `beforeTool` / `afterTool` / `afterToolError`）
- [x] 结构化日志（脱敏、Dev / JSON Formatter）
- [ ] `beforePermission` / `beforeSubagent` / `afterSubagent` 桥接
- [ ] 真实 Qcoder SDK 集成
- [ ] MCP server 接线

## Roadmap

1. 桥接剩余 `beforePermission` / `beforeSubagent` / `afterSubagent` 事件。
2. 基于 `qoder-agent-sdk` 实现真实 qcoder 后端。
3. 增加按后端的 capability 自省（`supports_*` 标志）。
4. 针对 dev extras 的 SDK 版本固化公共 API 的类型契约。

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q          # 45 passed
ruff check .       # lint
mypy -p agent_core # strict 类型检查
```

目录结构：

```
src/agent_core/
├── abc.py            # AgentAdapter（抽象基类）
├── adapter_base.py   # hooks 生命周期编排
├── factory.py        # create_agent
├── registry.py       # BackendRegistry
├── logging.py        # configure_logging / summarize / formatters
├── hooks/            # enums, context, result, dispatcher, emitter, base
├── types/            # 统一类型系统
├── utils/            # 输入归一化
└── backends/         # stub, qcoder, deepagents（adapter + mapping）
```

## License

MIT
