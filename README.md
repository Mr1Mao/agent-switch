# agent-core

**Unified abstraction layer for agent SDKs (deepagents, Qcoder SDK, etc.)**

`agent-core` gives your business code a single, stable API — `create_agent` + `run` / `stream` —
so you can switch underlying agent frameworks (deepagents, qcoder, …) without touching
your upper-layer types or call sites.

## Features

- **One API, many backends**: `create_agent(AgentBackend.DEEPAGENTS | "qcoder", config)` returns
  an adapter exposing the same `run(input) -> AgentResponse` and `stream(input) -> AsyncIterator[AgentChunk]`.
- **Rich, validated type system**: `AgentConfig`, `AgentMessage`, `AgentTool`, `AgentSkillsConfig`,
  `AgentMcpConfig`, `AgentSubagent`, `AgentChunk`, `AgentResponse`, … (Pydantic v2, `extra="forbid"`).
- **Hooks lifecycle**: 12 async hook events (`beforeAgent` … `afterStop`) with
  `BLOCK` / `MODIFY` outcomes and a per-`AgentConfig` hook list.
- **Structured logging**: redacts secrets, summarizes config/input, Dev & JSON formatters,
  configured only for the `agent_core` namespace (no global handlers).
- **Lazy dependencies**: `deepagents` is imported only when the `deepagents` backend is actually used;
  `import agent_core` never requires it.

## Installation

```bash
pip install agent-core
# or with extras
pip install "agent-core[deepagents]"    # deepagents backend
pip install "agent-core[qcoder]"        # qcoder backend (stub for now)
pip install "agent-core[all]"
```

## Quick start

```python
from agent_core import AgentConfig, AgentMessage, MessageRole, create_agent, AgentBackend

# qcoder is a stub backend — great for testing hooks & streaming without an SDK
config = AgentConfig(system_prompt="Be concise.")
agent = create_agent(AgentBackend.QCODER, config)

response = agent.run("Tell me a joke")
print(response.content)  # [stub] Tell me a joke

async def demo_stream() -> None:
    async for chunk in agent.stream("Hello"):
        print(chunk.delta_content, end="")
```

## Run the demos

```bash
python -m examples                         # unified API entry-point demo (all call styles)
python -m examples.basic_usage             # DEEPAGENTS + QCODER sync run & QCODER stream
python -m examples.deepseek_flash_usage    # DeepSeek Flash via env config (needs DEEPSEEK_API_KEY)
```

Reusable hook implementations live in `examples/hooks.py` (audit logging, rate
limiting, sensitive-word blocking, context injection) — import them in your own
entry point and pass them to `AgentConfig(hooks=[...])`.
```

## DeepAgents + `extra["model"]`

For the real `deepagents` backend, pass a pre-built LangChain `ChatModel` through
`AgentConfig.extra["model"]` — it takes priority over `AgentModel`:

```python
from langchain_deepseek import ChatDeepSeek
from agent_core import AgentConfig, create_agent, AgentBackend

model = ChatDeepSeek(model="deepseek-v4-flash", api_key="sk-...")
config = AgentConfig(
    system_prompt="You are a helpful assistant.",
    tools=[AgentTool(name="search", handler=my_search_tool)],
    extra={"model": model},  # ← pre-built ChatModel wins
)
agent = create_agent(AgentBackend.DEEPAGENTS, config)
response = agent.run("What is the weather in Paris?")
```

Alternatively let agent-core build the model: `AgentModel(name=..., api_key=..., base_url=...)`
maps to `langchain.chat_models.init_chat_model`, while a bare `AgentModel(name="openai:gpt-4o-mini")`
passes the string straight through.

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

# single instance or a list — both are normalized
config = AgentConfig(hooks=[AuditHooks(), RateLimitHooks()])
# or: AgentConfig(hooks=AuditHooks())
agent = create_agent(AgentBackend.QCODER, config)
response = agent.run("hello")
```

Run/stream trigger six lifecycle events in order:
`beforeAgent → beforePrompt → beforeLLM → [SDK] → afterLLM → afterAgent → afterStop`.
Hooks may return `HookResult(outcome=BLOCK, reason=...)` (raises `HookBlockedError`)
or `HookResult(outcome=MODIFY, data={"messages": [...]})` (replaces the prompt messages).

Hooks fire on two layers:

- **Agent level** (once per agent execution): `beforeAgent`, `beforePrompt`
  and `afterAgent` — for the `deepagents` backend these fire inside the SDK,
  on the graph's `before_agent` / `after_agent` entry/exit nodes.
- **Call level** (once per LLM / tool call inside the agent loop): `beforeLLM`,
  `afterLLM`, `beforeTool`, `afterTool`, `afterToolError` — bridged through an
  injected `AgentHooksMiddleware` (`wrap_model_call` / `wrap_tool_call`), so they
  fire at the real model/tool call points (e.g. several times when the agent loops
  over tools).
- `afterStop` (reason `complete` / `error`) is fired by the adapter at the
  `run` / `stream` boundary — `after_agent` only runs on the success path, so the
  adapter re-fires `afterAgent(error)` + `afterStop(error)` when the run fails.

For stub backends (`qcoder`) all six events fire once per run at the adapter level.
`beforePermission / beforeSubagent / afterSubagent` are declared but not yet bridged.

### Hooks ↔ deepagents implementation mapping

| agent-core hook          | deepagents implementation                                        | Level / timing                                        |
| ------------------------ | ---------------------------------------------------------------- | ----------------------------------------------------- |
| `beforeAgent`          | `AgentHooksMiddleware.before_agent` / `abefore_agent` (entry node) | agent, once per agent execution |
| `beforePrompt`           | `AgentHooksMiddleware.before_agent` / `abefore_agent` (entry node) | agent, once per agent execution |
| `beforeLLM`              | `AgentHooksMiddleware.wrap_model_call` / `awrap_model_call`, before `handler(request)` | call, once per LLM call |
| `afterLLM`               | `AgentHooksMiddleware.wrap_model_call` / `awrap_model_call`, after `handler(request)` | call, once per LLM call |
| `beforeTool`             | `AgentHooksMiddleware.wrap_tool_call` / `awrap_tool_call`, before executing the tool | call, once per tool call |
| `afterTool`              | `AgentHooksMiddleware.wrap_tool_call` / `awrap_tool_call`, after the tool returned | call, once per tool call |
| `afterToolError`         | `AgentHooksMiddleware.wrap_tool_call` exception branch, then re-raise | call, once per failed tool call |
| `afterAgent`           | `AgentHooksMiddleware.after_agent` / `aafter_agent` (exit node); adapter re-fires `afterAgent(error)` on failure | agent, once per successful execution |
| `afterStop`              | adapter (`_finalize_run_success_*` / `_finalize_run_error_*`, reason `complete` / `error`) | agent, once per `run` / `stream` |
| `beforePermission` / `beforeSubagent` / `afterSubagent` | not bridged yet | — |

Implementation details for the `deepagents` backend:

- `DeepAgentsAdapter._build_agent()` appends an `AgentHooksMiddleware` instance to
  `create_deep_agent(middleware=[...])` whenever `AgentConfig.hooks` is non-empty;
  it coexists with user middleware passed via `config.extra["middleware"]`.
- The middleware reads the current session ids through a `session_provider` closure
  (bound to the adapter's `_session_id` / `_correlation_id`), so all contexts share
  the same session (session_id / correlation_id) as the adapter-level ones.
- `before_agent` / `after_agent` are the graph's entry / exit nodes: each fires
  exactly once per agent execution (sub-agents are separately compiled graphs and
  do not trigger them). `beforePrompt` returning `MODIFY` rewrites the initial
  state via `{"messages": [...]}`; `BLOCK` raises `HookBlockedError` inside the
  SDK, aborting the whole run.
- `beforeLLM` returning `MODIFY` rewrites the real request via
  `request.override(messages=...)`.
- Because these events fire inside the SDK, `DeepAgentsAdapter` sets
  `call_hooks_via_middleware = True` and `agent_hooks_via_middleware = True` so
  the adapter layer does not fire them a second time; `afterStop` (and the error
  path) remain at the adapter, since `after_agent` never runs when the graph raises.
- The built graph is cached; the cache key includes a fingerprint of the configured
  hooks, so changing hooks rebuilds the agent instead of reusing a stale graph.

## Streaming

```python
async for chunk in agent.stream("hello"):
    if chunk.delta_content:
        print(chunk.delta_content, end="")
    if chunk.delta_thinking:
        print(f"\n[thinking] {chunk.delta_thinking}")
```

`AgentChunk` fields: `delta_content`, `delta_thinking`, `delta_tool_call`, `is_finish`, `meta`.
The stream always ends with a chunk carrying `is_finish=True`.

## Message model

| Field        | Type                   | Notes                                        |
| ------------ | ---------------------- | -------------------------------------------- |
| `role`       | `MessageRole`          | `user` / `assistant` / `system` / `tool`     |
| `content`    | `str`                  | text content                                 |
| `thinking`   | `str \| None`          | reasoning content (model dependent)          |
| `tool_calls` | `list[ToolCall]`       | `{id, name, arguments}`                      |
| `tool_result`| `ToolResult \| None`   | `{tool_call_id, content}`                    |
| `meta`       | `dict`                 | backend metadata (`langchain_type`, …)       |
| `_raw`       | `PrivateAttr`          | adapter debugging only — never serialized    |

## DeepAgents ↔ agent-core mapping

| agent-core             | deepagents / LangChain                                    |
| ---------------------- | --------------------------------------------------------- |
| `MessageRole.USER`     | `HumanMessage`                                            |
| `MessageRole.SYSTEM`   | `SystemMessage`                                           |
| `MessageRole.ASSISTANT`| `AIMessage` (with `tool_calls: [{id, name, args}]`)       |
| `MessageRole.TOOL`     | `ToolMessage` (`tool_call_id`)                            |
| `AgentMessage.thinking`| extracted with priority: `additional_kwargs.reasoning_content` → `additional_kwargs.thinking` → `content_blocks` of type `reasoning` / `thinking` |
| `AgentTool.handler`    | deepagents `tools` (or resolved via `extra["tools"]`)     |
| `AgentSkillsConfig.sources` | deepagents `skills`                                   |
| `AgentSubagent`        | deepagents `subagents` dicts                               |
| `AgentConfig.extra`    | whitelisted passthrough: `middleware, memory, permissions, backend, interrupt_on, response_format, state_schema, context_schema, checkpointer, store, debug, name, cache` |
| `AgentResponse.raw`    | raw graph `invoke` / `astream` result                     |
| streaming chunks       | `graph.astream(stream_mode="messages")` → one LangChain chunk may produce several `AgentChunk`s (`delta_content` / `delta_thinking` / `delta_tool_call`) |

`thinking` / `meta` are **not** sent to the backend (input direction); they are only
extracted on the way back.

## Current status

- [x] Type system & unified API (`create_agent` / `run` / `stream`)
- [x] `deepagents` backend (real implementation, lazy import)
- [x] `qcoder` backend (stub adapter, hooks + streaming demo)
- [x] Hooks lifecycle (12 events, BLOCK / MODIFY)
- [x] deepagents call-level hook bridging via `AgentHooksMiddleware` (`beforeLLM` / `afterLLM` / `beforeTool` / `afterTool` / `afterToolError`)
- [x] Structured logging (redaction, Dev / JSON formatters)
- [ ] `beforePermission` / `beforeSubagent` / `afterSubagent` bridging
- [ ] Real Qcoder SDK integration
- [ ] MCP server wiring

## Roadmap

1. Bridge the remaining `beforePermission` / `beforeSubagent` / `afterSubagent` events.
2. Implement the real `qcoder` backend on `qoder-agent-sdk`.
3. Add per-backend capability introspection (`supports_*` flags).
4. Officially type the public API against the dev extras' SDK versions.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q          # 45 passed
ruff check .       # lint
mypy -p agent_core # strict type check
```

Layout:

```
src/agent_core/
├── abc.py            # AgentAdapter (abstract)
├── adapter_base.py   # hooks lifecycle orchestration
├── factory.py        # create_agent
├── registry.py       # BackendRegistry
├── logging.py        # configure_logging / summarize / formatters
├── hooks/            # enums, context, result, dispatcher, emitter, base
├── types/            # unified type system
├── utils/            # input normalization
└── backends/         # stub, qcoder, deepagents (adapter + mapping)
```

## License

MIT
