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
pip install "agent-core[qcoder]"        # qcoder backend (qoder-agent-sdk)
pip install "agent-core[all]"
```

> The `qcoder` backend runs the real `qoder-agent-sdk`, which spawns the
> `qodercli` CLI. Install the CLI and log in once (`qodercli auth`) before use.

## Quick start

```python
from agent_core import AgentConfig, AgentMessage, MessageRole, create_agent, AgentBackend

# qcoder runs on the real qoder-agent-sdk (needs `qodercli` installed & logged in)
config = AgentConfig(system_prompt="Be concise.")
agent = create_agent(AgentBackend.QCODER, config)

response = agent.run("Tell me a joke")
print(response.content)

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

## Qcoder ↔ agent-core mapping

The `qcoder` backend runs on the real `qoder-agent-sdk` (which drives the
`qodercli` CLI). It supports message format normalization, the unified hooks
lifecycle, streaming, tools / skills / MCP configuration, and session identity.

### Message normalization

Input direction (`AgentMessage` → qoder CLI wire format, via
`agent_core.backends.qcoder.mapping.agent_messages_to_qoder_wire`):

| agent-core                | qoder wire                                                    |
| ------------------------- | ------------------------------------------------------------- |
| `MessageRole.USER`        | `{"type":"user","message":{"role":"user","content":<str>}}`   |
| `MessageRole.ASSISTANT`   | text + `tool_use` blocks (`{"type":"tool_use","id","name","input"}`) in one `user` message |
| `MessageRole.TOOL`        | `{"type":"tool_result","tool_use_id","content","is_error"}` block |
| `MessageRole.SYSTEM`      | not sent (mapped to `QoderAgentOptions.system_prompt`)        |
| `thinking` / `meta`       | not sent (input direction)                                    |

Output direction (qoder SDK `Message` → `AgentMessage`):

| qoder SDK                 | agent-core                              |
| ------------------------- | --------------------------------------- |
| `AssistantMessage`        | `role=assistant`, `content` (joined `TextBlock`s) |
| `ThinkingBlock`           | `thinking`                              |
| `ToolUseBlock`            | `ToolCall(id, name, input→arguments)`   |
| `UserMessage`             | `role=user`                             |
| `SystemMessage`           | `role=system` (only `meta`)             |
| `ResultMessage`           | terminal → `AgentResponse` (content, raw, backend) |

### Hooks mapping

Session-level events (`beforeAgent` / `beforePrompt` / `beforeLLM` / `afterLLM` /
`afterAgent` / `afterStop`) fire at the adapter level once per `run` / `stream`,
exactly as documented in the Hooks chapter. Call-level events are bridged into
the Qoder SDK native hook system:

| agent-core hook      | Qoder HookEvent      | BLOCK / MODIFY mapping                              |
| -------------------- | -------------------- | --------------------------------------------------- |
| `beforeTool`         | `PreToolUse`         | BLOCK → `continue_:False, decision:"block"` + `permissionDecision:"deny"`; MODIFY(`updated_input`) → `updatedInput` |
| `afterTool`          | `PostToolUse`        | MODIFY(`updated_tool_output`) → `updatedToolOutput` |
| `afterToolError`     | `PostToolUseFailure` | notification only                                   |
| `beforePermission`   | `PermissionRequest`  | BLOCK → `permissionDecision:"deny"`                 |
| `beforeSubagent`     | `SubagentStart`      | notification only                                   |
| `afterSubagent`      | `SubagentStop`       | notification only                                   |

Only events whose hook class actually overrides the method are registered, so an
empty hooks list adds no callbacks to the CLI.

### Configuration mapping (`AgentConfig` → `QoderAgentOptions`)

| agent-core              | QoderAgentOptions                                    |
| ----------------------- | ---------------------------------------------------- |
| `AgentModel.name` / `extra["model"]` (str) | `model`                       |
| `system_prompt`         | `system_prompt`                                      |
| `tools` (with `handler`) | in-process SDK MCP server via `create_sdk_mcp_server` + `allowed_tools` |
| `skills`                | `skills` (`sources` list / `enable_all` → `"all"`)   |
| `mcp` (`AgentMcpConfig`) | `mcp_servers` (stdio / http) + `allowed_mcp_server_names` |
| `extra` whitelist       | `permission_mode, max_turns, session_id, cwd, auth, allowed_tools, disallowed_tools, can_use_tool, include_partial_messages, continue_conversation, resume, settings, agents, agent, user, env, cli_path` |
| default                 | `auth=qodercli_auth()` (reuse local login state)     |

### Streaming

`stream()` iterates `qoder_agent_sdk.query(prompt=wire_messages, options=...)`;
each SDK `AssistantMessage` is mapped to one or more `AgentChunk`
(`delta_thinking` / `delta_content` / `delta_tool_call`), and the stream always
ends with a chunk carrying `is_finish=True`. `run()` wraps the same async flow
with `asyncio.run` and returns the terminal `ResultMessage` as `AgentResponse`
(falling back to the accumulated assistant text if no result message arrives).
Token-level partial messages (`StreamEvent`) are not enabled by default.

### Runtime requirements

- `pip install "agent-core[qcoder]"` (pulls `qoder-agent-sdk`, `mcp`, `anyio`)
- Install the `qodercli` CLI and log in once (`qodercli auth`)
- Sync `run()` uses `asyncio.run` internally: calling it inside a running event
  loop raises `RuntimeError` — use `stream()` in async code.

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

For the `qcoder` backend, the six session-level events fire at the adapter level
(one per `run` / `stream`), while `beforeTool` / `afterTool` / `afterToolError` /
`beforePermission` / `beforeSubagent` / `afterSubagent` are bridged to the Qoder
SDK's native hooks (`PreToolUse` / `PostToolUse` / `PostToolUseFailure` /
`PermissionRequest` / `SubagentStart` / `SubagentStop`) and fire inside the CLI.
`beforePermission / beforeSubagent / afterSubagent` are declared but not yet
bridged for the `deepagents` backend.

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
- [x] `qcoder` backend (real implementation on `qoder-agent-sdk`: message normalization, hooks bridging, streaming, tools / skills / MCP)
- [x] Hooks lifecycle (12 events, BLOCK / MODIFY)
- [x] deepagents call-level hook bridging via `AgentHooksMiddleware` (`beforeLLM` / `afterLLM` / `beforeTool` / `afterTool` / `afterToolError`)
- [x] qcoder call-level hook bridging via Qoder native hooks (`PreToolUse` / `PostToolUse` / `PostToolUseFailure` / `PermissionRequest` / `SubagentStart` / `SubagentStop`)
- [x] Structured logging (redaction, Dev / JSON formatters)
- [ ] `beforePermission` / `beforeSubagent` / `afterSubagent` bridging for the `deepagents` backend
- [ ] Token-level partial message streaming for `qcoder` (`StreamEvent`)
- [ ] Qoder `QoderSDKClient` bidirectional / interrupt support

## Roadmap

1. Bridge the remaining `beforePermission` / `beforeSubagent` / `afterSubagent` events for `deepagents`.
2. Add per-backend capability introspection (`supports_*` flags).
3. Officially type the public API against the dev extras' SDK versions.

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
└── backends/         # stub, qcoder, deepagents (adapter + mapping + hooks bridge)
```

## License

MIT
