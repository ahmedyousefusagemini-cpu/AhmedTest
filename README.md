# free-pi proxy

OpenAI-compatible local proxy in front of the free-pi backend, with a
concurrency slot pool so one account can serve many parallel requests.

```bash
cd proxy
node proxy.mjs          # PORT=8085 HOST=127.0.0.1 PROXY_MAX_PARALLEL=16
```

Point any OpenAI client at `http://127.0.0.1:8085/v1`.

## Concurrency model

The free-pi backend enforces two independent limits:

| Limit | Scope | Value | Error |
| --- | --- | --- | --- |
| Per-session streams | one `x-session-id` | **1** | `429` `concurrent` |
| Per-account streams | all sessions | **16** | `409` `concurrent_session` |

The backend keeps **no conversation state** per session: reusing a session id
while sending no history returns no prior context. The session id is therefore a
pure mutex, not a context key.

`proxy.mjs` keeps a pool of `PROXY_MAX_PARALLEL` (default 16, clamped 1-16)
distinct session ids and hands one to each in-flight request. Requests beyond the
pool size queue rather than fail, and `429`/`409` concurrency rejections are
retried up to 8 times with jittered exponential backoff (300ms rising to 3s).

### Sizing

Keep the pool below 16 when you also run the interactive `npx free-pi-cli@latest`
session, since the 16-stream cap is shared per account. `PROXY_MAX_PARALLEL=6` is
a reasonable default for interactive use.

Parallelism does not enlarge or shrink any single request's context window. Each
request carries its own message history and is billed separately, so wide
parallelism spends prompt tokens faster per unit of wall-clock time. Watch the
daily allowance with `/usage` in free-pi.

## Sub-agents

`test/orchestrator.mjs` demonstrates one main context dispatching work to many
concurrent sub-agents with isolated histories, then merging their reports:

```bash
PROXY_API_KEY=... node test/orchestrator.mjs
```

Verified with 16 concurrent sub-agents: 16/16 succeeded, each sub-agent confirmed
it saw no prior context, and the orchestrator merged the fragments into the
expected answer.

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` / `HOST` | `8085` / `127.0.0.1` | Listen address |
| `PROXY_API_KEY` | *(none)* | Require `Authorization: Bearer` on the proxy |
| `PROXY_MAX_PARALLEL` | `16` | Session-id pool size, clamped to 1-16 |
| `FREEPI_BASE_URL` | `https://api.freepi.ai` | Backend |
| `FREEPI_CREDENTIALS_PATH` | `~/.free-pi/agent/credentials.json` | Reused CLI credentials |
| `MCP_MODE` | `auto` | `auto` connects configured MCP servers, `off` disables |
