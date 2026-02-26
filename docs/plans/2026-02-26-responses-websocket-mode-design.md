# Responses API WebSocket Mode Design

## Overview

Add OpenAI Responses API WebSocket Mode to vLLM. Clients connect via
`wss://<host>/v1/responses`, send `response.create` messages, and receive
streaming events over the persistent WebSocket connection. This enables
incremental continuation via `previous_response_id` with connection-level
caching and reduced round-trip overhead for tool-heavy agent workflows.

Reference: https://developers.openai.com/api/docs/guides/websocket-mode/

## Decisions

- **Cache**: Independent per-connection (last response only). No interaction with the shared `response_store`.
- **Registration**: Under existing "generate" task — no new task type.
- **Timeout**: Graceful close with warning event at 55 min, close at 60 min.
- **Warmup** (`generate: false`): Cache input context, skip inference, return `response.created` + `response.completed`.
- **URL path**: Same `/v1/responses` path. Starlette routes HTTP POST vs WebSocket upgrade automatically.
- **Connection limit**: Default 100, configurable via `--max-websocket-connections`.
- **UUIDs**: UUID v7 (time-ordered, RFC 9562) for connection IDs and response IDs.

## Architecture

Approach: Mirror the existing Realtime API pattern (`RealtimeConnection`). A new
`WebSocketResponsesConnection` class manages the WebSocket lifecycle and delegates
to the existing `OpenAIServingResponses` for inference. No new inference pipeline.

## Section 1: Connection Management

### ConnectionContext

Per-connection state dataclass:

```python
@dataclass
class ConnectionContext:
    connection_id: str                              # "ws-{uuid7()}"
    last_response_id: str | None = None
    last_response: ResponsesResponse | None = None
    inflight: bool = False
    created_at: float = field(default_factory=time.monotonic)

    LIFETIME_SECONDS: ClassVar[int] = 3600          # 60 minutes
    WARNING_SECONDS: ClassVar[int] = 3300           # 55 minutes
```

### uuid7 helper

Python 3.10–3.12 lack `uuid.uuid7()`. A small helper generates RFC 9562 UUID v7
(48-bit unix-ms timestamp + version 7 + random bits).

### WebSocketResponsesConnection

Manages the WebSocket lifecycle:

- `__init__(websocket, serving)` — store references, create `ConnectionContext`
- `handle_connection()` — accept, start deadline timer, receive loop, cleanup
- `handle_event(event)` — parse and dispatch `response.create` messages
- `send_event(event)` — serialize `StreamingResponsesResponse` to JSON, send via WS
- `send_error(code, message, status)` — send error event in OpenAI format
- `cleanup()` — cancel in-flight generation, release resources

Main loop mirrors `RealtimeConnection.handle_connection()`:
1. Accept WebSocket
2. Start deadline timer (asyncio task)
3. Loop: `receive_text()` -> `json.loads()` -> `handle_event()`
4. On disconnect/timeout -> `cleanup()`

## Section 2: Request Handling & Event Flow

### response.create processing

1. Client sends: `{"type": "response.create", "model": "...", "input": [...]}`
2. Validate `type == "response.create"`, reject unknown types
3. Check `inflight == False`, reject concurrent requests
4. Set `inflight = True`
5. Construct `ResponsesRequest` from payload (force `stream = True`, strip `background`)
6. Resolve `previous_response_id` from connection-local cache
7. If `generate == false`: warmup — build context, cache, emit created + completed, return
8. Call `serving.create_responses(request)` — reuses existing pipeline
9. Iterate the `AsyncGenerator[StreamingResponsesResponse]`, send each event as WS JSON
10. On completion: cache response in `ConnectionContext`, set `inflight = False`
11. On error: send error event, evict cache, set `inflight = False`

The serving layer is reused as-is. The WebSocket layer replaces SSE transport only.

## Section 3: Server Integration & Connection Limits

### Connection tracking

Server-wide state in `app.state`:

```python
app.state.ws_responses_active_connections: int = 0
app.state.ws_responses_max_connections: int = 100
app.state.ws_responses_lock: asyncio.Lock
```

### WebSocket route

Added to `responses/api_router.py`:

```python
@router.websocket("/v1/responses")
async def create_responses_websocket(websocket: WebSocket):
    # Check connection limit
    # Increment counter
    # Create WebSocketResponsesConnection, run handle_connection()
    # Decrement counter in finally block
```

### CLI argument

`--max-websocket-connections` (default 100) added to server args.

No new task type — registered alongside existing HTTP routes when "generate" is in
supported_tasks.

## Section 4: Error Handling & Edge Cases

### Error event format

```json
{
  "type": "error",
  "status": 400,
  "error": {
    "code": "error_code_here",
    "message": "Human-readable description"
  }
}
```

### Error codes

| Code | Trigger | Action |
|---|---|---|
| `previous_response_not_found` | ID not in connection cache | Send error, keep connection |
| `websocket_connection_limit_reached` | Server at max connections | Send error, close connection |
| `concurrent_request` | New request while inflight | Send error, keep connection |
| `invalid_json` | Malformed JSON | Send error, keep connection |
| `unknown_event_type` | Type is not response.create | Send error, keep connection |
| `connection_expired` | 60-minute deadline reached | Warn at 55 min, close at 60 min |
| `processing_error` | Exception during generation | Send error, evict cache, keep connection |

### Cache eviction

Any generation error evicts `last_response_id` and `last_response` to prevent
stale/partial state from being used in continuation.

### Deadline enforcement

Background asyncio task:
- At 55 minutes: send `connection_expiring` warning event
- At 60 minutes: close WebSocket with code 1000

## Section 5: File Layout

### New files

| File | Purpose |
|---|---|
| `vllm/entrypoints/openai/responses/websocket.py` | `ConnectionContext`, `WebSocketResponsesConnection`, `uuid7()` |

### Modified files

| File | Changes |
|---|---|
| `vllm/entrypoints/openai/responses/api_router.py` | Add `@router.websocket("/v1/responses")` handler |
| `vllm/entrypoints/openai/api_server.py` | Initialize WS connection tracking state |
| CLI args | Add `--max-websocket-connections` argument |

### Tests

| File | Scope |
|---|---|
| `tests/entrypoints/openai/responses/test_websocket.py` | Unit: ConnectionContext state, uuid7 format, error codes, warmup, concurrency rejection |
| `tests/entrypoints/openai/responses/test_websocket_integration.py` | Integration: full WS flow, continuation, connection limits, deadline (mocked time) |

Uses Starlette's `TestClient.websocket_connect()` for WebSocket testing.

## Scope

### Included (MVP)

- WebSocket endpoint on `/v1/responses`
- `response.create` with streaming events
- `previous_response_id` continuation (connection-local cache)
- `generate: false` warmup mode
- Connection limit with `--max-websocket-connections`
- Single in-flight response per connection
- 60-minute connection lifetime with graceful close
- Error handling for all specified error codes

### Not included (future work)

- Persistence-backed hydrate when `store=true`
- Multiplexing
- `/responses/compact` integration
- Cancellation events
- Distributed connection cache
