# Responses API WebSocket Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OpenAI-compatible WebSocket mode to `/v1/responses` for persistent connections with connection-level response caching and incremental continuation.

**Architecture:** A `WebSocketResponsesConnection` class (mirroring the existing `RealtimeConnection` pattern) manages WebSocket lifecycle and delegates to the existing `OpenAIServingResponses` for inference. No new inference pipeline. The WebSocket route is registered alongside the existing HTTP POST route on the same `/v1/responses` path.

**Tech Stack:** FastAPI/Starlette built-in WebSocket support, asyncio, Pydantic. No new dependencies.

**Design doc:** `docs/plans/2026-02-26-responses-websocket-mode-design.md`

---

### Task 1: uuid7 helper function

**Files:**
- Create: `vllm/entrypoints/openai/responses/websocket.py`
- Test: `tests/entrypoints/openai/responses/test_websocket.py`

**Step 1: Write the failing test**

Create test file:

```python
# tests/entrypoints/openai/responses/test_websocket.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
import time


def test_uuid7_format():
    """uuid7 returns a 32-char hex string with version=7 and variant=10."""
    from vllm.entrypoints.openai.responses.websocket import uuid7

    result = uuid7()
    assert len(result) == 32
    assert re.fullmatch(r"[0-9a-f]{32}", result), f"Not hex: {result}"
    # Version nibble (bits 48-51) must be 7
    version = int(result[12], 16)
    assert version == 7, f"Version {version} != 7"
    # Variant bits (bits 64-65) must be 10 (value 8-b)
    variant = int(result[16], 16)
    assert variant in (8, 9, 0xa, 0xb), f"Variant nibble {variant:#x} invalid"


def test_uuid7_monotonic():
    """uuid7 values are time-ordered (monotonically increasing)."""
    from vllm.entrypoints.openai.responses.websocket import uuid7

    a = uuid7()
    time.sleep(0.002)  # 2ms to ensure different timestamp
    b = uuid7()
    assert a < b, f"{a} should be < {b}"


def test_uuid7_uniqueness():
    """Rapid uuid7 calls produce unique values."""
    from vllm.entrypoints.openai.responses.websocket import uuid7

    ids = [uuid7() for _ in range(1000)]
    assert len(set(ids)) == 1000
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py -v -x 2>&1 | head -30`
Expected: FAIL — `ModuleNotFoundError: No module named 'vllm.entrypoints.openai.responses.websocket'`

**Step 3: Write minimal implementation**

Create the websocket module with only uuid7:

```python
# vllm/entrypoints/openai/responses/websocket.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time


def uuid7() -> str:
    """Generate a UUID v7 string (time-ordered, RFC 9562).

    Returns a 32-character lowercase hex string.
    Layout: 48-bit unix-ms timestamp | 4-bit version (7) | 12-bit random
            | 2-bit variant (10) | 62-bit random
    """
    timestamp_ms = int(time.time() * 1000)
    rand = os.urandom(10)
    hi = (timestamp_ms & 0xFFFF_FFFF_FFFF) << 80
    hi |= 0x7 << 76  # version 7
    hi |= (int.from_bytes(rand[:2], "big") & 0x0FFF) << 64
    lo = 0x80 << 56  # variant 10
    lo |= int.from_bytes(rand[2:], "big") & 0x3F_FFFF_FFFF_FFFF
    return f"{(hi | lo):032x}"
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py -v -x 2>&1 | head -30`
Expected: 3 passed

**Step 5: Commit**

```bash
git add vllm/entrypoints/openai/responses/websocket.py tests/entrypoints/openai/responses/test_websocket.py
git commit -m "feat(responses): add uuid7 helper for WebSocket mode"
```

---

### Task 2: ConnectionContext dataclass

**Files:**
- Modify: `vllm/entrypoints/openai/responses/websocket.py`
- Modify: `tests/entrypoints/openai/responses/test_websocket.py`

**Step 1: Write the failing test**

Append to test file:

```python
def test_connection_context_defaults():
    """ConnectionContext initializes with correct defaults."""
    from vllm.entrypoints.openai.responses.websocket import ConnectionContext

    ctx = ConnectionContext(connection_id="ws-test123")
    assert ctx.connection_id == "ws-test123"
    assert ctx.last_response_id is None
    assert ctx.last_response is None
    assert ctx.inflight is False
    assert ctx.created_at > 0
    assert ConnectionContext.LIFETIME_SECONDS == 3600
    assert ConnectionContext.WARNING_SECONDS == 3300


def test_connection_context_is_expired():
    """ConnectionContext.is_expired checks against LIFETIME_SECONDS."""
    from unittest.mock import patch

    from vllm.entrypoints.openai.responses.websocket import ConnectionContext

    ctx = ConnectionContext(connection_id="ws-test")
    assert not ctx.is_expired()

    with patch("time.monotonic", return_value=ctx.created_at + 3601):
        assert ctx.is_expired()


def test_connection_context_should_warn():
    """ConnectionContext.should_warn checks against WARNING_SECONDS."""
    from unittest.mock import patch

    from vllm.entrypoints.openai.responses.websocket import ConnectionContext

    ctx = ConnectionContext(connection_id="ws-test")
    assert not ctx.should_warn()

    with patch("time.monotonic", return_value=ctx.created_at + 3301):
        assert ctx.should_warn()


def test_connection_context_evict_cache():
    """evict_cache clears last_response_id and last_response."""
    from vllm.entrypoints.openai.responses.websocket import ConnectionContext

    ctx = ConnectionContext(connection_id="ws-test")
    ctx.last_response_id = "resp_abc"
    ctx.last_response = "fake_response"  # type: ignore
    ctx.evict_cache()
    assert ctx.last_response_id is None
    assert ctx.last_response is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py::test_connection_context_defaults -v -x 2>&1 | head -20`
Expected: FAIL — `ImportError: cannot import name 'ConnectionContext'`

**Step 3: Write minimal implementation**

Add to `vllm/entrypoints/openai/responses/websocket.py`:

```python
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from vllm.entrypoints.openai.responses.protocol import ResponsesResponse


@dataclass
class ConnectionContext:
    """Per-connection state for WebSocket Responses mode."""

    connection_id: str
    last_response_id: str | None = None
    last_response: "ResponsesResponse | None" = None
    inflight: bool = False
    created_at: float = field(default_factory=time.monotonic)

    LIFETIME_SECONDS: ClassVar[int] = 3600   # 60 minutes
    WARNING_SECONDS: ClassVar[int] = 3300    # 55 minutes

    def is_expired(self) -> bool:
        return time.monotonic() - self.created_at >= self.LIFETIME_SECONDS

    def should_warn(self) -> bool:
        return time.monotonic() - self.created_at >= self.WARNING_SECONDS

    def evict_cache(self) -> None:
        self.last_response_id = None
        self.last_response = None
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py -v -x -k "connection_context" 2>&1 | head -20`
Expected: 4 passed

**Step 5: Commit**

```bash
git add vllm/entrypoints/openai/responses/websocket.py tests/entrypoints/openai/responses/test_websocket.py
git commit -m "feat(responses): add ConnectionContext dataclass for WebSocket mode"
```

---

### Task 3: WebSocketResponsesConnection — error/event sending

**Files:**
- Modify: `vllm/entrypoints/openai/responses/websocket.py`
- Modify: `tests/entrypoints/openai/responses/test_websocket.py`

**Step 1: Write the failing test**

Append to test file:

```python
import json

import pytest


@pytest.mark.asyncio
async def test_send_error_format():
    """send_error sends JSON in the OpenAI error event format."""
    from unittest.mock import AsyncMock

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    await conn.send_error("previous_response_not_found",
                          "Response not found", 400)

    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["type"] == "error"
    assert payload["status"] == 400
    assert payload["error"]["code"] == "previous_response_not_found"
    assert payload["error"]["message"] == "Response not found"


@pytest.mark.asyncio
async def test_send_event_serializes_pydantic():
    """send_event serializes a Pydantic model and sends as text."""
    from unittest.mock import AsyncMock

    from pydantic import BaseModel

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    class FakeEvent(BaseModel):
        type: str = "response.created"
        data: str = "hello"

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    await conn.send_event(FakeEvent())

    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["type"] == "response.created"
    assert payload["data"] == "hello"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py::test_send_error_format -v -x 2>&1 | head -20`
Expected: FAIL — `ImportError: cannot import name 'WebSocketResponsesConnection'`

**Step 3: Write minimal implementation**

Add to `vllm/entrypoints/openai/responses/websocket.py`:

```python
import json as json_module

from fastapi import WebSocket
from pydantic import BaseModel

from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.logger import init_logger

logger = init_logger(__name__)


class WebSocketResponsesConnection:
    """Manages WebSocket lifecycle for Responses API WebSocket mode.

    Mirrors the RealtimeConnection pattern from
    vllm/entrypoints/openai/realtime/connection.py.
    """

    def __init__(self, websocket: WebSocket,
                 serving: OpenAIServingResponses):
        self.websocket = websocket
        self.serving = serving
        self.ctx = ConnectionContext(connection_id=f"ws-{uuid7()}")
        self._is_connected = False
        self._generation_task: asyncio.Task | None = None
        self._deadline_task: asyncio.Task | None = None

    async def send_event(self, event: BaseModel) -> None:
        """Send a Pydantic event as JSON text over WebSocket."""
        await self.websocket.send_text(event.model_dump_json())

    async def send_error(self, code: str, message: str,
                         status: int = 400) -> None:
        """Send an error event in the OpenAI WebSocket error format."""
        payload = json_module.dumps({
            "type": "error",
            "status": status,
            "error": {
                "code": code,
                "message": message,
            },
        })
        await self.websocket.send_text(payload)
```

Note: Also add `import asyncio` to the top of the file.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py -v -x -k "send_error or send_event" 2>&1 | head -20`
Expected: 2 passed

**Step 5: Commit**

```bash
git add vllm/entrypoints/openai/responses/websocket.py tests/entrypoints/openai/responses/test_websocket.py
git commit -m "feat(responses): add WebSocketResponsesConnection with send_error/send_event"
```

---

### Task 4: WebSocketResponsesConnection — handle_event logic

**Files:**
- Modify: `vllm/entrypoints/openai/responses/websocket.py`
- Modify: `tests/entrypoints/openai/responses/test_websocket.py`

**Step 1: Write the failing tests**

Append to test file:

```python
@pytest.mark.asyncio
async def test_handle_event_unknown_type():
    """Unknown event type sends error, keeps connection open."""
    from unittest.mock import AsyncMock

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    await conn.handle_event({"type": "bogus.event"})

    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["error"]["code"] == "unknown_event_type"


@pytest.mark.asyncio
async def test_handle_event_concurrent_request_rejected():
    """Second response.create while inflight returns concurrent_request error."""
    from unittest.mock import AsyncMock

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)
    conn.ctx.inflight = True

    await conn.handle_event({
        "type": "response.create",
        "model": "test-model",
        "input": "hello",
    })

    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["error"]["code"] == "concurrent_request"


@pytest.mark.asyncio
async def test_handle_event_previous_response_not_found():
    """previous_response_id not in cache returns error."""
    from unittest.mock import AsyncMock

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    await conn.handle_event({
        "type": "response.create",
        "model": "test-model",
        "input": "hello",
        "previous_response_id": "resp_nonexistent",
    })

    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["error"]["code"] == "previous_response_not_found"
    assert not conn.ctx.inflight
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py::test_handle_event_unknown_type -v -x 2>&1 | head -20`
Expected: FAIL — `AttributeError: 'WebSocketResponsesConnection' object has no attribute 'handle_event'`

**Step 3: Write minimal implementation**

Add to the `WebSocketResponsesConnection` class in `websocket.py`:

```python
    async def handle_event(self, event: dict) -> None:
        """Parse and dispatch incoming WebSocket events."""
        event_type = event.get("type")

        if event_type != "response.create":
            await self.send_error(
                "unknown_event_type",
                f"Unknown event type: {event_type}",
            )
            return

        # Reject concurrent requests
        if self.ctx.inflight:
            await self.send_error(
                "concurrent_request",
                "A response is already being generated on this connection",
                409,
            )
            return

        # Resolve previous_response_id from connection-local cache
        prev_id = event.get("previous_response_id")
        if prev_id is not None and prev_id != self.ctx.last_response_id:
            await self.send_error(
                "previous_response_not_found",
                f"Response '{prev_id}' not found in connection cache",
                404,
            )
            return

        self.ctx.inflight = True
        try:
            await self._process_response_create(event)
        except Exception as e:
            logger.exception("Error processing response.create: %s", e)
            await self.send_error("processing_error", str(e), 500)
            self.ctx.evict_cache()
        finally:
            self.ctx.inflight = False

    async def _process_response_create(self, event: dict) -> None:
        """Process a response.create event — build request, run inference, stream events."""
        # Strip WebSocket-only fields, force streaming
        payload = {k: v for k, v in event.items() if k != "type"}
        payload.pop("generate", None)  # handled separately below
        payload["stream"] = True
        payload.pop("background", None)

        generate = event.get("generate", True)

        # Build the ResponsesRequest
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
        request = ResponsesRequest(**payload)

        if not generate:
            # Warmup mode: cache context, skip inference
            # Emit response.created + response.completed with no output
            from vllm.entrypoints.openai.responses.protocol import (
                ResponseCompletedEvent,
                ResponseCreatedEvent,
                ResponsesResponse,
            )
            response = ResponsesResponse(
                id=request.request_id,
                created_at=int(time.time()),
                model=request.model or "",
                output=[],
                status="completed",
            )
            await self.send_event(ResponseCreatedEvent(response=response))
            await self.send_event(ResponseCompletedEvent(response=response))
            self.ctx.last_response_id = response.id
            self.ctx.last_response = response
            return

        # Call existing serving pipeline (always streaming)
        result = await self.serving.create_responses(request)

        if isinstance(result, ErrorResponse):
            await self.send_error("processing_error", result.error.message,
                                  result.error.code)
            self.ctx.evict_cache()
            return

        # Non-streaming result (shouldn't happen since we forced stream=True)
        from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
        if isinstance(result, ResponsesResponse):
            from vllm.entrypoints.openai.responses.protocol import (
                ResponseCompletedEvent,
                ResponseCreatedEvent,
            )
            await self.send_event(ResponseCreatedEvent(response=result))
            await self.send_event(ResponseCompletedEvent(response=result))
            self.ctx.last_response_id = result.id
            self.ctx.last_response = result
            return

        # Streaming: iterate events, forward over WebSocket
        last_response = None
        async for event_obj in result:
            await self.send_event(event_obj)
            # Capture the response from completed event for caching
            if hasattr(event_obj, "type") and event_obj.type == "response.completed":
                last_response = getattr(event_obj, "response", None)

            if not self._is_connected:
                break

        if last_response is not None:
            self.ctx.last_response_id = last_response.id
            self.ctx.last_response = last_response
```

Note: Add `from vllm.entrypoints.openai.engine.protocol import ErrorResponse` to the imports.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py -v -x -k "handle_event" 2>&1 | head -25`
Expected: 3 passed

**Step 5: Commit**

```bash
git add vllm/entrypoints/openai/responses/websocket.py tests/entrypoints/openai/responses/test_websocket.py
git commit -m "feat(responses): add handle_event with validation, cache lookup, and streaming"
```

---

### Task 5: WebSocketResponsesConnection — handle_connection lifecycle & deadline

**Files:**
- Modify: `vllm/entrypoints/openai/responses/websocket.py`
- Modify: `tests/entrypoints/openai/responses/test_websocket.py`

**Step 1: Write the failing tests**

Append to test file:

```python
@pytest.mark.asyncio
async def test_handle_connection_accept_and_receive_loop():
    """handle_connection accepts, processes messages, handles disconnect."""
    from unittest.mock import AsyncMock

    from starlette.websockets import WebSocketDisconnect

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    # Simulate: one valid message, then disconnect
    ws.receive_text.side_effect = [
        '{"type": "response.create", "model": "m", "input": "hi"}',
        WebSocketDisconnect(),
    ]
    # Mock the serving to return an error (simplest path)
    serving.create_responses.side_effect = Exception("mock engine down")

    await conn.handle_connection()

    ws.accept.assert_called_once()
    assert not conn._is_connected


@pytest.mark.asyncio
async def test_handle_connection_invalid_json():
    """Invalid JSON sends error but keeps connection open."""
    from unittest.mock import AsyncMock

    from starlette.websockets import WebSocketDisconnect

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    ws.receive_text.side_effect = [
        "not valid json{{{",
        WebSocketDisconnect(),
    ]

    await conn.handle_connection()

    # First call: error event for invalid JSON. Possibly more calls.
    first_call = ws.send_text.call_args_list[0]
    payload = json.loads(first_call[0][0])
    assert payload["error"]["code"] == "invalid_json"


@pytest.mark.asyncio
async def test_cleanup_cancels_generation_task():
    """cleanup cancels in-flight generation task."""
    from unittest.mock import AsyncMock, MagicMock

    from vllm.entrypoints.openai.responses.websocket import (
        WebSocketResponsesConnection,
    )

    ws = AsyncMock()
    serving = AsyncMock()
    conn = WebSocketResponsesConnection(ws, serving)

    fake_task = MagicMock()
    fake_task.done.return_value = False
    conn._generation_task = fake_task

    await conn.cleanup()

    fake_task.cancel.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py::test_handle_connection_accept_and_receive_loop -v -x 2>&1 | head -20`
Expected: FAIL — `AttributeError: 'WebSocketResponsesConnection' object has no attribute 'handle_connection'`

**Step 3: Write minimal implementation**

Add to the `WebSocketResponsesConnection` class:

```python
    async def handle_connection(self) -> None:
        """Main WebSocket connection loop.

        Pattern mirrors RealtimeConnection.handle_connection() from
        vllm/entrypoints/openai/realtime/connection.py.
        """
        await self.websocket.accept()
        logger.debug("WebSocket responses connection accepted: %s",
                      self.ctx.connection_id)
        self._is_connected = True

        # Start deadline enforcement task
        self._deadline_task = asyncio.create_task(
            self._enforce_deadline()
        )

        try:
            while True:
                message = await self.websocket.receive_text()
                try:
                    event = json_module.loads(message)
                    await self.handle_event(event)
                except json_module.JSONDecodeError:
                    await self.send_error("invalid_json", "Invalid JSON")
                except Exception as e:
                    logger.exception("Error handling event: %s", e)
                    await self.send_error("processing_error", str(e), 500)
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s",
                         self.ctx.connection_id)
            self._is_connected = False
        except Exception as e:
            logger.exception("Unexpected error in connection %s: %s",
                             self.ctx.connection_id, e)
        finally:
            await self.cleanup()

    async def _enforce_deadline(self) -> None:
        """Background task: warn at WARNING_SECONDS, close at LIFETIME_SECONDS."""
        try:
            warn_delay = (self.ctx.WARNING_SECONDS
                         - (time.monotonic() - self.ctx.created_at))
            if warn_delay > 0:
                await asyncio.sleep(warn_delay)
            if self._is_connected:
                await self.send_error(
                    "connection_expiring",
                    "Connection will close in 5 minutes",
                )

            close_delay = (self.ctx.LIFETIME_SECONDS
                          - (time.monotonic() - self.ctx.created_at))
            if close_delay > 0:
                await asyncio.sleep(close_delay)
            if self._is_connected:
                self._is_connected = False
                await self.websocket.close(
                    code=1000,
                    reason="Connection lifetime exceeded",
                )
        except asyncio.CancelledError:
            pass

    async def cleanup(self) -> None:
        """Release resources on disconnect."""
        # Cancel deadline task
        if self._deadline_task and not self._deadline_task.done():
            self._deadline_task.cancel()

        # Cancel generation task
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()

        logger.debug("Connection cleanup complete: %s",
                     self.ctx.connection_id)
```

Note: Add `from starlette.websockets import WebSocketDisconnect` to imports.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py -v -x -k "handle_connection or cleanup" 2>&1 | head -25`
Expected: 3 passed

**Step 5: Commit**

```bash
git add vllm/entrypoints/openai/responses/websocket.py tests/entrypoints/openai/responses/test_websocket.py
git commit -m "feat(responses): add handle_connection lifecycle, deadline, and cleanup"
```

---

### Task 6: CLI arg --max-websocket-connections

**Files:**
- Modify: `vllm/entrypoints/openai/cli_args.py:279` (add field after `enable_offline_docs`)
- Modify: `tests/entrypoints/openai/responses/test_websocket.py`

**Step 1: Write the failing test**

Append to test file:

```python
def test_cli_arg_max_websocket_connections_default():
    """--max-websocket-connections defaults to 100."""
    from vllm.entrypoints.openai.cli_args import FrontendArgs

    assert hasattr(FrontendArgs, "max_websocket_connections")
    # Instantiate with minimal required args to check default
    # FrontendArgs is a @config dataclass; check class attribute default
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(FrontendArgs)}
    field = fields["max_websocket_connections"]
    assert field.default == 100
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py::test_cli_arg_max_websocket_connections_default -v -x 2>&1 | head -20`
Expected: FAIL — `AssertionError` (field doesn't exist yet)

**Step 3: Write minimal implementation**

In `vllm/entrypoints/openai/cli_args.py`, add after line 279 (`enable_offline_docs` field):

```python
    max_websocket_connections: int = 100
    """Maximum number of concurrent WebSocket connections for the
    Responses API WebSocket mode. Default: 100."""
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py::test_cli_arg_max_websocket_connections_default -v -x 2>&1 | head -20`
Expected: 1 passed

**Step 5: Commit**

```bash
git add vllm/entrypoints/openai/cli_args.py tests/entrypoints/openai/responses/test_websocket.py
git commit -m "feat(responses): add --max-websocket-connections CLI arg (default 100)"
```

---

### Task 7: WebSocket route and server state initialization

**Files:**
- Modify: `vllm/entrypoints/openai/responses/api_router.py:140` (add websocket route before `attach_router`)
- Modify: `vllm/entrypoints/openai/api_server.py:378` (add WS state init after `enable_server_load_tracking`)
- Modify: `tests/entrypoints/openai/responses/test_websocket.py`

**Step 1: Write the failing test**

Append to test file:

```python
@pytest.mark.asyncio
async def test_websocket_route_rejects_when_limit_reached():
    """WebSocket route sends connection_limit_reached when at max."""
    from unittest.mock import AsyncMock, MagicMock, PropertyMock

    from vllm.entrypoints.openai.responses.api_router import (
        create_responses_websocket,
    )

    ws = AsyncMock()
    # Simulate app state at limit
    app_state = MagicMock()
    app_state.openai_serving_responses = MagicMock()
    app_state.ws_responses_active_connections = 5
    app_state.ws_responses_max_connections = 5
    import asyncio
    app_state.ws_responses_lock = asyncio.Lock()

    app = MagicMock()
    app.state = app_state
    type(ws).app = PropertyMock(return_value=app)

    await create_responses_websocket(ws)

    # Should accept, send error, then close
    ws.accept.assert_called_once()
    ws.send_text.assert_called_once()
    payload = json.loads(ws.send_text.call_args[0][0])
    assert payload["error"]["code"] == "websocket_connection_limit_reached"
    ws.close.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py::test_websocket_route_rejects_when_limit_reached -v -x 2>&1 | head -20`
Expected: FAIL — `ImportError: cannot import name 'create_responses_websocket'`

**Step 3: Write implementation**

**3a. Modify `vllm/entrypoints/openai/responses/api_router.py`:**

Add imports at the top (after existing imports, around line 8):

```python
import json

from fastapi import APIRouter, Depends, FastAPI, Request, WebSocket
```

Add the WebSocket route before `attach_router` (before line 140):

```python
@router.websocket("/v1/responses")
async def create_responses_websocket(websocket: WebSocket):
    """WebSocket endpoint for Responses API WebSocket mode.

    Protocol:
    1. Client connects to ws://host/v1/responses
    2. Client sends {"type": "response.create", ...} messages
    3. Server streams response events as JSON text frames
    4. Client can use previous_response_id for continuation
    """
    app = websocket.app
    serving = app.state.openai_serving_responses

    if serving is None:
        await websocket.close(code=1008,
                              reason="Responses API not supported")
        return

    # Connection limit check
    async with app.state.ws_responses_lock:
        if (app.state.ws_responses_active_connections
                >= app.state.ws_responses_max_connections):
            await websocket.accept()
            await websocket.send_text(json.dumps({
                "type": "error",
                "status": 429,
                "error": {
                    "code": "websocket_connection_limit_reached",
                    "message": "Maximum WebSocket connections reached",
                },
            }))
            await websocket.close()
            return
        app.state.ws_responses_active_connections += 1

    try:
        from vllm.entrypoints.openai.responses.websocket import (
            WebSocketResponsesConnection,
        )

        connection = WebSocketResponsesConnection(websocket, serving)
        await connection.handle_connection()
    finally:
        async with app.state.ws_responses_lock:
            app.state.ws_responses_active_connections -= 1
```

**3b. Modify `vllm/entrypoints/openai/api_server.py`:**

After line 378 (`state.enable_server_load_tracking = args.enable_server_load_tracking`), add:

```python
    # WebSocket Responses mode connection tracking
    if "generate" in supported_tasks:
        state.ws_responses_active_connections = 0
        state.ws_responses_max_connections = getattr(
            args, "max_websocket_connections", 100)
        state.ws_responses_lock = asyncio.Lock()
```

Also add `import asyncio` to the imports at the top of `api_server.py` (if not already present).

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py::test_websocket_route_rejects_when_limit_reached -v -x 2>&1 | head -20`
Expected: 1 passed

**Step 5: Commit**

```bash
git add vllm/entrypoints/openai/responses/api_router.py vllm/entrypoints/openai/api_server.py tests/entrypoints/openai/responses/test_websocket.py
git commit -m "feat(responses): add WebSocket route with connection limit and server state init"
```

---

### Task 8: Integration test — full WebSocket flow

**Files:**
- Create: `tests/entrypoints/openai/responses/test_websocket_integration.py`

This test uses Starlette's TestClient to test the full WebSocket flow without a real model engine. It mocks `OpenAIServingResponses.create_responses` to return a known stream of events.

**Step 1: Write the integration test**

```python
# tests/entrypoints/openai/responses/test_websocket_integration.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from vllm.entrypoints.openai.responses.api_router import router


def _make_app(max_connections: int = 100) -> FastAPI:
    """Create a minimal FastAPI app with the responses router."""
    app = FastAPI()
    app.include_router(router)

    # Mock the serving state
    app.state.openai_serving_responses = MagicMock()
    app.state.openai_serving_tokenization = MagicMock()
    app.state.ws_responses_active_connections = 0
    app.state.ws_responses_max_connections = max_connections
    app.state.ws_responses_lock = asyncio.Lock()

    return app


def test_websocket_connect_and_receive_events():
    """Full flow: connect, send response.create, receive streaming events."""
    app = _make_app()

    # Mock create_responses to return an async generator of fake events
    async def fake_stream(request, raw_request=None):
        from pydantic import BaseModel

        class FakeCreated(BaseModel):
            type: str = "response.created"
            response: dict = {"id": "resp_test1", "status": "in_progress"}

        class FakeDelta(BaseModel):
            type: str = "response.output_text.delta"
            delta: str = "Hello"

        class FakeCompleted(BaseModel):
            type: str = "response.completed"
            response: dict = {"id": "resp_test1", "status": "completed"}

        async def gen():
            yield FakeCreated()
            yield FakeDelta()
            yield FakeCompleted()

        return gen()

    app.state.openai_serving_responses.create_responses = AsyncMock(
        side_effect=fake_stream
    )

    client = TestClient(app)
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_text(json.dumps({
            "type": "response.create",
            "model": "test-model",
            "input": "Say hello",
        }))

        events = []
        # Read all events (created, delta, completed)
        for _ in range(3):
            data = ws.receive_text()
            events.append(json.loads(data))

        assert events[0]["type"] == "response.created"
        assert events[1]["type"] == "response.output_text.delta"
        assert events[2]["type"] == "response.completed"


def test_websocket_connection_limit():
    """Connection limit is enforced."""
    app = _make_app(max_connections=1)

    # Simulate one active connection
    app.state.ws_responses_active_connections = 1

    client = TestClient(app)
    with client.websocket_connect("/v1/responses") as ws:
        data = ws.receive_text()
        payload = json.loads(data)
        assert payload["error"]["code"] == "websocket_connection_limit_reached"


def test_websocket_invalid_json():
    """Invalid JSON produces error but connection stays open."""
    app = _make_app()

    client = TestClient(app)
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_text("not json{{{")
        data = ws.receive_text()
        payload = json.loads(data)
        assert payload["error"]["code"] == "invalid_json"


def test_websocket_unknown_event_type():
    """Unknown event type produces error."""
    app = _make_app()

    client = TestClient(app)
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_text(json.dumps({"type": "session.update"}))
        data = ws.receive_text()
        payload = json.loads(data)
        assert payload["error"]["code"] == "unknown_event_type"
```

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket_integration.py -v -x 2>&1 | head -30`
Expected: 4 passed (these tests exercise the already-implemented code from Tasks 1-7)

Note: If any fail, debug and fix. The integration test is the final verification of all previous tasks working together.

**Step 3: Commit**

```bash
git add tests/entrypoints/openai/responses/test_websocket_integration.py
git commit -m "test(responses): add WebSocket mode integration tests"
```

---

### Task 9: Run full test suite and fix any issues

**Files:**
- Potentially modify any file from Tasks 1-8

**Step 1: Run all WebSocket tests**

Run: `python -m pytest tests/entrypoints/openai/responses/test_websocket.py tests/entrypoints/openai/responses/test_websocket_integration.py -v 2>&1 | tail -30`
Expected: All tests pass

**Step 2: Run existing responses tests to check for regressions**

Run: `python -m pytest tests/entrypoints/openai/responses/ -v --ignore=tests/entrypoints/openai/responses/test_simple.py --ignore=tests/entrypoints/openai/responses/test_harmony.py --ignore=tests/entrypoints/openai/responses/test_mcp_tools.py 2>&1 | tail -30`

Note: `test_simple.py`, `test_harmony.py`, and `test_mcp_tools.py` require a real model engine via `RemoteOpenAIServer`, so skip them in unit test runs. The tests we CAN run are `test_errors.py`, `test_function_call_parsing.py`, `test_parsable_context.py`, `test_sampling_params.py`.

Expected: All pass with no regressions

**Step 3: Run linting**

Run: `python -m ruff check vllm/entrypoints/openai/responses/websocket.py vllm/entrypoints/openai/responses/api_router.py vllm/entrypoints/openai/api_server.py vllm/entrypoints/openai/cli_args.py 2>&1 | head -20`

Fix any lint errors.

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix(responses): address lint and test issues in WebSocket mode"
```

---

## Summary of all files touched

| File | Action | Task |
|---|---|---|
| `vllm/entrypoints/openai/responses/websocket.py` | Create | 1, 2, 3, 4, 5 |
| `vllm/entrypoints/openai/responses/api_router.py` | Modify | 7 |
| `vllm/entrypoints/openai/api_server.py` | Modify | 7 |
| `vllm/entrypoints/openai/cli_args.py` | Modify | 6 |
| `tests/entrypoints/openai/responses/test_websocket.py` | Create | 1, 2, 3, 4, 5, 6, 7 |
| `tests/entrypoints/openai/responses/test_websocket_integration.py` | Create | 8 |
