# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
import sys
import time
import types

import pytest

# Mock torch and heavy vllm dependencies so this unit-test module can run
# on machines without torch / CUDA.  Must happen before any ``from vllm …``
# import because ``vllm/__init__.py`` transitively imports torch via
# ``vllm.env_override``.
# ---------------------------------------------------------------------------
# Lightweight shim so the test module can be collected and run on machines
# that do **not** have torch / CUDA / numpy installed.
#
# ``vllm/__init__.py`` does ``import vllm.env_override`` which in turn does
# ``import torch`` at module level.  We short-circuit that by inserting a
# no-op ``vllm.env_override`` into ``sys.modules`` *before* the real
# ``vllm`` package is imported.
# ---------------------------------------------------------------------------
if "torch" not in sys.modules:
    # Insert a no-op env_override so vllm/__init__.py skips torch
    sys.modules["vllm.env_override"] = types.ModuleType("vllm.env_override")


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
