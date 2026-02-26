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
