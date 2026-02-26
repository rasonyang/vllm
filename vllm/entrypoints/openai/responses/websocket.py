# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json as json_module
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vllm.logger import init_logger

if TYPE_CHECKING:
    from fastapi import WebSocket

    from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

logger = init_logger(__name__)


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
    lo |= int.from_bytes(rand[2:], "big") & 0x3FFF_FFFF_FFFF_FFFF
    return f"{(hi | lo):032x}"


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


class WebSocketResponsesConnection:
    """Manages WebSocket lifecycle for Responses API WebSocket mode."""

    def __init__(self, websocket: "WebSocket",
                 serving: "OpenAIServingResponses"):
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
