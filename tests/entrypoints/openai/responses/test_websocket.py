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
