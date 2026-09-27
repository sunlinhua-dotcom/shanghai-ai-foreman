"""空 ASR：不识别，只回固定文本。用于不装模型时跑通链路。"""
from __future__ import annotations

import os


class NullASR:
    name = "null"

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        return os.environ.get("XZ_ASR_NULL_TEXT", "")

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None
