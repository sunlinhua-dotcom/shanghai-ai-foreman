"""省豆包字数的两层包装（9/26 用户充了 10 万字：「在不影响体验的情况下帮我优化好，别浪费」）。

- TimeRoutedTTS：XZ_TTS_LOCAL_HOURS（默认 23:00–11:00）先用本地 CosyVoice，其余时间豆包优先。
- CachedTTS：豆包念过的短句（打招呼、「目录 X 个任务做好了」「有两条提醒」这类）存成 PCM，
  同一句再念直接放存档，不再花字数。只包在豆包外面：退到备胎念出来的不会冒充豆包存进去。
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import logging
import os
import re
from pathlib import Path
from zoneinfo import ZoneInfo

from ..audio import pcm16_to_wav

log = logging.getLogger("xz.tts")


def _window(spec: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", spec or "")
    if not m:
        return None
    a, b = int(m[1]) * 60 + int(m[2]), int(m[3]) * 60 + int(m[4])
    return None if a == b else (a, b)


class TimeRoutedTTS:
    def __init__(self, day, local, hours: str, tz: str):
        self.day, self.local = day, local
        self.window = _window(hours)
        self.tz = ZoneInfo(tz)
        self.sample_rate = day.sample_rate

    def use_local(self, now: dt.datetime | None = None) -> bool:
        if not self.window:
            return False
        now = now or dt.datetime.now(self.tz)
        m = now.hour * 60 + now.minute
        a, b = self.window
        return (a <= m < b) if a < b else (m >= a or m < b)

    def _pick(self):
        return self.local if self.use_local() else self.day

    @property
    def name(self) -> str:
        return self._pick().name

    async def synth_stream(self, text: str, prewarm=None):
        async for chunk in self._pick().synth_stream(text):
            yield chunk

    async def synth(self, text: str):
        return await self._pick().synth(text)

    async def health(self) -> dict:
        return {"route": "local" if self.use_local() else "day", **(await self._pick().health())}

    async def close(self) -> None:
        for tts in (self.day, self.local):
            await tts.close()


class CachedTTS:
    MAX_FILES = 5000

    def __init__(self, inner, cache_dir: str, max_chars: int = 40):
        self.inner = inner
        self.name = inner.name
        self.sample_rate = inner.sample_rate
        self.max_chars = max_chars
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        # 换音色 / 指令 / 语速，存档自动作废
        self._salt = "|".join(str(getattr(inner, k, "")) for k in ("voice", "instruction", "speech_rate", "sample_rate"))
        self.saved_chars = 0

    def _path(self, text: str) -> Path:
        return self.dir / (hashlib.sha1(f"{self._salt}|{text}".encode()).hexdigest() + ".pcm")

    def _store(self, path: Path, pcm: bytes) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pcm)
        tmp.replace(path)
        files = sorted(self.dir.glob("*.pcm"), key=lambda p: p.stat().st_mtime)
        for p in files[: max(0, len(files) - self.MAX_FILES)]:
            p.unlink(missing_ok=True)

    async def synth_stream(self, text: str, prewarm=None):
        path = self._path(text) if len(text) <= self.max_chars else None
        if path is not None and path.exists():
            pcm = await asyncio.to_thread(path.read_bytes)
            os.utime(path)
            self.saved_chars += len(text)
            log.info("豆包存档命中，省 %d 字（本次开机共省 %d 字）：%r", len(text), self.saved_chars, text[:20])
            step = self.sample_rate // 5 * 2  # 200 ms 一块
            for i in range(0, len(pcm), step):
                yield pcm[i:i + step]
            return
        chunks = []
        async for chunk in self.inner.synth_stream(text, prewarm):
            chunks.append(chunk)
            yield chunk
        if path is not None and chunks:
            await asyncio.to_thread(self._store, path, b"".join(chunks))

    async def synth(self, text: str):
        pcm = b"".join([c async for c in self.synth_stream(text)])
        return pcm16_to_wav(pcm, self.sample_rate), 0.0

    async def health(self) -> dict:
        return await self.inner.health()

    async def close(self) -> None:
        await self.inner.close()
