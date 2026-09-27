"""voxtts（VoxCPM2）客户端。

硬规矩（来自 ~/.claude/skills/identity）：
1. 按句合成，不要整段 —— 长文本明显变慢。
2. **同一时刻只发一个请求**：这里用进程级 asyncio.Lock 保证，
   跨会话也串行，绝不并发打 voxtts。
3. 服务会自己掉（ping 通但端口关）；这里只做超时与错误上报，不自动重启远端。
"""
from __future__ import annotations

import asyncio
import logging
import time

import aiohttp

log = logging.getLogger("xz.tts")

# 进程级：所有会话共用一把锁 —— 绝不并发打 voxtts
_GLOBAL_TTS_LOCK = asyncio.Lock()


class TTSError(RuntimeError):
    pass


class VoxTTS:
    name = "voxtts"

    def __init__(self, base_url: str, voice: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                trust_env=False,  # 不走系统代理
            )
        return self._session

    async def health(self) -> dict:
        sess = await self._ensure_session()
        async with sess.get(f"{self.base_url}/health") as resp:
            resp.raise_for_status()
            return await resp.json()

    async def synth(self, text: str) -> tuple[bytes, float]:
        """一句话 → (wav bytes, 往返秒数)。严格串行。"""
        if not text.strip():
            return b"", 0.0
        sess = await self._ensure_session()
        body = {"text": text, "voice": self.voice}
        async with _GLOBAL_TTS_LOCK:
            t0 = time.monotonic()
            try:
                async with sess.post(f"{self.base_url}/tts", json=body) as resp:
                    if resp.status != 200:
                        detail = (await resp.text())[:200]
                        raise TTSError(f"voxtts HTTP {resp.status}: {detail}")
                    data = await resp.read()
            except aiohttp.ClientError as e:
                raise TTSError(f"voxtts 请求失败: {e}") from e
            dt = time.monotonic() - t0
        log.info("TTS %.2fs  %d 字  %d 字节  %r", dt, len(text), len(data), text[:30])
        return data, dt

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
