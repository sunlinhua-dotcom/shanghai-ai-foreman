"""GX10 上的 CosyVoice3 客户端：用户本人的声音讲上海话，本地跑、不花钱（9/26 起）。

服务端是 tools/cosyvoice_server.py（GX10 systemd cosyvoice-sh，:8097）：
POST /tts {"text": …} → 流式 PCM16 单声道，采样率在响应头 X-Sample-Rate（24000）。
送出去之前先按上海话规矩改数字（wu_numbers：廿八、两百廿二……），只在这条路上改。
服务一次只念一句，这里也用锁排队，不并发打它。
"""
from __future__ import annotations

import asyncio
import logging
import time

import aiohttp

from ..audio import pcm16_to_wav, resample_pcm16
from ..wu_text import wu_numbers
from . import voices

log = logging.getLogger("xz.tts")


class CosyVoiceError(RuntimeError):
    def __init__(self, msg: str, transient: bool = True):
        super().__init__(msg)
        self.transient = transient  # False = 服务没开 / 连不上，FallbackTTS 10 分钟内直接跳过


class CosyVoiceTTS:
    name = "cosyvoice"

    def __init__(self, base_url: str, sample_rate: int = 24000, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.sample_rate = sample_rate
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def synth_stream(self, text: str, prewarm=None):
        say = wu_numbers(text)
        if say != text:
            log.info("上海话数字：%r", say)
        t0 = time.monotonic()
        got = 0
        async with self._lock:
            sess = await self._ensure_session()
            timeout = aiohttp.ClientTimeout(total=self.timeout, sock_connect=3)
            try:
                async with sess.post(f"{self.base_url}/tts", json={"text": say, "voice": voices.current()}, timeout=timeout) as resp:
                    if resp.status != 200:
                        raise CosyVoiceError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                    src = int(resp.headers.get("X-Sample-Rate", self.sample_rate))
                    carry = b""
                    async for chunk in resp.content.iter_any():
                        buf = carry + chunk
                        n = len(buf) & ~1  # PCM16 按两字节对齐
                        carry, buf = buf[n:], buf[:n]
                        if not buf:
                            continue
                        if not got:
                            log.info("CosyVoice 首块 %.2fs（%d 字）%r", time.monotonic() - t0, len(say), say[:20])
                        got += len(buf)
                        yield buf if src == self.sample_rate else resample_pcm16(buf, src, self.sample_rate)
            except aiohttp.ClientConnectionError as e:
                if got:
                    raise
                raise CosyVoiceError(f"连不上 GX10：{e}", transient=False) from e
            except asyncio.TimeoutError as e:
                if got:
                    raise
                raise CosyVoiceError("GX10 超时") from e
        if not got:
            raise CosyVoiceError("CosyVoice 没有返回音频")
        log.info("CosyVoice 整句 %.2fs（%d 字，%.1f s 音频）", time.monotonic() - t0, len(say),
                 got / 2 / self.sample_rate)

    async def synth(self, text: str) -> tuple[bytes, float]:
        t0 = time.monotonic()
        pcm = b"".join([c async for c in self.synth_stream(text)])
        return pcm16_to_wav(pcm, self.sample_rate), time.monotonic() - t0

    async def health(self) -> dict:
        sess = await self._ensure_session()
        async with sess.get(f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            resp.raise_for_status()
            return {"provider": "cosyvoice", **(await resp.json())}

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
