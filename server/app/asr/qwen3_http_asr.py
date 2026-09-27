"""GX10 上的 Qwen3-ASR-1.7B（llama.cpp，OpenAI 兼容 /v1/audio/transcriptions）。整段识别，不流式。

2026-09-23 实测（24 句带混响和噪声的标准句）：错字率 0.4%，本机 sherpa 是 3.1–3.8%，
中英混说（「GPT 六和 Claude」被 sherpa 听成「PPT6合盒」）差得最多；每句 0.11–0.18 s。
GX10 连不上就回退本机 sherpa，盒子不至于变哑巴。
"""
from __future__ import annotations

import io
import logging
import re
import time
import wave

import aiohttp

from . import ASRProvider

log = logging.getLogger("xz.asr")

# 「呜呜呜呜……」这种复读是模型卡在一个字上，多半是电视里的歌（实测一段拖了 84 s）
_LOOP = re.compile(r"(.)\1{7,}")


class Qwen3HttpASR(ASRProvider):
    name = "qwen3-asr@gx10"
    streaming = False

    def __init__(self, url: str, timeout: float, fallback: ASRProvider | None = None) -> None:
        self.url = url
        self.timeout = timeout
        self.fallback = fallback
        self._session: aiohttp.ClientSession | None = None

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        t0 = time.monotonic()
        dur_ms = len(pcm16) / 2 / sample_rate * 1000
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm16)
        form = aiohttp.FormData()
        form.add_field("file", buf.getvalue(), filename="u.wav", content_type="audio/wav")
        form.add_field("model", "qwen3-asr")
        form.add_field("language", "zh")
        try:
            sess = await self._http()
            async with sess.post(self.url, data=form) as r:
                r.raise_for_status()
                text = ((await r.json(content_type=None)).get("text") or "").strip()
        except TimeoutError:
            # 超时几乎都是模型在复读（电视里的歌），当噪声丢掉，别再让 sherpa 猜一遍
            log.warning("GX10 识别超过 %.0fs，当噪声丢掉", self.timeout)
            return ""
        except Exception as e:  # noqa: BLE001 - 连不上就回退
            log.warning("GX10 识别失败（%s），这句改用本机 sherpa", type(e).__name__)
            return await self.fallback.transcribe(pcm16, sample_rate) if self.fallback else ""
        log.info("ASR[gx10] %.0fms 音频 → %.2fs → %r", dur_ms, time.monotonic() - t0, text)
        if _LOOP.search(text):
            log.info("识别结果在复读（%r），当噪声丢掉", text[:16])
            return ""
        return text

    async def warmup(self) -> None:
        if self.fallback is not None and getattr(self.fallback, "warmup", None):
            await self.fallback.warmup()
        t0 = time.monotonic()
        try:
            sess = await self._http()
            async with sess.get(self.url.split("/v1/")[0] + "/health") as r:
                ok = r.status
        except Exception as e:  # noqa: BLE001
            log.warning("GX10 识别（%s）连不上：%s——先用本机 sherpa 顶着", self.url, type(e).__name__)
            return
        log.info("GX10 识别就绪（%s，health %s，%.2fs）", self.url, ok, time.monotonic() - t0)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
        if self.fallback is not None and getattr(self.fallback, "close", None):
            await self.fallback.close()
