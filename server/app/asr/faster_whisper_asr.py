"""本机 CPU faster-whisper。中文优先，模型大小可配。"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

log = logging.getLogger("xz.asr")


class FasterWhisperASR:
    name = "faster-whisper"

    def __init__(
        self,
        model: str = "small",
        language: str = "zh",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_name = model
        self.language = language or None
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = asyncio.Lock()

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            t0 = time.time()
            log.info("加载 faster-whisper 模型 %s (%s/%s)…", self.model_name, self.device, self.compute_type)
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
            log.info("faster-whisper 就绪，耗时 %.1fs", time.time() - t0)
        return self._model

    def _transcribe_sync(self, pcm16: bytes, sample_rate: int) -> str:
        model = self._ensure_model()
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        if sample_rate != 16000:
            from ..audio import resample_pcm16

            audio = (
                np.frombuffer(resample_pcm16(pcm16, sample_rate, 16000), dtype="<i2").astype(
                    np.float32
                )
                / 32768.0
            )
        segments, _info = model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt="以下是普通话的句子。",
        )
        return "".join(seg.text for seg in segments).strip()

    async def warmup(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._ensure_model)
            # 再跑一次真转写（1s 静音）：只 load 模型盖不住首轮 lazy-init，
            # 服务重启后的第一问会因此多花 1–2s
            await asyncio.to_thread(self._transcribe_sync, b"\x00" * 32000, 16000)

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        if not pcm16:
            return ""
        async with self._lock:  # CPU 推理，串行即可
            t0 = time.time()
            text = await asyncio.to_thread(self._transcribe_sync, pcm16, sample_rate)
        dur_ms = len(pcm16) / 2 / sample_rate * 1000
        log.info("ASR %.0fms 音频 → %.2fs → %r", dur_ms, time.time() - t0, text)
        return text

    async def close(self) -> None:
        self._model = None
