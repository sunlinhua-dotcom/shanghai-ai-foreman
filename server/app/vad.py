"""语音端点检测。

设备自己也会发 listen stop（手动/自动模式），所以 VAD 只是"另一条"断句依据，
两条路都要能收尾 —— 见 session.py 的 _finish_utterance()。

优先用 webrtcvad（16k/20ms 帧），装不上就退到能量门限。
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - 取决于安装环境
    import webrtcvad

    _HAVE_WEBRTCVAD = True
except ImportError:  # pragma: no cover
    webrtcvad = None
    _HAVE_WEBRTCVAD = False


class VadSegmenter:
    """喂 16 kHz int16 PCM，回答"这一段该收尾了吗"。

    用法：
        seg.feed(pcm)  -> bool，True 表示检测到句尾（静音够久）
        seg.speech_ms  -> 累计判定为语音的毫秒数
    """

    FRAME_MS = 20

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_ms: int = 800,
        min_speech_ms: int = 300,
        max_utterance_ms: int = 15000,
        aggressiveness: int = 2,
    ):
        self.sample_rate = sample_rate
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_utterance_ms = max_utterance_ms
        self.frame_bytes = sample_rate * self.FRAME_MS // 1000 * 2
        self._vad = None
        if _HAVE_WEBRTCVAD and sample_rate in (8000, 16000, 32000, 48000):
            self._vad = webrtcvad.Vad(max(0, min(3, aggressiveness)))
        self._noise_floor = 250.0  # 能量兜底用的自适应底噪
        self._tail = b""
        self.reset()

    @property
    def backend(self) -> str:
        return "webrtcvad" if self._vad is not None else "energy"

    def reset(self) -> None:
        self._tail = b""
        self.speech_ms = 0
        self.silence_after_speech_ms = 0
        self.total_ms = 0
        self.triggered = False

    def _is_speech(self, frame: bytes) -> bool:
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame, self.sample_rate)
            except Exception:
                pass
        x = np.frombuffer(frame, dtype="<i2").astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
        if rms < self._noise_floor:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
        return rms > max(300.0, self._noise_floor * 3.0)

    def feed(self, pcm: bytes) -> bool:
        """返回 True = 该收尾了（静音够久，或超过最长时长）。"""
        buf = self._tail + pcm
        n = len(buf) // self.frame_bytes
        for i in range(n):
            frame = buf[i * self.frame_bytes : (i + 1) * self.frame_bytes]
            self.total_ms += self.FRAME_MS
            if self._is_speech(frame):
                self.speech_ms += self.FRAME_MS
                self.silence_after_speech_ms = 0
                self.triggered = True
            elif self.triggered:
                self.silence_after_speech_ms += self.FRAME_MS
        self._tail = buf[n * self.frame_bytes :]

        if self.total_ms >= self.max_utterance_ms and self.speech_ms >= self.min_speech_ms:
            return True
        return (
            self.triggered
            and self.speech_ms >= self.min_speech_ms
            and self.silence_after_speech_ms >= self.silence_ms
        )

    def has_speech(self) -> bool:
        return self.speech_ms >= self.min_speech_ms
