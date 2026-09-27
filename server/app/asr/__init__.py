"""ASR provider 接口 + 注册表。

两种 provider：
  - **整段式**（faster-whisper）：`listen stop` 之后把整段 PCM 丢进去，等一个结果。
    简单，但「等人说完才开始算」，一段 3.5 s 的话要额外花 1.2–1.6 s。
  - **流式 / 边听边认**（sherpa-onnx）：`streaming = True`，另外实现 `create_stream()`。
    说话期间每来一帧就 `feed()`，`listen stop` 时 `finalize()` 只处理最后那点尾巴，
    实测 stop 之后 20–60 ms。

session.py 用 `provider.streaming` 决定走哪条路；两条路都要能收尾
（设备自己发 listen stop，服务端 VAD 也会判句尾）。

换 provider 只要实现 ASRProvider（流式再加 ASRStream）并在 create_asr() 的分支里加一行。
"""
from __future__ import annotations

import abc


class ASRStream(abc.ABC):
    """一次发话的流式识别句柄。生命周期 = listen start → listen stop。"""

    @abc.abstractmethod
    async def feed(self, pcm16: bytes) -> None:
        """喂一帧 16 kHz 单声道 int16 PCM（说话期间不断调用）。"""

    @abc.abstractmethod
    async def finalize(self) -> str:
        """收尾：补一点静音、把剩下的解码完，返回整句文本。"""

    @property
    def partial(self) -> str:
        """当前已识别出的文本（可选，调试用）。"""
        return ""

    async def close(self) -> None:
        return None


class ASRProvider(abc.ABC):
    name = "base"
    streaming = False  # True = 支持 create_stream() 的边听边认

    @abc.abstractmethod
    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        """接收 int16 单声道 PCM，返回识别文本（可能为空串）。"""

    def create_stream(self) -> ASRStream | None:
        """流式 provider 返回一个新的 ASRStream；整段式返回 None。"""
        return None

    async def warmup(self) -> None:  # 可选：预热模型
        return None

    async def close(self) -> None:
        return None


def create_asr(cfg) -> ASRProvider:
    provider = (cfg.asr_provider or "sherpa-onnx").lower()
    if provider in ("sherpa-onnx", "sherpa_onnx", "sherpa", "streaming"):
        from .sherpa_onnx_asr import SherpaOnnxASR

        return SherpaOnnxASR(
            model_dir=cfg.asr_model_dir,
            compute_type=cfg.asr_compute_type,
            num_threads=cfg.asr_num_threads,
            tail_padding_ms=cfg.asr_tail_padding_ms,
        )
    if provider in ("qwen3-http", "qwen3", "gx10"):
        from .qwen3_http_asr import Qwen3HttpASR

        fallback = None
        if cfg.asr_model_dir:
            from .sherpa_onnx_asr import SherpaOnnxASR

            fallback = SherpaOnnxASR(
                model_dir=cfg.asr_model_dir,
                compute_type=cfg.asr_compute_type,
                num_threads=cfg.asr_num_threads,
                tail_padding_ms=cfg.asr_tail_padding_ms,
            )
        return Qwen3HttpASR(cfg.asr_http_url, cfg.asr_http_timeout, fallback)
    if provider in ("faster-whisper", "faster_whisper", "fw"):
        from .faster_whisper_asr import FasterWhisperASR

        return FasterWhisperASR(
            model=cfg.asr_model,
            language=cfg.asr_language,
            device=cfg.asr_device,
            compute_type=cfg.asr_compute_type,
        )
    if provider in ("null", "none", "echo"):
        from .null_asr import NullASR

        return NullASR()
    raise ValueError(f"未知 ASR provider: {cfg.asr_provider}")
