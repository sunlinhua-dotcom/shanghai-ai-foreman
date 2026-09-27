"""Opus 编解码 + 重采样 + wav 读写。

上行：设备恒 16 kHz / 单声道 / 60 ms Opus（research §1.3, audio_service.h:39）。
下行：按服务端 hello 里声明的 sample_rate（默认 24 kHz）/ 60 ms Opus。
"""
from __future__ import annotations

import io
import wave

import numpy as np
import opuslib_next as opuslib

# opuslib 的 application 常量
_APP_VOIP = opuslib.APPLICATION_VOIP


class OpusDecoder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._dec = opuslib.Decoder(sample_rate, channels)

    def decode(self, payload: bytes, frame_samples: int) -> bytes:
        """返回 int16 小端 PCM bytes。"""
        return self._dec.decode(payload, frame_samples)


class OpusEncoder:
    def __init__(self, sample_rate: int = 24000, channels: int = 1, bitrate: int = 24000):
        self.sample_rate = sample_rate
        self.channels = channels
        self._enc = opuslib.Encoder(sample_rate, channels, _APP_VOIP)
        try:
            self._enc.bitrate = bitrate
        except Exception:  # 老版本 opuslib 没有该属性
            pass

    def encode(self, pcm: bytes, frame_samples: int) -> bytes:
        return self._enc.encode(pcm, frame_samples)


def pcm_to_frames(pcm: bytes, frame_samples: int, channels: int = 1) -> list[bytes]:
    """把 int16 PCM 切成定长帧，最后一帧补零。"""
    bytes_per_frame = frame_samples * channels * 2
    frames: list[bytes] = []
    for off in range(0, len(pcm), bytes_per_frame):
        chunk = pcm[off : off + bytes_per_frame]
        if len(chunk) < bytes_per_frame:
            chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))
        frames.append(chunk)
    return frames


def encode_pcm_to_opus_frames(
    pcm: bytes, encoder: OpusEncoder, frame_samples: int
) -> list[bytes]:
    return [encoder.encode(f, frame_samples) for f in pcm_to_frames(pcm, frame_samples)]


# ---------- 重采样 ----------

def _resample_soxr(x: np.ndarray, src: int, dst: int) -> np.ndarray | None:
    try:
        import soxr
    except ImportError:
        return None
    return soxr.resample(x, src, dst, quality="HQ")


def _resample_numpy(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """兜底：降采样先过一个简单 FIR 低通再线性插值，避免混叠。"""
    xf = x.astype(np.float64)
    if dst < src:
        taps = max(3, int(round(2 * src / dst)) | 1)
        kernel = np.hanning(taps + 2)[1:-1]
        kernel /= kernel.sum()
        xf = np.convolve(xf, kernel, mode="same")
    n_out = int(round(len(xf) * dst / src))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float64)
    src_idx = np.linspace(0, len(xf) - 1, n_out)
    return np.interp(src_idx, np.arange(len(xf)), xf)


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate or not pcm:
        return pcm
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    y = _resample_soxr(x, src_rate, dst_rate)
    if y is None:
        y = _resample_numpy(x, src_rate, dst_rate)
    y = np.clip(np.asarray(y, dtype=np.float64), -1.0, 1.0)
    return (y * 32767.0).astype("<i2").tobytes()


# ---------- wav ----------

def wav_to_pcm16_mono(data: bytes) -> tuple[bytes, int]:
    """读 wav bytes → (int16 单声道 PCM, sample_rate)。只支持 16/24/32-bit PCM。"""
    with wave.open(io.BytesIO(data), "rb") as wf:
        n_ch = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if width == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        v = np.where(v & 0x800000, v - 0x1000000, v)
        arr = v.astype(np.float32) / 8388608.0
    elif width == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported wav sample width {width}")

    if n_ch > 1:
        arr = arr.reshape(-1, n_ch).mean(axis=1)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return pcm, rate


def pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
