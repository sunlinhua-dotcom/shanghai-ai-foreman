"""sherpa-onnx ASR：**边听边认**（流式）。

为什么换掉 faster-whisper：whisper 的 encoder 永远吃一个 30 s 窗口，哪怕只送
0.5 s 音频也要付几乎整份 encoder 的钱，所以「说完再整段转写」是硬性的 1.2–1.6 s。
sherpa-onnx 的 online 模型是真流式（zipformer transducer / paraformer）：
说话期间每 60 ms 帧就解一次，`listen stop` 时只剩不到一个 chunk 的尾巴，
收尾开销是几十毫秒。

支持三种模型目录（按文件名自动认）：

| 目录里有 | 引擎 | 流式 |
|---|---|---|
| `encoder-*.onnx` + `decoder-*.onnx` + `joiner-*.onnx` | OnlineRecognizer.from_transducer | ✅ |
| `encoder.onnx` + `decoder.onnx`（无 joiner） | OnlineRecognizer.from_paraformer | ✅ |
| `model.onnx` / `model.int8.onnx` | OfflineRecognizer.from_sense_voice | ❌（整段） |

模型不从 HuggingFace 拿（EX10 到不了）：GitHub release 的 tar.bz2 解开就是目录，
路径写进 `XZ_ASR_MODEL_DIR`。
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("xz.asr")

_CJK = r"㐀-䶿一-鿿豈-﫿　-〿＀-￯"
_SPACE_AROUND_CJK = re.compile(rf"(?<=[{_CJK}])\s+|\s+(?=[{_CJK}])")


# 紧跟在拉丁字母/数字后面的汉字数字：模型不做 ITN，"HY4" 会识别成 "HY四"、
# "ESP32" 成 "ESP三二"。只在「字母数字 + 汉字数字」直接相连时改写，
# 正常中文（"三个人"、"今天三点"）不受影响。
_CN_DIGITS = {"零": "0", "〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
              "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
_LATIN_THEN_CN_DIGITS = re.compile(r"(?<=[0-9A-Za-z])([零〇一二三四五六七八九]+)")


def tidy(text: str) -> str:
    """BPE 结果里中文之间会带空格（"你 好"），去掉；英文词间的空格留着。

    顺手把「字母后紧跟的汉字数字」还原成阿拉伯数字（HY四 → HY4）：
    设备指令里英数混排很常见，而这个模型本身不做 inverse text normalization。
    """
    text = _SPACE_AROUND_CJK.sub("", (text or "").strip()).strip()
    return _LATIN_THEN_CN_DIGITS.sub(
        lambda m: "".join(_CN_DIGITS[c] for c in m.group(1)), text
    )


def _pcm_to_float(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0


class _ModelFiles:
    """扫一遍模型目录，认出是哪种模型、该用哪几个 onnx。"""

    def __init__(self, model_dir: Path, prefer_int8: bool):
        self.dir = model_dir
        self.prefer_int8 = prefer_int8
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"ASR 模型目录不存在：{model_dir}\n"
                "从 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ "
                "下载对应 tar.bz2 解开，把目录路径写进 XZ_ASR_MODEL_DIR。"
            )
        self.tokens = self._one("tokens.txt")
        joiner = self._pick("joiner")
        encoder = self._pick("encoder")
        decoder = self._pick("decoder")
        model = self._pick("model")
        if encoder and decoder and joiner:
            self.kind = "transducer"
            # 官方建议：encoder/joiner 量化到 int8，decoder（embedding+conv，很小）留 fp32
            self.encoder, self.joiner = encoder, joiner
            self.decoder = self._pick("decoder", force_fp32=True) or decoder
        elif encoder and decoder:
            self.kind = "paraformer"
            self.encoder, self.decoder, self.joiner = encoder, decoder, None
        elif model:
            self.kind = "sense-voice"
            self.model = model
            self.encoder = self.decoder = self.joiner = None
        else:
            raise FileNotFoundError(f"{model_dir} 里认不出 sherpa-onnx 模型文件（*.onnx）")

    def _one(self, name: str) -> str:
        p = self.dir / name
        if not p.exists():
            raise FileNotFoundError(f"缺 {p}")
        return str(p)

    def _pick(self, stem: str, force_fp32: bool = False) -> str | None:
        """选 `<stem>*.onnx`；prefer_int8 时优先 `.int8.onnx`。"""
        cands = sorted(self.dir.glob(f"{stem}*.onnx"))
        if not cands:
            return None
        int8 = [c for c in cands if ".int8." in c.name]
        fp32 = [c for c in cands if ".int8." not in c.name]
        if force_fp32:
            return str(fp32[0]) if fp32 else str(int8[0])
        if self.prefer_int8 and int8:
            return str(int8[0])
        return str(fp32[0]) if fp32 else str(int8[0])

    def describe(self) -> str:
        if self.kind == "sense-voice":
            return f"{self.kind} {Path(self.model).name}"
        parts = [Path(p).name for p in (self.encoder, self.decoder, self.joiner) if p]
        return f"{self.kind} " + " + ".join(parts)


class SherpaOnnxASR:
    name = "sherpa-onnx"

    def __init__(
        self,
        model_dir: str,
        compute_type: str = "int8",
        num_threads: int = 4,
        tail_padding_ms: int = 300,
    ):
        self.model_dir = Path(model_dir).expanduser()
        self.prefer_int8 = (compute_type or "int8").lower().startswith("int8")
        self.num_threads = max(1, int(num_threads))
        self.tail_padding_ms = max(0, int(tail_padding_ms))
        self.files = _ModelFiles(self.model_dir, self.prefer_int8)
        self.streaming = self.files.kind in ("transducer", "paraformer")
        self._rec = None
        self._load_lock = asyncio.Lock()
        # 整段式（sense-voice）时 CPU 推理串行即可；流式各 stream 互不相干，不用锁
        self._offline_lock = asyncio.Lock()

    # ---------- 模型加载 ----------

    def _ensure_sync(self):
        if self._rec is not None:
            return self._rec
        import sherpa_onnx

        t0 = time.time()
        f = self.files
        if f.kind == "transducer":
            self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=f.tokens,
                encoder=f.encoder,
                decoder=f.decoder,
                joiner=f.joiner,
                num_threads=self.num_threads,
                sample_rate=16000,
                feature_dim=80,
                decoding_method="greedy_search",
                # 断句交给设备的 listen stop + 服务端 webrtcvad，识别器不自作主张
                enable_endpoint_detection=False,
                provider="cpu",
            )
        elif f.kind == "paraformer":
            self._rec = sherpa_onnx.OnlineRecognizer.from_paraformer(
                tokens=f.tokens,
                encoder=f.encoder,
                decoder=f.decoder,
                num_threads=self.num_threads,
                sample_rate=16000,
                feature_dim=80,
                decoding_method="greedy_search",
                enable_endpoint_detection=False,
                provider="cpu",
            )
        else:  # sense-voice：整段式
            self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=f.model,
                tokens=f.tokens,
                num_threads=self.num_threads,
                use_itn=True,
                language="auto",
                provider="cpu",
            )
        log.info(
            "sherpa-onnx 就绪：%s（%s，%d 线程，%.1fs）",
            f.describe(), "流式" if self.streaming else "整段", self.num_threads, time.time() - t0,
        )
        return self._rec

    async def _ensure(self):
        if self._rec is None:
            async with self._load_lock:
                if self._rec is None:
                    await asyncio.to_thread(self._ensure_sync)
        return self._rec

    async def warmup(self) -> None:
        await self._ensure()
        # 真跑一次推理：只 load 盖不住首轮 lazy-init（onnxruntime 的 arena/线程池）
        await self.transcribe(b"\x00" * 16000, 16000)

    async def close(self) -> None:
        self._rec = None

    # ---------- 整段接口（listen detect / 非流式回退 / 基准测试用） ----------

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        if not pcm16:
            return ""
        rec = await self._ensure()
        if sample_rate != 16000:
            from ..audio import resample_pcm16

            pcm16 = resample_pcm16(pcm16, sample_rate, 16000)
        t0 = time.monotonic()
        if self.streaming:
            stream = self.create_stream()
            await stream.feed(pcm16)
            text = await stream.finalize()
        else:
            async with self._offline_lock:
                text = await asyncio.to_thread(self._offline_sync, rec, pcm16)
        dur_ms = len(pcm16) / 2 / 16000 * 1000
        log.info("ASR %.0fms 音频 → %.2fs → %r", dur_ms, time.monotonic() - t0, text)
        return text

    @staticmethod
    def _offline_sync(rec, pcm16: bytes) -> str:
        s = rec.create_stream()
        s.accept_waveform(16000, _pcm_to_float(pcm16))
        rec.decode_stream(s)
        return tidy(s.result.text)

    # ---------- 流式接口 ----------

    def create_stream(self):
        return _SherpaStream(self)


class _SherpaStream:
    """一次发话：feed 期间滚动解码，finalize 只处理尾巴。

    sherpa 的 OnlineStream 是 C++ 对象、解码时放开 GIL，但同一个 stream 不能并发进，
    所以 feed/finalize 各自 `to_thread` 且由 session 顺序调用（WS 读循环天然串行）。
    """

    def __init__(self, provider: SherpaOnnxASR):
        self.p = provider
        self._stream = None
        self._rec = None
        self._buffer = bytearray()  # 模型还没加载完时先攒着
        self._text = ""
        self._finished = False

    async def _ensure(self):
        if self._stream is None:
            self._rec = await self.p._ensure()
            self._stream = self._rec.create_stream()
        return self._stream

    def _feed_sync(self, pcm16: bytes) -> None:
        self._stream.accept_waveform(16000, _pcm_to_float(pcm16))
        if self.p.streaming:
            while self._rec.is_ready(self._stream):
                self._rec.decode_stream(self._stream)

    async def feed(self, pcm16: bytes) -> None:
        if self._finished or not pcm16:
            return
        if self._stream is None:
            # 模型还在后台加载（服务刚起、第一句话）：先攒，别丢音频
            self._buffer.extend(pcm16)
            if self.p._rec is None:
                return
            await self._ensure()
            pcm16, self._buffer = bytes(self._buffer), bytearray()
        await asyncio.to_thread(self._feed_sync, pcm16)
        if self.p.streaming:
            self._text = tidy(self._rec.get_result(self._stream))

    def _finalize_sync(self) -> str:
        pad = self.p.tail_padding_ms
        if pad:
            self._stream.accept_waveform(16000, np.zeros(16000 * pad // 1000, np.float32))
        if self.p.streaming:
            self._stream.input_finished()
            while self._rec.is_ready(self._stream):
                self._rec.decode_stream(self._stream)
            return tidy(self._rec.get_result(self._stream))
        self._rec.decode_stream(self._stream)
        return tidy(self._stream.result.text)

    async def finalize(self) -> str:
        if self._finished:
            return self._text
        self._finished = True
        if self._stream is None:
            await self._ensure()
            if self._buffer:
                pcm, self._buffer = bytes(self._buffer), bytearray()
                await asyncio.to_thread(self._feed_sync, pcm)
        elif self._buffer:
            pcm, self._buffer = bytes(self._buffer), bytearray()
            await asyncio.to_thread(self._feed_sync, pcm)
        self._text = await asyncio.to_thread(self._finalize_sync)
        return self._text

    @property
    def partial(self) -> str:
        return self._text

    async def close(self) -> None:
        self._stream = None
        self._rec = None
        self._buffer = bytearray()
