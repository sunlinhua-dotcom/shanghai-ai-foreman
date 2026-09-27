"""声纹：只认用户本人的声音（9/25 用户：「有时候会有杂音，能不能记录我的声纹只认得我的声音」）。

sherpa-onnx 的说话人向量模型（CAM++ 中文 16k）每句话算一个向量，跟录好的声纹比余弦相似度。
两种模式（XZ_VOICEPRINT_MODE）：
- enroll：只录不拦。每句被接受的话都存成 wav + 一行索引，给 tools/voiceprint.py build 建声纹用
- on：低于门槛（XZ_VOICEPRINT_THRESHOLD）就当不是用户在说，不回答
录音和声纹放在网站目录外面，目录权限 700。
"""
from __future__ import annotations

import json
import logging
import time
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("xz.voiceprint")


class VoicePrint:
    def __init__(self, model: str, data_dir: str, mode: str, threshold: float, num_threads: int = 2):
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model, num_threads=num_threads, provider="cpu")
        if not cfg.validate():
            raise ValueError(f"声纹模型不可用：{model}")
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.dir = Path(data_dir)
        self.mode = mode
        self.threshold = threshold
        self.print_: np.ndarray | None = None
        self.reload()

    def reload(self) -> None:
        p = self.dir / "voiceprint.json"
        if p.exists():
            v = np.asarray(json.loads(p.read_text(encoding="utf-8"))["mean"], dtype=np.float32)
            self.print_ = v / (np.linalg.norm(v) + 1e-9)

    def embed(self, pcm: bytes, sr: int) -> np.ndarray:
        s = self._ex.create_stream()
        s.accept_waveform(sr, np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0)
        s.input_finished()
        v = np.asarray(self._ex.compute(s), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    # 9/25 用户三句录音切片实测（别的两句建声纹、比这一句）：3 秒片段最低 0.68，2 秒片段中位 0.77，
    # 但带停顿的 1.5–2 秒片段能掉到 0.2–0.3；Edge 六种别人的声音 0.01–0.24。门槛 0.35，太短的不查
    MIN_S = 1.5

    def score(self, pcm: bytes, sr: int) -> float | None:
        """跟声纹的余弦相似度；还没建声纹、太短（「几点了」这种）或算不出来就是 None（放行）。在线程里跑。"""
        if self.print_ is None or len(pcm) / 2 / sr < self.MIN_S:
            return None
        try:
            return float(np.dot(self.embed(pcm, sr), self.print_))
        except Exception as e:  # noqa: BLE001  太短之类：不拦
            log.warning("声纹算不出来：%s", e)
            return None

    def passes(self, score: float | None) -> bool:
        return self.mode != "on" or score is None or score >= self.threshold

    def save(self, pcm: bytes, sr: int, text: str, score: float | None) -> None:
        """enroll 模式：这句存下来建声纹用。在线程里跑。"""
        d = self.dir / "enroll"
        d.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + ".wav"
        with wave.open(str(d / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm)
        with open(d / "index.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"file": name, "text": text, "score": score}, ensure_ascii=False) + "\n")


def keep_recent(data_dir: str, keep: int, pcm: bytes, sr: int, text: str, score: float | None) -> None:
    """滚动存最近几段本人说的话（wav + 同名 json），在线程里跑。放在网站目录外面、目录 700。"""
    try:
        _keep_recent(Path(data_dir), keep, pcm, sr, text, score)
    except OSError as e:
        log.warning("存最近录音失败：%s", e)


def _keep_recent(d: Path, keep: int, pcm: bytes, sr: int, text: str, score: float | None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    name = time.strftime("%Y%m%d-%H%M%S")
    with wave.open(str(d / f"{name}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    meta = {"text": text, "score": score, "seconds": round(len(pcm) / 2 / sr, 1)}
    (d / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    for p in sorted(d.glob("*.wav"))[:-keep]:
        p.unlink(missing_ok=True)
        p.with_suffix(".json").unlink(missing_ok=True)
