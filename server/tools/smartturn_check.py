"""对照 app/turn.py 的 numpy 特征和官方 WhisperFeatureExtractor，顺带导出 mel_filters.npy。

在 Mac 上跑（服务器的 venv 里没装 transformers，也不需要装）：

  uv run --with transformers --with onnxruntime --with numpy \\
      python tools/smartturn_check.py --model /path/smart-turn-v3.2-cpu.onnx --export-mel a.wav b.wav

--export-mel 把梅尔滤波器组写到模型旁边的 mel_filters.npy（服务端加载模型时一起读）。
wav 要 16 kHz 单声道 16-bit（ffmpeg -ar 16000 -ac 1 -sample_fmt s16）。每条后面补
--silence-ms 毫秒静音，模拟服务端「停顿 200 ms 时问一次」。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app.turn import N_SAMPLES, log_mel  # noqa: E402


def read_wav(p) -> bytes:
    with wave.open(str(p)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), p
        return w.readframes(w.getnframes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--export-mel", action="store_true")
    ap.add_argument("--silence-ms", type=int, default=200)
    ap.add_argument("wavs", nargs="+")
    a = ap.parse_args()

    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(chunk_length=8)
    mel_path = pathlib.Path(a.model).with_name("mel_filters.npy")
    if a.export_mel:
        np.save(mel_path, np.asarray(fe.mel_filters, dtype=np.float64))
        print(f"已导出 {mel_path}（{fe.mel_filters.shape}）")
    mel = np.load(mel_path)
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    for p in a.wavs:
        pcm = read_wav(p) + b"\x00\x00" * (16 * a.silence_ms)
        x = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0
        # 官方 audio_utils.truncate_audio_to_last_n_seconds：超长留最后 8 秒，不够在前面补零
        x8 = x[-N_SAMPLES:]
        x8 = np.pad(x8, (N_SAMPLES - len(x8), 0))
        ref = fe(x8, sampling_rate=16000, return_tensors="np", padding="max_length",
                 max_length=N_SAMPLES, truncation=True,
                 do_normalize=True).input_features.astype(np.float32)
        mine = log_mel(x, mel)[None]
        p_ref = float(np.asarray(sess.run(None, {"input_features": ref})[0]).reshape(-1)[0])
        p_mine = float(np.asarray(sess.run(None, {"input_features": mine})[0]).reshape(-1)[0])
        print(f"{pathlib.Path(p).name:14} 特征最大差 {np.abs(ref - mine).max():.1e}  "
              f"说完了的概率 官方 {p_ref:.3f} / 我们 {p_mine:.3f}")


if __name__ == "__main__":
    main()
