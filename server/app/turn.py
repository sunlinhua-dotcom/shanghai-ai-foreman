"""句尾判断：Pipecat 的 smart-turn v3（github.com/pipecat-ai/smart-turn，BSD-2）。

光看静音会把想词的停顿当成说完了（9/24 主人原话「我没说完就结束了」）。smart-turn
听的是语气和语调：最后 8 秒音频 → Whisper-tiny 编码器 → 「这句说完了」的概率，
CPU 上十几毫秒，23 种语言里有中文（官方基准中文准确率 85.8%）。

session.py 的用法：停顿 XZ_TURN_CHECK_MS（200 ms）时问它一次——像说完了就照旧
XZ_TURN_COMPLETE_MS（800 ms）收尾（它对中文有一成多误判，不拿它来抢话）；像没说完就放宽到
XZ_TURN_INCOMPLETE_MS（1.6 s），给人想词的时间。又开口就恢复默认，下一个停顿再问。
抢跑识别（停顿 350 ms 时）出来的字也算数：只有「嗯」等 2.5 s，停在「然后 / 就是」上等 2 s，
带问号的完整问句不管模型怎么判都按 800 ms 收尾。

特征照 transformers 的 WhisperFeatureExtractor(chunk_length=8) 用 numpy 重写（不为这一件事装
transformers）：取最后 8 秒，不够的在前面补零，整段零均值单位方差，400 点 Hann 窗、
160 点步长、80 维梅尔（滤波器组从 transformers 导出成 mel_filters.npy 跟模型放一起），
log10、压到最大值往下 8、(x+4)/4。tools/smartturn_check.py 对照过官方实现。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("xz.turn")

SR = 16000
N_FFT = 400
HOP = 160
N_SAMPLES = 8 * SR


def log_mel(x: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """16 kHz float 波形 → (80, 800) 的 Whisper 对数梅尔谱，跟 WhisperFeatureExtractor 一致。"""
    x = np.asarray(x, dtype=np.float32)[-N_SAMPLES:]
    buf = np.zeros(N_SAMPLES, dtype=np.float32)
    if len(x):
        buf[N_SAMPLES - len(x):] = x   # 不够 8 秒在前面补零（官方 audio_utils 就是这么补的）
    # 均值方差按整段 8 秒（含补的零）算：官方先补齐再进特征提取器，注意力掩码是满的
    buf = (buf - buf.mean()) / np.sqrt(buf.var() + 1e-7)
    padded = np.pad(buf, N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]   # (801, 400)
    window = np.hanning(N_FFT + 1)[:-1]                                      # 周期 Hann
    power = np.abs(np.fft.rfft(frames * window, n=N_FFT, axis=-1)) ** 2      # (801, 201)
    spec = np.log10(np.maximum(power @ mel_filters, 1e-10)).T[:, :-1]        # (80, 800)
    spec = np.maximum(spec, spec.max() - 8.0)
    return ((spec + 4.0) / 4.0).astype(np.float32)


class SmartTurn:
    def __init__(self, model_path: str | Path, threads: int = 1):
        import onnxruntime as ort

        model_path = Path(model_path)
        self.mel_filters = np.load(model_path.with_name("mel_filters.npy"))   # (201, 80)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = max(1, int(threads))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.name = model_path.stem

    def predict(self, pcm16: bytes) -> float:
        """16 kHz int16 PCM → 「说完了」的概率（0~1）。阻塞，调用方放线程里跑。"""
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        feats = log_mel(x, self.mel_filters)[None]
        out = self.sess.run(None, {"input_features": feats})
        return float(np.asarray(out[0]).reshape(-1)[0])


def load_smart_turn(cfg) -> SmartTurn | None:
    path = (getattr(cfg, "turn_model", "") or "").strip()
    if not path:
        return None
    try:
        st = SmartTurn(path, threads=getattr(cfg, "turn_threads", 1))
        st.predict(b"\x00\x00" * SR)   # 预热：第一次跑要建图，别让第一句话来付
    except Exception as e:  # noqa: BLE001 - 断句模型挂了就退回纯静音断句
        log.warning("断句模型加载失败，退回只看静音：%s", e)
        return None
    log.info("断句模型已加载：%s", st.name)
    return st
