"""小喇叭调音（9/27）：吴语模型出来的声音在盒子的小喇叭上发闷，出声前过一道 EQ + 增益 + 软限幅。

切 120 Hz 以下（小喇叭放不出来，只会糊）→ 压 400 Hz 一带（闷的来源）→ 抬 2.5 kHz 以上（咬字清楚）
→ 按声音定的增益（把几种声音拉到一样响）→ 软限幅（峰值不顶破）。
流式逐块处理，滤波器状态跨块保留：每句话新建一个 Voicing。wu_server.py 用它，跟它放在同一目录。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import sosfilt


def _row(b0, b1, b2, a0, a1, a2):
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


def _highpass(f0, sr, q=0.707):
    w = 2 * math.pi * f0 / sr
    c, al = math.cos(w), math.sin(w) / (2 * q)
    return _row((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al)


def _peaking(f0, sr, db, q):
    w = 2 * math.pi * f0 / sr
    a, c, al = 10 ** (db / 40), math.cos(w), math.sin(w) / (2 * q)
    return _row(1 + al * a, -2 * c, 1 - al * a, 1 + al / a, -2 * c, 1 - al / a)


def _highshelf(f0, sr, db, s=0.8):
    w = 2 * math.pi * f0 / sr
    a, c = 10 ** (db / 40), math.cos(w)
    al = math.sin(w) / 2 * math.sqrt((a + 1 / a) * (1 / s - 1) + 2)
    r = 2 * math.sqrt(a) * al
    return _row(a * ((a + 1) + (a - 1) * c + r), -2 * a * ((a - 1) + (a + 1) * c), a * ((a + 1) + (a - 1) * c - r),
                (a + 1) - (a - 1) * c + r, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - r)


def soft_clip(x: np.ndarray, t: float = 0.7) -> np.ndarray:
    """|x| ≤ t 原样，超过的部分用 tanh 压，最大不过 1。"""
    ax = np.abs(x)
    over = ax > t
    if over.any():
        x = x.copy()
        x[over] = np.sign(x[over]) * (t + (1 - t) * np.tanh((ax[over] - t) / (1 - t)))
    return x


class Voicing:
    def __init__(self, sr: int, gain_db: float = 0.0, hp_hz: float = 120, mud_db: float = -3.0,
                 shelf_db: float = 5.0, shelf_hz: float = 2500):
        rows = [_highpass(hp_hz, sr)]
        if mud_db:
            rows.append(_peaking(400, sr, mud_db, 1.0))
        if shelf_db:
            rows.append(_highshelf(shelf_hz, sr, shelf_db))
        self.sos = np.array(rows)
        self.zi = np.zeros((len(rows), 2))
        self.g = 10 ** (gain_db / 20)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return soft_clip(y * self.g)
