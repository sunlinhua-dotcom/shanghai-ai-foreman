"""建声纹 / 给录音打分（EX10 部署目录里跑，读 .env 里的 XZ_VOICEPRINT_*）。

  .venv/bin/python tools/voiceprint.py build [--skip a.wav,b.wav]   # enroll/*.wav → voiceprint.json
  .venv/bin/python tools/voiceprint.py score x.wav ...                # 每个 wav 跟现有声纹的分数

build 打印每句的「留一分」：拿其余几句的平均去比这一句，分低的多半不是本人（或太短太吵），用 --skip 去掉重建。
建完服务要重启才读新声纹。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.voiceprint import VoicePrint  # noqa: E402


def _env() -> dict:
    out = {}
    for ln in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _wav(p: Path) -> tuple[bytes, int]:
    with wave.open(str(p)) as w:
        return w.readframes(w.getnframes()), w.getframerate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--skip", default="", help="build 时不用的文件名，逗号分隔")
    a = ap.parse_args()
    env = _env()
    model = env.get("XZ_VOICEPRINT_MODEL", "")
    data = os.path.expanduser(env.get("XZ_VOICEPRINT_DIR", "~/xz-data/voiceprint"))
    vp = VoicePrint(model, data, "enroll", 0.0)

    if a.cmd == "score":
        if vp.print_ is None:
            sys.exit("还没建声纹，先 build")
        for f in a.files:
            pcm, sr = _wav(Path(f))
            print(f"{vp.score(pcm, sr):.3f}  {f}")
        return

    d = vp.dir / "enroll"
    text = {}
    if (d / "index.jsonl").exists():
        for ln in (d / "index.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            text[r["file"]] = r.get("text", "")
    skip = {x for x in a.skip.split(",") if x}
    files = [p for p in sorted(d.glob("*.wav")) if p.name not in skip]
    if len(files) < 2:
        sys.exit(f"{d} 里只有 {len(files)} 句，至少要 2 句")
    embs = np.stack([vp.embed(*_wav(p)) for p in files])
    for i, p in enumerate(files):
        rest = np.delete(embs, i, axis=0).mean(axis=0)
        rest /= np.linalg.norm(rest) + 1e-9
        pcm, sr = _wav(p)
        print(f"{float(embs[i] @ rest):.3f}  {p.name}  {len(pcm) / 2 / sr:.1f}s  {text.get(p.name, '')}")
    mean = embs.mean(axis=0)
    out = {"model": Path(model).name, "n": len(files), "files": [p.name for p in files],
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "mean": mean.tolist()}
    (vp.dir / "voiceprint.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"声纹已存：{vp.dir / 'voiceprint.json'}（{len(files)} 句）")


if __name__ == "__main__":
    main()
