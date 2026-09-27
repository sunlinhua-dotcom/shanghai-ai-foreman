#!/usr/bin/env python3
"""ASR 引擎对照：字错率（CER）+「listen stop 之后还要多久」。

两步：
  1) 在能上网的机器上合成测试语音（Edge TTS，和盒子听到的同一路音色管线）：
        python tools/asr_bench.py --synth
     10 句固定语料落在 tmp/asr_test/，rsync 到目标机器保证两边听的是同一份音频。
  2) 在目标机器（EX10）上跑引擎：
        python tools/asr_bench.py --engine sherpa-onnx --model-dir /srv/models/xxx
        python tools/asr_bench.py --engine faster-whisper --model small

关键指标 **stop_ms**：模拟真机——音频按 60 ms 一帧喂进去（流式引擎说话期间就在解），
到「用户说完」那一刻才开始计时，量的是收尾还要多久。整段式引擎这一段就是整段转写时间。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from app.audio import resample_pcm16, wav_to_pcm16_mono  # noqa: E402
from app.config import load_config  # noqa: E402

# 长短混合 / 英数混排 / 人名地名，10 句
SENTENCES = [
    "你好。",
    "今天天气怎么样？",
    "帮我打开 HY4 的日志。",
    "把客厅的灯调到百分之三十。",
    "我想听周杰伦的稻香。",
    "明天下午三点提醒我去中关村开会。",
    "张伟说他周五要去杭州出差，你帮我记一下。",
    "从北京南站到上海虹桥的高铁大概要多久？",
    "用 Python 写一个快速排序，顺便解释一下时间复杂度。",
    "帮我查一下 ESP32 S3 的 PSRAM 有多大，然后把结果发到我的邮箱。",
]

DEFAULT_WAV_DIR = SERVER_DIR / "tmp" / "asr_test"


# ---------- CER ----------

_KEEP = re.compile(r"[0-9A-Za-z一-鿿]")


def normalize(text: str) -> str:
    """比对前拉平：全角→半角、去标点空格、英文统一大写。"""
    text = unicodedata.normalize("NFKC", text or "").upper()
    return "".join(ch for ch in text if _KEEP.match(ch))


def edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    return edit_distance(r, h) / max(1, len(r))


# ---------- 合成 ----------

async def synth_all(wav_dir: Path) -> None:
    from app.audio import pcm16_to_wav
    from app.tts import create_tts

    cfg = load_config(require_api_key=False)
    wav_dir.mkdir(parents=True, exist_ok=True)
    tts = create_tts(cfg)
    try:
        for i, text in enumerate(SENTENCES, 1):
            out = wav_dir / f"{i:02d}.wav"
            if out.exists() and out.stat().st_size > 2000:
                print(f"  {i:02d} 已有，跳过")
                continue
            wav, _ = await tts.synth(text)
            if not wav:
                raise RuntimeError(f"第 {i} 句合成失败")
            # 统一成 16 k 单声道，跟真机上行一样
            pcm, rate = wav_to_pcm16_mono(wav)
            out.write_bytes(pcm16_to_wav(resample_pcm16(pcm, rate, 16000), 16000))
            print(f"  {i:02d} {text}  → {out.name} ({len(out.read_bytes())} B)")
            await asyncio.sleep(0.2)  # 别把 Edge 打太密
    finally:
        close = getattr(tts, "close", None)
        if close:
            await close()
    (wav_dir / "refs.json").write_text(
        json.dumps(SENTENCES, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------- 跑引擎 ----------

def build_asr(args):
    cfg = load_config(require_api_key=False)
    cfg.asr_provider = args.engine
    if args.model_dir:
        cfg.asr_model_dir = args.model_dir
    if args.model:
        cfg.asr_model = args.model
    if args.num_threads:
        cfg.asr_num_threads = args.num_threads
    if args.compute_type:
        cfg.asr_compute_type = args.compute_type
    from app.asr import create_asr

    return create_asr(cfg)


async def run_one(asr, pcm16: bytes, frame_bytes: int) -> tuple[str, float, float]:
    """按 60 ms 帧喂完再收尾。返回 (文本, 喂帧总耗时 ms, stop 后耗时 ms)。"""
    stream = asr.create_stream() if getattr(asr, "streaming", False) else None
    t_feed0 = time.monotonic()
    if stream is not None:
        for i in range(0, len(pcm16), frame_bytes):
            await stream.feed(pcm16[i : i + frame_bytes])
    feed_ms = (time.monotonic() - t_feed0) * 1000
    t_stop = time.monotonic()
    if stream is not None:
        text = await stream.finalize()
        await stream.close()
    else:
        text = await asr.transcribe(pcm16, 16000)
    return text, feed_ms, (time.monotonic() - t_stop) * 1000


async def bench(args) -> int:
    wav_dir = Path(args.wav_dir)
    refs = json.loads((wav_dir / "refs.json").read_text(encoding="utf-8"))
    asr = build_asr(args)
    print(f"引擎 {asr.name}（streaming={getattr(asr, 'streaming', False)}）预热中…", flush=True)
    t0 = time.monotonic()
    await asr.warmup()
    print(f"预热完成 {time.monotonic()-t0:.1f}s\n", flush=True)

    frame_bytes = 16000 * 60 // 1000 * 2
    rows = []
    for i, ref in enumerate(refs, 1):
        wav = wav_dir / f"{i:02d}.wav"
        pcm, rate = wav_to_pcm16_mono(wav.read_bytes())
        pcm = resample_pcm16(pcm, rate, 16000)
        dur = len(pcm) / 2 / 16000
        best = None
        for _ in range(max(1, args.repeat)):
            text, feed_ms, stop_ms = await run_one(asr, pcm, frame_bytes)
            if best is None or stop_ms < best[2]:
                best = (text, feed_ms, stop_ms)
        text, feed_ms, stop_ms = best
        rows.append(
            {"i": i, "ref": ref, "hyp": text, "cer": cer(ref, text),
             "dur_s": dur, "feed_ms": feed_ms, "stop_ms": stop_ms}
        )
        print(f"{i:02d} [{dur:4.2f}s] stop后 {stop_ms:6.1f}ms  CER {rows[-1]['cer']*100:5.1f}%  {text!r}",
              flush=True)

    total_ref = sum(len(normalize(r["ref"])) for r in rows)
    total_err = sum(edit_distance(normalize(r["ref"]), normalize(r["hyp"])) for r in rows)
    print("\n===== 汇总 =====")
    print(f"引擎            : {asr.name} {args.model_dir or args.model}")
    print(f"总体 CER        : {total_err/max(1,total_ref)*100:.2f}%  ({total_err}/{total_ref} 字)")
    print(f"stop 后耗时     : 中位 {sorted(r['stop_ms'] for r in rows)[len(rows)//2]:.0f} ms, "
          f"最大 {max(r['stop_ms'] for r in rows):.0f} ms, "
          f"均值 {sum(r['stop_ms'] for r in rows)/len(rows):.0f} ms")
    print(f"说话期间总解码  : 均值 {sum(r['feed_ms'] for r in rows)/len(rows):.0f} ms "
          f"（音频均长 {sum(r['dur_s'] for r in rows)/len(rows):.2f}s）")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"engine": asr.name, "model": args.model_dir or args.model,
                        "overall_cer": total_err / max(1, total_ref), "rows": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已存 {args.json_out}")
    await asr.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ASR 引擎 CER + 收尾延迟对照")
    ap.add_argument("--synth", action="store_true", help="只合成测试语音")
    ap.add_argument("--wav-dir", default=str(DEFAULT_WAV_DIR))
    ap.add_argument("--engine", default="sherpa-onnx")
    ap.add_argument("--model-dir", default="", help="sherpa-onnx 模型目录")
    ap.add_argument("--model", default="", help="faster-whisper 模型名")
    ap.add_argument("--compute-type", default="")
    ap.add_argument("--num-threads", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="每句跑几遍取最快（去掉抖动）")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    if args.synth:
        asyncio.run(synth_all(Path(args.wav_dir)))
        return 0
    return asyncio.run(bench(args))


if __name__ == "__main__":
    raise SystemExit(main())
