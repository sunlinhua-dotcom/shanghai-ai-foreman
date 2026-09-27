"""用豆包语音念一句存成 wav，出样音、试音色和语音指令用（在 EX10 上跑，key 从 .env 读）。

    .venv/bin/python tools/tts_say.py --voice zh_female_xiaohe_uranus_bigtts \\
        --instruction 用上海话说 --out tmp/volc/xiaohe.wav "你好呀，今天有点冷。"

只打印文件名和耗时，不打印 key。
"""
import argparse
import asyncio
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.audio import pcm16_to_wav  # noqa: E402
from app.config import load_config  # noqa: E402
from app.tts.volc import VolcTTS  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--voice", default="")
    ap.add_argument("--instruction", default=None, help="默认用 .env 的 XZ_VOLC_INSTRUCTION")
    ap.add_argument("--rate", type=int, default=None, help="语速 -50..100")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(require_api_key=False)
    if not cfg.volc_api_key:
        print("EX10 的 .env 里还没有 XZ_VOLC_API_KEY，先敲 doubao-key", file=sys.stderr)
        return 1
    tts = VolcTTS(
        api_key=cfg.volc_api_key,
        voice=a.voice or cfg.tts_voice,
        resource_id=cfg.volc_resource_id,
        instruction=cfg.volc_instruction if a.instruction is None else a.instruction,
        speech_rate=cfg.volc_speech_rate if a.rate is None else a.rate,
        sample_rate=cfg.downlink_sample_rate,
    )
    t0 = time.monotonic()
    first = None
    chunks = []
    try:
        async for chunk in tts.synth_stream(a.text):
            if first is None:
                first = time.monotonic() - t0
            chunks.append(chunk)
    finally:
        await tts.close()
    pcm = b"".join(chunks)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pcm16_to_wav(pcm, tts.sample_rate))
    print(f"{out}  {tts.voice}  首块 {first:.2f}s  整句 {time.monotonic() - t0:.2f}s  "
          f"音频 {len(pcm) / 2 / tts.sample_rate:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
