"""长按 BOOT 换声音的服务端探针（9/27）：学固件 2.2.4.15 长按时发的 listen detect text=__voice__，
把服务端念回来的话存成 wav，看每种声音都能出声。

用法（EX10 部署目录里，token 从 .env 读，不打印）：
  .venv/bin/python tools/voice_probe.py [次数=4] [输出目录=/tmp/xz-voice-probe]
每跑一次声音轮到下一种，跑满一圈（现在是 3 次）回到原来那种。hello 里 mcp=false：服务端不做 MCP 握手，
也就不会把「催盒子重启升级」那一次用掉。测完 sudo systemctl restart xiaozhi-server，把测试留在内存里的上下文清掉
（服务归 systemd 管；别用 ./run.sh restart，它会另起一个抢不到端口的进程，旧进程照跑，9/27 踩过）。
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
import wave
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import protocol  # noqa: E402
from app.audio import OpusDecoder  # noqa: E402
from app.config import load_config  # noqa: E402


async def once(cfg, url: str, headers: dict, out: Path) -> None:
    sr = cfg.downlink_sample_rate
    frame = sr * cfg.frame_ms // 1000
    dec = OpusDecoder(sr, 1)
    pcm = bytearray()
    said: list[str] = []
    first = None
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(url, headers={**headers, "Client-Id": str(uuid.uuid4())}) as ws:
            await ws.send_str(json.dumps({
                "type": "hello", "version": cfg.protocol_version, "transport": "websocket",
                "features": {"mcp": False},
                "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1,
                                 "frame_duration": cfg.frame_ms}}))
            async for msg in ws:  # 播报念完服务端自己挂断
                if msg.type == aiohttp.WSMsgType.TEXT:
                    d = json.loads(msg.data)
                    if d.get("type") == "hello":
                        await ws.send_str(json.dumps({"type": "listen", "state": "detect", "text": "__voice__"}))
                    elif d.get("type") == "tts" and d.get("state") == "sentence_start":
                        said.append(d.get("text", ""))
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    if first is None:
                        first = time.monotonic() - t0
                    pcm += dec.decode(protocol.unpack_audio(msg.data, cfg.protocol_version), frame)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(pcm))
    print(f"{out.name}: 首个声音 {first if first is None else round(first, 2)}s，"
          f"音频 {len(pcm) / 2 / sr:.1f}s，挂断 {time.monotonic() - t0:.1f}s | {' '.join(said)}", flush=True)


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/xz-voice-probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(require_api_key=False)
    url = f"ws://127.0.0.1:{cfg.port}/xiaozhi/v1/"
    dev = sorted(cfg.allowed_device_ids)[0] if cfg.allowed_device_ids else "fake-box"
    headers = {"Device-Id": dev, "Protocol-Version": str(cfg.protocol_version)}
    if cfg.token:
        headers["Authorization"] = cfg.token if " " in cfg.token else f"Bearer {cfg.token}"
    for i in range(n):
        await asyncio.wait_for(once(cfg, url, headers, out_dir / f"voice_{i + 1}.wav"), timeout=60)
        await asyncio.sleep(1.5)


if __name__ == "__main__":
    asyncio.run(main())
