"""假盒子：在 EX10 上模拟真设备把整条链路走一遍，量「说完 → 听到第一个字」。

用法（EX10 部署目录里，token 从 .env 读，不打印）：
  .venv/bin/python tools/fake_box.py [--wait-close] 句子1 句子2 ...
每个句子是一个 wav（16 kHz 单声道），或 `say:文字`（现场用 Edge 合成成 16 kHz）。
句子之间照真盒子的样子：等服务端 tts stop（或 8 秒没动静）再发下一句的 listen start。
--wait-close：最后一句之后一直送静音，看服务端多久挂断（只听用户的窗口）。

Device-Id 用白名单里第一个（真盒子的 MAC），测完要 ./run.sh restart 或重新部署，
把这次测试留在内存里的上下文清掉，免得真盒子接上。
"""
from __future__ import annotations

import argparse
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
from app.audio import OpusEncoder  # noqa: E402
from app.config import load_config  # noqa: E402

SR = 16000
FRAME_MS = 60
FRAME = SR * FRAME_MS // 1000
SILENCE = b"\x00\x00" * FRAME


async def load_pcm(spec: str) -> bytes:
    if spec.startswith("say:"):
        from app.tts.edge_tts import EdgeTTS

        from app.audio import resample_pcm16

        tts = EdgeTTS()  # Edge 只出 24 kHz，自己降到 16 kHz
        try:
            pcm = b"".join([c async for c in tts.synth_stream(spec[4:])])
        finally:
            await tts.close()
        return resample_pcm16(pcm, tts.sample_rate, SR)
    with wave.open(spec, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, spec
        return w.readframes(w.getnframes())


class Box:
    def __init__(self, ws, ver: int):
        self.ws = ws
        self.ver = ver
        self.enc = OpusEncoder(SR, 1)
        self.events: list[tuple[float, str]] = []
        self.tts_state = "idle"
        self.first_audio: float | None = None
        self.stop_evt = asyncio.Event()
        self.closed_at: float | None = None

    async def send_pcm(self, pcm: bytes) -> None:
        """按 60 ms 实时节奏上行，跟真盒子一样。"""
        t0 = time.monotonic()
        for i in range(0, len(pcm), FRAME * 2):
            chunk = pcm[i:i + FRAME * 2].ljust(FRAME * 2, b"\x00")
            if self.ws.closed:
                return
            await self.ws.send_bytes(protocol.pack_audio(self.enc.encode(chunk, FRAME), self.ver))
            n = i // (FRAME * 2) + 1
            delay = t0 + n * FRAME_MS / 1000 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    async def reader(self) -> None:
        async for msg in self.ws:
            now = time.monotonic()
            if msg.type == aiohttp.WSMsgType.TEXT:
                d = json.loads(msg.data)
                t = d.get("type")
                if t == "tts":
                    st = d.get("state")
                    self.events.append((now, f"tts {st} {d.get('text', '')}".strip()))
                    if st == "start":
                        self.tts_state = "speaking"
                    elif st == "stop":
                        self.tts_state = "idle"
                        self.stop_evt.set()
                elif t == "stt":
                    self.events.append((now, f"stt {d.get('text')}"))
                elif t in ("hello", "llm"):
                    pass
            elif msg.type == aiohttp.WSMsgType.BINARY:
                if self.first_audio is None:
                    self.first_audio = now
        self.closed_at = time.monotonic()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("utts", nargs="+")
    ap.add_argument("--wait-close", action="store_true")
    ap.add_argument("--url", default="")
    args = ap.parse_args()

    cfg = load_config(require_api_key=False)
    url = args.url or f"ws://127.0.0.1:{cfg.port}/xiaozhi/v1/"
    dev = sorted(cfg.allowed_device_ids)[0] if cfg.allowed_device_ids else "fake-box"
    headers = {"Device-Id": dev, "Client-Id": str(uuid.uuid4()),
               "Protocol-Version": str(cfg.protocol_version)}
    if cfg.token:
        headers["Authorization"] = cfg.token if " " in cfg.token else f"Bearer {cfg.token}"

    pcms = [await load_pcm(u) for u in args.utts]
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(url, headers=headers) as ws:
            box = Box(ws, cfg.protocol_version)
            await ws.send_str(json.dumps({
                "type": "hello", "version": cfg.protocol_version, "transport": "websocket",
                "features": {"mcp": False},
                "audio_params": {"format": "opus", "sample_rate": SR, "channels": 1,
                                 "frame_duration": FRAME_MS}}))
            rd = asyncio.create_task(box.reader())
            await asyncio.sleep(0.3)
            for spec, pcm in zip(args.utts, pcms):
                box.events.clear()
                box.first_audio = None
                box.stop_evt.clear()
                await ws.send_str(json.dumps({"type": "listen", "state": "start", "mode": "auto"}))
                await box.send_pcm(SILENCE * 5 + pcm)
                t_end = time.monotonic()  # 人说完的时刻
                # 说完以后盒子还在收音：送静音直到服务端开口或 8 秒没反应
                waiter = asyncio.create_task(box.stop_evt.wait())
                t_lim = t_end + 45   # 新闻类一答能念 20 多秒，25 s 会串到下一句
                while not waiter.done() and time.monotonic() < t_lim and not ws.closed:
                    if box.tts_state == "speaking":
                        await asyncio.sleep(0.06)   # 没有回声消除：说话时不上行
                        continue
                    if box.first_audio is None and time.monotonic() - t_end > 8 and \
                            not any(e[1].startswith("tts start") for e in box.events):
                        break
                    await box.send_pcm(SILENCE)
                waiter.cancel()
                name = spec if spec.startswith("say:") else Path(spec).name
                print(f"\n== {name}（{len(pcm) / 2 / SR:.1f}s）")
                for t, e in box.events:
                    print(f"  +{t - t_end:5.2f}s  {e}")
                if box.first_audio:
                    print(f"  说完 → 第一个音频帧：{box.first_audio - t_end:.2f}s")
                else:
                    print("  没有下发音频")
                if ws.closed:
                    break
            if args.wait_close and not ws.closed:
                t0 = time.monotonic()
                while not ws.closed and time.monotonic() - t0 < 40:
                    await box.send_pcm(SILENCE)
                print(f"\n== 最后一句之后送静音：{'服务端 %.1fs 后挂断' % (time.monotonic() - t0) if ws.closed else '40s 内没挂断'}")
            if not ws.closed:
                await ws.close()
            rd.cancel()


if __name__ == "__main__":
    asyncio.run(main())
