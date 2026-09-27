#!/usr/bin/env python3
"""假设备：完整模拟一台跑 v2.2.4 固件的星智 1.54(ML307)。

流程与真机一致：
  POST /xiaozhi/ota/ (7 个头 + 设备自述 JSON)  → 拿 websocket.url/token/version
  WS 握手（Authorization / Protocol-Version / Device-Id / Client-Id）
  → 上行 hello（features.mcp=true, audio_params 16k/60ms）
  → 等服务端 hello（10 s 内，取 session_id 与下行 audio_params）
  → listen start → 上传 16k/60ms Opus 帧 → listen stop
  → 收 stt / tts / mcp，音频帧解码拼回 wav 存 server/tmp/

用法：
  python tools/fake_device.py                       # 端到端跑一轮
  python tools/fake_device.py --text "你今天怎么样"  # 换测试语音内容
  python tools/fake_device.py --wav path.wav        # 用现成 wav 当"用户说的话"
  python tools/fake_device.py --abort-after 1.5     # 收到音频 1.5 s 后发 abort 测打断
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

import aiohttp  # noqa: E402

from app import protocol  # noqa: E402
from app.audio import (  # noqa: E402
    OpusDecoder,
    OpusEncoder,
    encode_pcm_to_opus_frames,
    pcm16_to_wav,
    resample_pcm16,
    wav_to_pcm16_mono,
)
from app.config import load_config  # noqa: E402

TMP = SERVER_DIR / "tmp"
TMP.mkdir(exist_ok=True)

FAKE_MAC = "aa:bb:cc:dd:ee:01"
FAKE_UUID = "f1e2d3c4-b5a6-4978-8899-aabbccddee01"
BOARD_NAME = "xingzhi-cube-1.54tft-ml307"
FW_VERSION = "2.2.4"

# 设备内置 MCP 工具的最小仿真（星智 1.54 ML307 无额外工具，只有 common 那几个）
DEVICE_TOOLS = [
    {
        "name": "self.get_device_status",
        "description": "Provides the real-time information of the device.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "self.audio_speaker.set_volume",
        "description": "Set the volume of the audio speaker.",
        "inputSchema": {
            "type": "object",
            "properties": {"volume": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["volume"],
        },
    },
    {
        "name": "self.screen.set_brightness",
        "description": "Set the brightness of the screen.",
        "inputSchema": {
            "type": "object",
            "properties": {"brightness": {"type": "integer"}},
            "required": ["brightness"],
        },
    },
]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [fake-device] {msg}", flush=True)


def system_info_body(mac: str) -> dict:
    """board.GetSystemInfoJson() 的形状（research §1.1）。"""
    return {
        "language": "zh-CN",
        "flash_size": 16777216,
        "minimum_free_heap_size": "4194304",
        "mac_address": mac,
        "uuid": FAKE_UUID,
        "chip_model_name": "esp32s3",
        "chip_info": {"model": 9, "cores": 2, "revision": 0, "features": 18},
        "application": {
            "name": "xiaozhi",
            "version": FW_VERSION,
            "compile_time": "2026-03-09T00:00:00Z",
            "idf_version": "v5.4",
            "elf_sha256": "0" * 64,
        },
        "partition_table": [
            {"label": "ota_0", "type": 0, "subtype": 16, "address": 65536, "size": 6291456}
        ],
        "ota": {"label": "ota_0"},
        "display": {"monochrome": False, "width": 240, "height": 240},
        "board": {
            "type": "ml307",
            "name": BOARD_NAME,
            "revision": "ML307R-DL",
            "carrier": "CHINA MOBILE",
            "csq": "21",
            "imei": "860000000000001",
            "iccid": "89860000000000000001",
            "cereg": "1",
        },
    }


async def ensure_test_wav(text: str, cfg, path: Path) -> Path:
    """用当前 TTS provider 合成一句测试语音（只在文件不存在时合成一次）。"""
    if path.exists() and path.stat().st_size > 1000:
        log(f"复用已有测试语音 {path.name}")
        return path
    from app.tts import create_tts

    tts = create_tts(cfg)
    log(f"用 {getattr(tts, 'name', 'tts')} 合成测试语音：{text!r}")
    try:
        wav, _rtt = await tts.synth(text)
    finally:
        close = getattr(tts, "close", None)
        if close:
            await close()
    if not wav:
        raise RuntimeError("测试语音合成失败")
    path.write_bytes(wav)
    log(f"测试语音已存 {path}（{len(wav)} 字节）")
    return path


async def run(args) -> int:
    cfg = load_config(require_api_key=False)
    base = args.server or f"http://127.0.0.1:{cfg.port}"
    device_id = args.device_id or FAKE_MAC

    # ---------- 0. 准备"用户说的话" ----------
    if args.wav:
        wav_path = Path(args.wav)
    else:
        wav_path = TMP / "fake_device_input.wav"
        await ensure_test_wav(args.text, cfg, wav_path)
    pcm, src_rate = wav_to_pcm16_mono(wav_path.read_bytes())
    pcm16k = resample_pcm16(pcm, src_rate, 16000)
    dur = len(pcm16k) / 2 / 16000
    log(f"上行素材 {wav_path.name}: {src_rate} Hz → 16000 Hz，{dur:.2f}s")

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as sess:
        # ---------- 1. OTA ----------
        ota_url = f"{base.rstrip('/')}/xiaozhi/ota/"
        headers = {
            "Activation-Version": "1",
            "Device-Id": device_id,
            "Client-Id": FAKE_UUID,
            "User-Agent": f"{BOARD_NAME}/{FW_VERSION}",
            "Accept-Language": "zh-CN",
            "Content-Type": "application/json",
        }
        t0 = time.monotonic()
        async with sess.post(ota_url, headers=headers, json=system_info_body(device_id)) as resp:
            body = await resp.json()
        log(f"OTA {resp.status} ({(time.monotonic()-t0)*1000:.0f} ms) → "
            f"{json.dumps(body, ensure_ascii=False)}")
        assert resp.status == 200, "OTA 必须 200"
        assert "websocket" in body and body["websocket"].get("url"), "OTA 必须带 websocket.url"
        assert "firmware" not in body, "不该回 firmware 段（会触发升级）"
        assert "activation" not in body, "不该回 activation 段（会进激活流程）"
        ws_url = body["websocket"]["url"]
        token = body["websocket"].get("token") or ""
        ver = int(body["websocket"].get("version") or 1)
        log(f"OTA 校验通过：ws={ws_url} version={ver} token={'有' if token else '空'}")

        if args.force_local_ws:  # 公网地址回来时仍连本机
            from urllib.parse import urlsplit, urlunsplit

            parts = urlsplit(ws_url)
            ws_url = urlunsplit(("ws", f"127.0.0.1:{cfg.port}", parts.path, "", ""))
            log(f"--force-local-ws：改连 {ws_url}")

        # ---------- 2. WS 握手 ----------
        ws_headers = {
            "Protocol-Version": str(ver),
            "Device-Id": device_id,
            "Client-Id": FAKE_UUID,
        }
        if token:
            ws_headers["Authorization"] = token if " " in token else f"Bearer {token}"
        t0 = time.monotonic()
        async with sess.ws_connect(ws_url, headers=ws_headers, heartbeat=None) as ws:
            log(f"WS 已连接（{(time.monotonic()-t0)*1000:.0f} ms）")

            # ---------- 3. 上行 hello ----------
            await ws.send_str(json.dumps({
                "type": "hello",
                "version": ver,
                "features": {"mcp": True},
                "transport": "websocket",
                "audio_params": {
                    "format": "opus", "sample_rate": 16000,
                    "channels": 1, "frame_duration": 60,
                },
            }, ensure_ascii=False))
            log("→ hello 已发，等服务端 hello（10 s 上限）")

            hello_deadline = time.monotonic() + 10
            server_hello = None
            while time.monotonic() < hello_deadline:
                msg = await asyncio.wait_for(ws.receive(), timeout=hello_deadline - time.monotonic())
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                if data.get("type") == "hello":
                    server_hello = data
                    break
            assert server_hello, "10 s 内没收到服务端 hello"
            log(f"← hello {json.dumps(server_hello, ensure_ascii=False)}")
            assert server_hello.get("transport") == "websocket", "transport 必须逐字 websocket"
            session_id = server_hello.get("session_id") or ""
            assert session_id, "服务端必须分配 session_id"
            ap = server_hello.get("audio_params") or {}
            down_rate = int(ap.get("sample_rate") or 24000)
            down_frame_ms = int(ap.get("frame_duration") or 60)
            log(f"服务端 hello 校验通过：session_id={session_id} 下行 {down_rate} Hz/{down_frame_ms} ms")

            state = {
                "stt": None, "sentences": [], "tts_started": False, "tts_stopped": False,
                "audio_frames": 0, "pcm": bytearray(), "first_audio_at": None,
                "t_listen_stop": None, "mcp_initialize": False, "mcp_tools_list": False,
                "order_ok": True, "order_problem": "",
            }
            decoder = OpusDecoder(down_rate, 1)
            down_frame_samples = down_rate * down_frame_ms // 1000

            async def reader():
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        mtype = data.get("type")
                        if mtype == "mcp":
                            await handle_mcp(ws, data, session_id, state)
                            continue
                        log(f"← {msg.data[:300]}")
                        if mtype == "stt":
                            state["stt"] = data.get("text")
                        elif mtype == "tts":
                            st = data.get("state")
                            if st == "start":
                                state["tts_started"] = True
                            elif st == "sentence_start":
                                if not state["tts_started"]:
                                    state["order_ok"] = False
                                    state["order_problem"] = "sentence_start 早于 tts start"
                                state["sentences"].append(data.get("text"))
                            elif st == "stop":
                                state["tts_stopped"] = True
                                return
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        if not state["tts_started"]:
                            state["order_ok"] = False
                            state["order_problem"] = "音频帧早于 tts start（真机会静音）"
                        if state["first_audio_at"] is None:
                            state["first_audio_at"] = time.monotonic()
                        payload = protocol.unpack_audio(msg.data, ver)
                        try:
                            state["pcm"].extend(decoder.decode(payload, down_frame_samples))
                        except Exception as e:
                            log(f"下行 Opus 解码失败: {e}")
                        state["audio_frames"] += 1
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return

            reader_task = asyncio.create_task(reader())

            # ---------- 4. listen start + 上传音频 ----------
            await ws.send_str(json.dumps(
                {"session_id": session_id, "type": "listen", "state": "start", "mode": "manual"}
            ))
            log("→ listen start (mode=manual)")

            enc = OpusEncoder(16000, 1)
            frames = encode_pcm_to_opus_frames(pcm16k, enc, 960)  # 60 ms @16k = 960 样本
            log(f"→ 上传 {len(frames)} 个 60 ms Opus 帧（{dur:.2f}s）")
            t_send = time.monotonic()
            for i, f in enumerate(frames):
                await ws.send_bytes(protocol.pack_audio(f, ver))
                target = t_send + (i + 1) * 0.06
                delay = target - time.monotonic()
                if delay > 0 and not args.fast_upload:
                    await asyncio.sleep(delay)

            await ws.send_str(json.dumps(
                {"session_id": session_id, "type": "listen", "state": "stop"}
            ))
            state["t_listen_stop"] = time.monotonic()
            log("→ listen stop，等 stt / tts")

            if args.abort_after:
                async def do_abort():
                    while state["first_audio_at"] is None:
                        await asyncio.sleep(0.05)
                    await asyncio.sleep(args.abort_after)
                    await ws.send_str(json.dumps(
                        {"session_id": session_id, "type": "abort",
                         "reason": "wake_word_detected"}))
                    log(f"→ abort（首帧音频后 {args.abort_after}s）")
                asyncio.create_task(do_abort())

            try:
                await asyncio.wait_for(reader_task, timeout=args.timeout)
            except asyncio.TimeoutError:
                reader_task.cancel()
                log(f"!! 等待超时（{args.timeout}s）")

            await ws.close()

    # ---------- 5. 汇总 ----------
    out_wav = TMP / f"fake_device_reply_{time.strftime('%Y%m%d-%H%M%S')}.wav"
    if state["pcm"]:
        out_wav.write_bytes(pcm16_to_wav(bytes(state["pcm"]), down_rate))
        played = len(state["pcm"]) / 2 / down_rate
        log(f"下行音频已拼成 wav：{out_wav}（{state['audio_frames']} 帧，{played:.2f}s）")
    else:
        log("!! 没收到任何下行音频")

    print("\n================ 端到端结果 ================")
    print(f"STT 文本            : {state['stt']!r}")
    print(f"TTS 句子（{len(state['sentences'])} 句）  : {state['sentences']}")
    print(f"tts start / stop    : {state['tts_started']} / {state['tts_stopped']}")
    print(f"消息顺序            : {'OK' if state['order_ok'] else '错！' + state['order_problem']}")
    print(f"下行音频帧          : {state['audio_frames']}")
    if state["first_audio_at"] and state["t_listen_stop"]:
        print(f"首句音频延迟        : {state['first_audio_at'] - state['t_listen_stop']:.2f}s"
              "（从 listen stop 到第一帧音频，含 ASR+LLM+TTS）")
    print(f"下行 wav            : {out_wav if state['pcm'] else '(无)'}")
    print(f"MCP initialize/list : {state['mcp_initialize']} / {state['mcp_tools_list']}")
    print("===========================================\n")

    ok = (
        state["stt"]
        and state["tts_started"]
        and state["tts_stopped"]
        and state["audio_frames"] > 0
        and state["order_ok"]
    )
    if args.abort_after:
        ok = bool(state["stt"] and state["tts_started"] and state["audio_frames"] > 0)
    log("端到端 " + ("通过 ✅" if ok else "失败 ❌"))
    return 0 if ok else 1


async def handle_mcp(ws, data: dict, session_id: str, state: dict) -> None:
    """设备是 MCP server：收到 initialize / tools/list 就按固件的形状回 result。"""
    payload = data.get("payload") or {}
    method = payload.get("method")
    req_id = payload.get("id")
    log(f"← mcp {method} id={req_id}")
    if method == "initialize":
        state["mcp_initialize"] = True
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": BOARD_NAME, "version": FW_VERSION},
        }
    elif method == "tools/list":
        state["mcp_tools_list"] = True
        result = {"tools": DEVICE_TOOLS}
    elif method and method.startswith("notifications/"):
        return
    else:
        await ws.send_str(json.dumps({
            "session_id": session_id, "type": "mcp",
            "payload": {"jsonrpc": "2.0", "id": req_id,
                        "error": {"message": f"unknown method {method}"}},
        }, ensure_ascii=False))
        return
    await ws.send_str(json.dumps({
        "session_id": session_id, "type": "mcp",
        "payload": {"jsonrpc": "2.0", "id": req_id, "result": result},
    }, ensure_ascii=False))
    log(f"→ mcp {method} result 已回")


def main() -> int:
    ap = argparse.ArgumentParser(description="小智假设备端到端测试器")
    ap.add_argument("--server", default="", help="OTA 基地址，默认 http://127.0.0.1:<XZ_PORT>")
    ap.add_argument("--device-id", default="",
                    help="Device-Id(MAC)，默认用内置假 MAC；测白名单时填真机 MAC")
    ap.add_argument("--text", default="你好，帮我用一句话介绍一下你自己。",
                    help="要合成成测试语音的中文句子")
    ap.add_argument("--wav", default="", help="直接用这个 wav 当用户语音")
    ap.add_argument("--timeout", type=float, default=180.0, help="等服务端回完的秒数")
    ap.add_argument("--fast-upload", action="store_true", help="不按实时节奏，尽快上传音频")
    ap.add_argument("--abort-after", type=float, default=0.0,
                    help="收到首帧音频后 N 秒发 abort（测打断）")
    ap.add_argument("--force-local-ws", action="store_true",
                    help="忽略 OTA 回的 ws 主机名，强制连本机")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
