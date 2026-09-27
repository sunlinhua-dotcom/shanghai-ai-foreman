"""小智 WebSocket 协议的帧编解码与消息构造。

依据 research/2026-09-06-selfhost-server-protocol.md §1.3：
  v1 无帧头，binary payload 即裸 Opus              (protocol.h:17-31, websocket_protocol.cc:56,139-146)
  v2 16 字节大端头 version/type/reserved/timestamp/size
  v3 4  字节大端头 type/reserved/size
接收侧固件不检查 type 字节，binary 一律当音频 → 控制消息必须走 text 帧。
"""
from __future__ import annotations

import struct
import time

BINARY_TYPE_OPUS = 0
BINARY_TYPE_JSON = 1


def pack_audio(payload: bytes, version: int, timestamp_ms: int | None = None) -> bytes:
    if version == 1:
        return payload
    if version == 2:
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000) & 0xFFFFFFFF
        return struct.pack(">HHIII", 2, BINARY_TYPE_OPUS, 0, ts, len(payload)) + payload
    if version == 3:
        return struct.pack(">BBH", BINARY_TYPE_OPUS, 0, len(payload)) + payload
    raise ValueError(f"unsupported protocol version {version}")


def unpack_audio(frame: bytes, version: int) -> bytes:
    """从设备上行的 binary 帧里取出 Opus payload。"""
    if version == 1:
        return frame
    if version == 2:
        if len(frame) < 16:
            raise ValueError("v2 frame shorter than 16-byte header")
        _ver, _type, _rsv, _ts, size = struct.unpack(">HHIII", frame[:16])
        return frame[16 : 16 + size] if size else frame[16:]
    if version == 3:
        if len(frame) < 4:
            raise ValueError("v3 frame shorter than 4-byte header")
        _type, _rsv, size = struct.unpack(">BBH", frame[:4])
        return frame[4 : 4 + size] if size else frame[4:]
    raise ValueError(f"unsupported protocol version {version}")


# ---------- 下行 text 消息 ----------

def msg_hello(session_id: str, sample_rate: int, frame_ms: int) -> dict:
    """服务端 hello。transport 必须逐字 "websocket"（websocket_protocol.cc:228-254）。
    设备只读 audio_params.sample_rate / frame_duration；format/channels 会被忽略。"""
    return {
        "type": "hello",
        "transport": "websocket",
        "session_id": session_id,
        "audio_params": {
            "format": "opus",
            "sample_rate": sample_rate,
            "channels": 1,
            "frame_duration": frame_ms,
        },
    }


def msg_stt(text: str, session_id: str) -> dict:
    return {"type": "stt", "text": text, "session_id": session_id}


def msg_tts_start(session_id: str) -> dict:
    return {"type": "tts", "state": "start", "session_id": session_id}


def msg_tts_sentence_start(text: str, session_id: str) -> dict:
    return {"type": "tts", "state": "sentence_start", "text": text, "session_id": session_id}


def msg_tts_stop(session_id: str) -> dict:
    return {"type": "tts", "state": "stop", "session_id": session_id}


def msg_llm_emotion(emotion: str, session_id: str) -> dict:
    """情绪在 llm 里，没有独立 emotion type（application.cc:558-564）。"""
    return {"type": "llm", "emotion": emotion, "session_id": session_id}


def msg_mcp(payload: dict, session_id: str) -> dict:
    return {"type": "mcp", "payload": payload, "session_id": session_id}
