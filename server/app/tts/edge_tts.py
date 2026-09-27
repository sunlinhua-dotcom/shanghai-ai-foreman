"""微软 Edge「大声朗读」在线神经语音（免 key、免费，ClaudeVoice 同款）。

照 ClaudeVoice 的 edge-tts.js 手写 aiohttp 版；**没有**用 pip 的 edge-tts：
那个包 outputFormat 写死成 mp3 且不支持连接预热，这里除了 mp3 还做了
「先握手发 speech.config、等 LLM 出首句再发 ssml」的预热（EX10 实测省 ~0.9s
握手，闲置 5s 连接不掉，ssml 后 0.2–0.5s 出首块）。

格式现实（2026-09-06 Mac/EX10 双端实测）：readaloud 端点**只对 mp3 出音频**，
riff/raw PCM 一律 turn.start+response 后无音频关连接（ClaudeVoice 的 wav 分支
其实也坏了）。所以这里请求 audio-24khz-48kbitrate-mono-mp3，收一块 mp3 就用
PyAV 增量解一块成 PCM16 单声道，24k 正好等于下行采样率，不用重采样。

反滥用要求（2026 起缺一即 403）：
- Sec-MS-GEC = SHA256(取整到 5 分钟的 Windows FILETIME + TrustedClientToken) 大写 hex
- Sec-MS-GEC-Version 与 UA 的 Chromium 主版本一致（143）
- Cookie: muid=<32位大写hex>（本地随机生成，无需向 bing 取）
- 403 时用响应 Date 校准时钟后重试一次（edge-tts.js 的做法）

并发：云端允许并发（voxtts 的进程级串行锁在这里不适用），
但别超过 2 路同时，免得被限流 —— 进程级 Semaphore(2)。
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import time
import uuid as uuid_mod
from email.utils import parsedate_to_datetime
from xml.sax.saxutils import escape

import aiohttp

from ..audio import pcm16_to_wav

log = logging.getLogger("xz.tts")

TRUSTED_CLIENT_TOKEN = "6A5AA1D4EAFF4E9FB37E23D68491D6F4"  # Edge 浏览器 / edge-tts 里的公开常量，不是密钥 gitleaks:allow
GEC_VERSION = "1-143.0.3650.75"
CHROMIUM = "143.0.0.0"
WSS_URL = "wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1"

# 中文显示名 → Edge 神经语音（与 ClaudeVoice 的 edge-tts.js 同一张表）
VOICE_MAP = {
    "茉莉": "zh-CN-XiaoxiaoNeural",  # f
    "冰糖": "zh-CN-XiaoyiNeural",    # f
    "白桦": "zh-CN-YunxiNeural",     # m
    "苏打": "zh-CN-YunyangNeural",   # m
}
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"  # 茉莉

_VOICE_RE = re.compile(r"[a-z]{2}-[A-Z]{2}-\w+Neural")

# 端点支持的 mp3 输出（解码后即该采样率的 PCM16 单声道）
_MP3_FORMATS = {
    16000: "audio-16khz-32kbitrate-mono-mp3",
    24000: "audio-24khz-48kbitrate-mono-mp3",
    48000: "audio-48khz-96kbitrate-mono-mp3",
}

# 进程级：所有会话共用，最多 2 路并发生成（第 N 句发送时预取第 N+1 句刚好用满）
_EDGE_SEMAPHORE = asyncio.Semaphore(2)


class TTSError(RuntimeError):
    pass


def resolve_voice(v: str) -> str:
    """中文别名 / xx-XX-…Neural 原名都可以；不认识的回落茉莉。"""
    if not v:
        return DEFAULT_VOICE
    if v in VOICE_MAP:
        return VOICE_MAP[v]
    if _VOICE_RE.fullmatch(v):
        return v
    return DEFAULT_VOICE


def _sec_ms_gec(skew_s: float = 0.0) -> str:
    ticks = time.time() + skew_s + 11644473600  # unix 秒 → Windows FILETIME 的秒
    ticks -= ticks % 300                        # 取整到 5 分钟
    ticks *= 1e7                                # 秒 → 100 纳秒
    return hashlib.sha256(f"{int(ticks)}{TRUSTED_CLIENT_TOKEN}".encode("ascii")).hexdigest().upper()


def _timestamp() -> str:
    return time.strftime("%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)", time.gmtime())


def _ssml(text: str, voice: str) -> str:
    return (
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='zh-CN'>"
        f"<voice name='{voice}'><prosody pitch='+0Hz' rate='+0%' volume='+0%'>"
        f"{escape(text)}</prosody></voice></speak>"
    )


class _Mp3Decoder:
    """PyAV 增量 mp3 → PCM16。一块进一块出，喂给流式 Opus 编码。"""

    def __init__(self):
        try:
            import av
        except ImportError as e:
            raise TTSError("缺 PyAV（mp3 解码用）：.venv/bin/pip install av") from e
        self._av = av
        self.ctx = av.CodecContext.create("mp3", "r")

    def feed(self, data: bytes) -> bytes:
        out: list[bytes] = []
        try:
            packets = self.ctx.parse(data)
        except self._av.error.InvalidDataError:
            return b""  # ID3 之类的非帧数据，跳过
        for pkt in packets:
            for frame in self.ctx.decode(pkt):
                out.append(bytes(frame.planes[0]))  # s16p 单声道：plane0 就是 s16 LE
        return b"".join(out)

    def flush(self) -> bytes:
        out: list[bytes] = []
        try:
            for frame in self.ctx.decode(None):
                out.append(bytes(frame.planes[0]))
        except Exception:
            pass
        return b"".join(out)


class _Prewarmed:
    """已握手并发出 speech.config 的连接，等 ssml。闲置几秒不会掉。"""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse, ts: str):
        self.ws = ws
        self.ts = ts
        self.used = False

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.ws.close()


class EdgeTTS:
    name = "edge"

    def __init__(self, voice: str = "", sample_rate: int = 24000, timeout: float = 20.0):
        self.voice = resolve_voice(voice)
        self.sample_rate = sample_rate if sample_rate in _MP3_FORMATS else 24000
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # trust_env=False：不吃系统代理环境变量
            self._session = aiohttp.ClientSession(
                trust_env=False,
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=self.timeout),
            )
        return self._session

    # ---------- 连接 ----------

    async def _open(self, sess: aiohttp.ClientSession) -> tuple[aiohttp.ClientWebSocketResponse, str]:
        """握手（403 按服务器 Date 校准时钟重试一次）+ 发 speech.config。"""
        skew = 0.0
        for attempt in (1, 2):
            url = (
                f"{WSS_URL}?TrustedClientToken={TRUSTED_CLIENT_TOKEN}"
                f"&ConnectionId={uuid_mod.uuid4().hex}"
                f"&Sec-MS-GEC={_sec_ms_gec(skew)}"
                f"&Sec-MS-GEC-Version={GEC_VERSION}"
            )
            headers = {
                "Origin": "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    f"(KHTML, like Gecko) Chrome/{CHROMIUM} Safari/537.36 Edg/{CHROMIUM}"
                ),
                "Cookie": f"muid={uuid_mod.uuid4().hex.upper()}",
                "Pragma": "no-cache",
                "Cache-Control": "no-cache",
            }
            try:
                ws = await sess.ws_connect(url, headers=headers, timeout=self._ws_timeout())
            except aiohttp.WSServerHandshakeError as e:
                if attempt == 1 and e.status == 403 and e.headers:
                    try:
                        skew = parsedate_to_datetime(e.headers.get("Date", "")).timestamp() - time.time()
                        log.warning("Edge 403，按服务器时间校准时钟（偏差 %.0fs）重试", skew)
                        continue
                    except Exception:
                        pass
                raise TTSError(f"Edge 握手失败 HTTP {e.status}") from e

            ts = _timestamp()
            cfg_msg = (
                f"X-Timestamp:{ts}\r\nContent-Type:application/json; charset=utf-8\r\n"
                "Path:speech.config\r\n\r\n"
                + json.dumps(
                    {
                        "context": {
                            "synthesis": {
                                "audio": {
                                    "metadataoptions": {
                                        "sentenceBoundaryEnabled": "false",
                                        "wordBoundaryEnabled": "false",
                                    },
                                    "outputFormat": _MP3_FORMATS[self.sample_rate],
                                }
                            }
                        }
                    }
                )
            )
            try:
                await ws.send_str(cfg_msg)
            except Exception:
                await ws.close()
                raise
            return ws, ts
        raise TTSError("Edge 连接重试用尽")  # 不可达，for 循环必 return/raise

    def _ws_timeout(self):
        try:  # aiohttp >= 3.10
            from aiohttp import ClientWSTimeout

            return ClientWSTimeout(ws_receive=self.timeout)
        except ImportError:
            return self.timeout

    @staticmethod
    def _audio_payload(data: bytes) -> bytes:
        """二进制帧：2 字节大头头长 + 头文本 + 音频。纯 metadata 帧返回空。"""
        if len(data) < 3:
            return b""
        hlen = (data[0] << 8) | data[1]
        if 2 + hlen >= len(data):
            return b""
        return data[2 + hlen :]

    async def _send_ssml(self, ws: aiohttp.ClientWebSocketResponse, ts: str, text: str) -> None:
        ssml_msg = (
            f"X-RequestId:{uuid_mod.uuid4().hex}\r\nContent-Type:application/ssml+xml\r\n"
            f"X-Timestamp:{ts}Z\r\nPath:ssml\r\n\r\n{_ssml(text, self.voice)}"
        )
        try:
            await ws.send_str(ssml_msg)
        except Exception:
            await ws.close()
            raise

    # ---------- 预热 ----------

    async def prewarm(self) -> _Prewarmed:
        """先建连接+发 speech.config，把 ~0.9s 握手藏进 ASR/LLM 等待里。"""
        sess = await self._ensure_session()
        ws, ts = await self._open(sess)
        return _Prewarmed(ws, ts)

    # ---------- 主路径：流式 ----------

    async def synth_stream(self, text: str, prewarm: _Prewarmed | None = None):
        """一句话 → 流式 yield PCM16 单声道块（self.sample_rate）。

        prewarm：用 prewarm() 拿到的现成连接（只发 ssml）；连接已死则退回常规连接。
        async generator；消费方随时弃掉（aclose/cancel）即断开 Edge 连接。
        """
        if not text.strip():
            return
        if prewarm is not None and not prewarm.used:
            prewarm.used = True
            got_any = False
            try:
                try:
                    async for chunk in self._stream_once(text, prewarm):
                        got_any = True
                        yield chunk
                    return
                except TTSError:
                    if got_any:
                        raise
                    log.warning("预热连接不可用，退回常规连接重连")
            finally:
                await prewarm.close()
        # Edge 偶尔握手成功却一个音频包都不发（日志里的「Edge 没有返回音频」），
        # 那一句就整句被丢掉。换条新连接再试一次——只在还没吐出任何音频时重试，
        # 已经出过声再重来会把前半句念两遍。
        got_any = False
        try:
            async for chunk in self._stream_once(text, None):
                got_any = True
                yield chunk
            return
        except TTSError as e:
            if got_any:
                raise
            log.warning("Edge 这句没出音频（%s），换条连接重试一次", e)
        async for chunk in self._stream_once(text, None):
            yield chunk

    async def _stream_once(self, text: str, prewarmed: _Prewarmed | None):
        async with _EDGE_SEMAPHORE:
            sess = await self._ensure_session()
            t0 = time.monotonic()
            own_conn = prewarmed is None
            ws: aiohttp.ClientWebSocketResponse | None = None
            got_mp3 = False
            try:
                if prewarmed is not None:
                    ws = prewarmed.ws
                    await self._send_ssml(ws, prewarmed.ts, text)
                    t_ssml = time.monotonic()
                else:
                    ws, ts = await self._open(sess)
                    await self._send_ssml(ws, ts, text)
                    t_ssml = time.monotonic()
                decoder = _Mp3Decoder()
                first = True
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        payload = self._audio_payload(msg.data)
                        if not payload:
                            continue
                        got_mp3 = True
                        pcm = decoder.feed(payload)
                        if pcm:
                            if first:
                                first = False
                                log.info(
                                    "Edge 首块 %.2fs（ssml 后 %.2fs，%d 字）%r",
                                    time.monotonic() - t0,
                                    time.monotonic() - t_ssml,
                                    len(text),
                                    text[:20],
                                )
                            yield pcm
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        if "Path:turn.end" in msg.data:
                            break
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        if not got_mp3:
                            raise TTSError("Edge 连接在出音频前关闭")
                        break
                tail = decoder.flush()
                if tail:
                    yield tail
                if not got_mp3:
                    raise TTSError("Edge 没有返回音频")
                log.info("Edge 整句 %.2fs（%d 字）", time.monotonic() - t0, len(text))
            except (TimeoutError, aiohttp.ClientError) as e:
                raise TTSError(f"Edge 请求失败: {e}") from e
            finally:
                if ws is not None and own_conn:
                    await ws.close()

    # ---------- 整句版（fake_device 造测试语音 / 兜底） ----------

    async def synth(self, text: str) -> tuple[bytes, float]:
        """一句话 → (wav bytes, 往返秒数)。主路径请用 synth_stream。"""
        t0 = time.monotonic()
        chunks = []
        async for chunk in self.synth_stream(text):
            chunks.append(chunk)
        pcm = b"".join(chunks)
        if not pcm:
            return b"", 0.0
        return pcm16_to_wav(pcm, self.sample_rate), time.monotonic() - t0

    async def health(self) -> dict:
        """真合成一个字验证握手+签名+出网，顺带预热 DNS。"""
        agen = self.synth_stream("你好")
        t0 = time.monotonic()
        try:
            async for _chunk in agen:
                return {
                    "ok": True,
                    "voice": self.voice,
                    "sample_rate": self.sample_rate,
                    "first_chunk_s": round(time.monotonic() - t0, 2),
                }
            raise TTSError("Edge 没有返回音频")
        finally:
            await agen.aclose()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
