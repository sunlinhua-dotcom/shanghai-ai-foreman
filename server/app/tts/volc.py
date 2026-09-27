"""豆包语音（火山引擎）语音合成模型 2.0：HTTP 单向流式 V3。

POST https://openspeech.bytedance.com/api/v3/tts/unidirectional
- 鉴权用新版控制台的 API Key：X-Api-Key + X-Api-Resource-Id。seed-tts-2.0 只认
  `*_uranus_bigtts` 这类 2.0 音色，配错报 resource ID is mismatched。
- 回包一行一个 JSON：{"code":0,"data":"<base64 音频>"}…，最后 {"code":20000000} 收尾；
  其它非 0 的 code 是出错，message 里有原因。
- 直接要 pcm + 下行采样率（24k），不解码、不重采样。
- 方言、语气靠语音指令 additions.context_texts（只认第一条，不计费），如 ["用上海话说"]。
  additions 必须是 JSON **字符串**，传对象会被静默忽略。
音色表 https://www.volcengine.com/docs/6561/1257544 ：小何 / Vivi / 云舟 / 小天 2.0
会粤语、上海、河南、北京、天津、四川、陕西、东北八种方言。
接口文档 https://www.volcengine.com/docs/6561/1598757 。
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
import uuid as uuid_mod

import aiohttp

from ..audio import pcm16_to_wav

log = logging.getLogger("xz.tts")

URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
DEFAULT_VOICE = "zh_female_xiaohe_uranus_bigtts"  # 小何 2.0
DONE = 20000000
_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

# 云端允许并发；一轮里「念第 N 句时预取第 N+1 句」最多同时 2 路，跟 Edge 一样封顶
_VOLC_SEMAPHORE = asyncio.Semaphore(2)


async def _lines(resp: aiohttp.ClientResponse):
    """流式回包按 \\n 切行。不用 resp.content 的逐行迭代：一行 base64 音频能过 64 KB 行长上限。"""
    buf = b""
    async for piece in resp.content.iter_any():
        buf += piece
        *lines, buf = buf.split(b"\n")
        for line in lines:
            yield line
    if buf.strip():
        yield buf


class TTSError(RuntimeError):
    def __init__(self, msg: str, transient: bool = True):
        super().__init__(msg)
        # False = key / 开通 / 额度这类问题，重试没用
        self.transient = transient


def _is_volc_voice(v: str) -> bool:
    """豆包音色 ID 是 zh_female_xxx_uranus_bigtts / S_xxx 这种；茉莉、zh-CN-…Neural 是 Edge 的。"""
    return bool(v) and v.isascii() and "_" in v and "-" not in v


class VolcTTS:
    name = "volc"

    def __init__(
        self,
        api_key: str,
        voice: str = "",
        resource_id: str = "seed-tts-2.0",
        instruction: str = "",
        speech_rate: int = 0,
        sample_rate: int = 24000,
        timeout: float = 20.0,
    ):
        self.api_key = api_key
        self.voice = voice if _is_volc_voice(voice) else DEFAULT_VOICE
        self.resource_id = resource_id
        self.instruction = instruction.strip()
        self.speech_rate = max(-50, min(100, speech_rate))
        self.sample_rate = sample_rate if sample_rate in _RATES else 24000
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # trust_env=False：国内接口直连，不吃代理环境变量；连接留 60 s 给下一句复用
            self._session = aiohttp.ClientSession(
                trust_env=False,
                connector=aiohttp.TCPConnector(keepalive_timeout=60, ttl_dns_cache=300),
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=self.timeout),
            )
        return self._session

    def _body(self, text: str) -> dict:
        additions = {"disable_markdown_filter": True}
        if self.instruction:
            additions["context_texts"] = [self.instruction]
        return {
            "user": {"uid": "xiaozhi"},
            "req_params": {
                "text": text,
                "speaker": self.voice,
                "audio_params": {
                    "format": "pcm",
                    "sample_rate": self.sample_rate,
                    "speech_rate": self.speech_rate,
                },
                "additions": json.dumps(additions, ensure_ascii=False),
            },
        }

    def _headers(self) -> dict:
        return {
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid_mod.uuid4()),
            # 收尾那行带 usage.text_words，日志里看得到这句扣了多少字
            "X-Control-Require-Usage-Tokens-Return": "text_words",
        }

    # ---------- 主路径：流式 ----------

    async def synth_stream(self, text: str, prewarm=None):
        """一句话 → 流式 yield PCM16 单声道块（self.sample_rate）。

        prewarm 只为和 Edge 同一个调用口径，用不上（HTTP 连接池自己复用）。
        超时 / 断线 / 5xx 这类临时问题，还没出声就再试一次；key、开通、额度问题不重试。
        """
        if not text.strip():
            return
        got_any = False
        try:
            async for chunk in self._stream_once(text):
                got_any = True
                yield chunk
            return
        except TTSError as e:
            if got_any or not e.transient:
                raise
            log.warning("豆包这句没出音频（%s），重试一次", e)
        async for chunk in self._stream_once(text):
            yield chunk

    async def _stream_once(self, text: str):
        async with _VOLC_SEMAPHORE:
            sess = await self._ensure_session()
            t0 = time.monotonic()
            got = 0
            carry = b""  # 块边界可能劈开一个 16 位采样，余下的 1 字节留给下一块
            try:
                async with sess.post(URL, json=self._body(text), headers=self._headers()) as resp:
                    if resp.status != 200:
                        detail = (await resp.text())[:200]
                        raise TTSError(
                            f"豆包 HTTP {resp.status}: {detail}",
                            transient=resp.status >= 500 or resp.status == 429,
                        )
                    async with contextlib.aclosing(_lines(resp)) as lines:
                        async for line in lines:
                            pcm, done = self._parse(line, text)
                            pcm, carry = carry + pcm, b""
                            if len(pcm) % 2:
                                pcm, carry = pcm[:-1], pcm[-1:]
                            if pcm:
                                if not got:
                                    log.info("豆包 首块 %.2fs（%d 字）%r",
                                             time.monotonic() - t0, len(text), text[:20])
                                got += len(pcm)
                                yield pcm
                            if done:
                                break
            except (TimeoutError, aiohttp.ClientError) as e:
                raise TTSError(f"豆包请求失败: {e}") from e
            if not got:
                raise TTSError("豆包没有返回音频")
            log.info("豆包 整句 %.2fs（%d 字，%.1f s 音频）",
                     time.monotonic() - t0, len(text), got / 2 / self.sample_rate)

    def _parse(self, line: bytes, text: str) -> tuple[bytes, bool]:
        """一行回包 → (PCM, 是否收尾)。"""
        line = line.strip()
        if not line:
            return b"", False
        try:
            msg = json.loads(line)
        except ValueError:
            log.warning("豆包回包不是 JSON：%r", line[:80])
            return b"", False
        code = msg.get("code", 0)
        if code == DONE:
            usage = msg.get("usage") or {}
            if usage:
                log.info("豆包 用量 %s（%r）", usage, text[:20])
            return b"", True
        if code != 0:
            # 45000000 音色没授权 / 服务没开通；55000000 服务端错误；并发超限只是一时的
            message = str(msg.get("message", ""))[:200]
            transient = code >= 50000000 or "concurren" in message
            raise TTSError(f"豆包 {code}: {message}", transient=transient)
        data = msg.get("data")
        return (base64.b64decode(data) if data else b""), False

    # ---------- 整句版（fake_device 造测试语音 / 兜底） ----------

    async def synth(self, text: str) -> tuple[bytes, float]:
        t0 = time.monotonic()
        chunks = [chunk async for chunk in self.synth_stream(text)]
        pcm = b"".join(chunks)
        if not pcm:
            return b"", 0.0
        return pcm16_to_wav(pcm, self.sample_rate), time.monotonic() - t0

    async def health(self) -> dict:
        """真合成两个字（扣 2 个字符），验证 key、开通和出网。"""
        agen = self.synth_stream("你好")
        t0 = time.monotonic()
        try:
            async for _chunk in agen:
                return {
                    "ok": True,
                    "voice": self.voice,
                    "instruction": self.instruction,
                    "sample_rate": self.sample_rate,
                    "first_chunk_s": round(time.monotonic() - t0, 2),
                }
            raise TTSError("豆包没有返回音频")
        finally:
            await agen.aclose()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class FallbackTTS:
    """主引擎这句还没出声就失败，改用备胎（Edge 茉莉）念，盒子不至于哑掉。

    key 不对、没开通、额度用完这类问题，10 分钟内直接走备胎，不每句都白等一次。
    """

    SKIP_S = 600

    def __init__(self, primary, backup):
        self.primary = primary
        self.backup = backup
        self.name = primary.name
        self.sample_rate = primary.sample_rate
        self._skip_until = 0.0

    async def synth_stream(self, text: str, prewarm=None):
        if time.monotonic() >= self._skip_until:
            got_any = False
            try:
                async for chunk in self.primary.synth_stream(text):
                    got_any = True
                    yield chunk
                return
            except Exception as e:
                if got_any:
                    raise
                if not getattr(e, "transient", True):
                    self._skip_until = time.monotonic() + self.SKIP_S
                    log.warning("%s 用不了（%s），%d 分钟内改用 %s",
                                self.primary.name, e, self.SKIP_S // 60, self.backup.name)
                else:
                    log.warning("%s 这句没念出来（%s），这句改用 %s",
                                self.primary.name, e, self.backup.name)
        async for chunk in self.backup.synth_stream(text):
            yield chunk

    async def synth(self, text: str) -> tuple[bytes, float]:
        t0 = time.monotonic()
        chunks = [chunk async for chunk in self.synth_stream(text)]
        pcm = b"".join(chunks)
        if not pcm:
            return b"", 0.0
        return pcm16_to_wav(pcm, self.sample_rate), time.monotonic() - t0

    async def health(self) -> dict:
        try:
            return await self.primary.health()
        except Exception as e:
            info = await self.backup.health()
            info["fallback_because"] = str(e)[:200]
            return info

    async def close(self) -> None:
        for tts in (self.primary, self.backup):
            try:
                await tts.close()
            except Exception:  # noqa: BLE001
                pass
