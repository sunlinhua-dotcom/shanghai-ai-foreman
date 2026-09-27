"""一条 WebSocket 连接 = 一个设备会话。

状态机要点（全部来自 research/2026-09-06-selfhost-server-protocol.md）：
- 10 s 内必须回 hello，transport 逐字 "websocket"，session_id 服务端生成（§1.3）
- 上行音频恒 16 kHz / 单声道 / 60 ms Opus；binary 帧一律当音频
- 下行音频**只有设备处于 Speaking 才收**，所以顺序必须是
  tts start → (sentence_start + 音频帧)* → tts stop（application.cc:498-502）
- 设备从不主动 ping，服务端自己保活（<120 s）
- WS 下没有 goodbye（websocket_protocol.cc:78-81），这里只做兜底日志
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
import uuid
import wave

import numpy as np

from aiohttp import WSMsgType, web

from . import protocol
from .announce import WAKE_TEXT as _ANNOUNCE_WAKE
from .tts import voices
from .audio import (
    OpusDecoder,
    OpusEncoder,
    resample_pcm16,
    wav_to_pcm16_mono,
)
from .llm.claude import DIALECT_LINES, SEARCH_FILLER, SLOW_FILLER, TOOL_FILLER, dialect_line
from .vad import VadSegmenter
from .voiceprint import keep_recent
from .wu_text import fix_readings

_VOICE_WAKE = "__voice__"   # 固件 2.2.4.15 起长按 BOOT 发这个（原来长按是转屏）
# 「调」只认上海话「调只声音」那种（后面跟量词）；「调一下声音」多半是要调音量，留给 LLM
_VOICE_CMD = re.compile(r"(换|切换)(一)?(个|只|种|下)?(声音|声线|嗓音|音色)|调(一)?(个|只|种)(声音|声线|嗓音|音色)")

# 只有语气词：「嗯。」「啊。」「对。」。9/23–24 的日志里一小半「一句话」是这种，盒子每次都回一句
# 「好了」「有事随时叫我」，念的那几秒不听人说话——主人想好了要说的后半句就这么丢了
_FILLER_ONLY = re.compile(
    r"(?:嗯|啊|呃|额|哦|噢|喔|唔|呜|嘶|哎|诶|欸|哈|嘿|呀|哼|那个|这个|就是|然后|对|好|好的|行|是|是的|ok)+",
    re.I)
_OPEN_TAIL = re.compile(r"(?:然后|就是|那个|这个|还有|而且|但是|可是|所以|因为|如果|或者|还是|比如|"
                        r"呃|嗯|额|的话|以及|另外|包括)$")
_PUNCT = re.compile(r"[\s，,。.！!？?、…~～：:；;\"'“”]+")
_ASKED = re.compile(r"[？?]|[吗伐么][。.！!]?$|要不要|是不是|好不好|行不行|对不对|要伐|好伐")


def _dangling(text: str) -> str:
    """filler = 只有语气词；open = 话像没说完（逗号或「然后 / 就是」结尾）；空串 = 像句完整的话。"""
    t = (text or "").strip()
    bare = _PUNCT.sub("", t)
    if not bare:
        return ""
    if _FILLER_ONLY.fullmatch(bare):
        return "filler"
    if re.search(r"[，,、]$", t) or _OPEN_TAIL.search(bare):
        return "open"
    return ""

log = logging.getLogger("xz.session")

# 模型判定「这句不是跟我说的」时只回这个（规则见 persona._VOICE_RULES）。
# 必须带括号才算：正常回答也可能以「忽略」两个字开头（「忽略就行」）。
_IGNORE = re.compile(r"^\s*[【\[]\s*忽略\s*[】\]]|^\s*忽略\s*[。.！!]?\s*$")
# 垫话不进历史（模型会学着说），方言版也算
_FILLERS = {SEARCH_FILLER, SLOW_FILLER, TOOL_FILLER} | {
    lines[f] for lines in DIALECT_LINES.values()
    for f in (SEARCH_FILLER, SLOW_FILLER, TOOL_FILLER) if f in lines}

# 挂断回待机以后上下文不能丢：「今天天气怎么样」答完挂了，再按一下问「那明天呢」还得接得上。
# device_id → (最后活动时刻, 历史)；过了 session_idle_seconds 就作废。
_SAVED_HISTORY: dict[str, tuple[float, list[dict]]] = {}

_ANNOUNCE_SPLIT = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
# 播报念完等多久再挂：下发只领先 3 帧（180 ms），再留出 4G 抖动，让盒子把尾巴放完
_ANNOUNCE_TAIL_S = 1.0

# 唤醒词只是在叫它，不算一句话：原样交给模型，「打桩模子」这种它会去解释词义。
_WAKE_GREETING = "你好"


class _SpeechItem:
    """一句话的流式音频缓冲：synth 任务生产，发送方按节奏消费。

    预取也靠它——第 N+1 句在后台攒 chunks，第 N 句发完直接从这里接着读。
    """

    def __init__(self, sentence: str):
        self.sentence = sentence
        self.chunks: list[bytes] = []
        self.done = False
        self.error: BaseException | None = None
        self._cond = asyncio.Condition()

    async def push(self, chunk: bytes) -> None:
        async with self._cond:
            self.chunks.append(chunk)
            self._cond.notify_all()

    async def finish(self, error: BaseException | None = None) -> None:
        async with self._cond:
            self.done = True
            self.error = error
            self._cond.notify_all()

    async def wait_chunk(self, idx: int) -> bytes | None:
        """等第 idx 块 PCM；流结束（或出错且没有更多）返回 None。"""
        async with self._cond:
            await self._cond.wait_for(lambda: len(self.chunks) > idx or self.done)
            return self.chunks[idx] if len(self.chunks) > idx else None


class _SpeculativeTurn:
    """抢跑的 LLM：识别结果稳定下来就先问，别等 listen stop。

    边听边认的第二份红利——说完最后一个字到设备发 stop 之间还有几百毫秒
    （手动模式是松手前的停顿，自动模式是 VAD 的 XZ_VAD_SILENCE_MS）。
    这段时间里 partial 已经不变了，正好拿去起 LLM，把它 ~1s 的首字延迟藏掉一截。

    stop 时最终文本和抢跑用的文本一致就直接接着用（命中），不一致就取消重来
    （只多花一次很短的 flash 调用，延迟不比不抢跑差）。
    """

    def __init__(self, session: "DeviceSession", user_text: str, messages: list[dict]):
        self.user_text = user_text
        self.started_at = time.monotonic()
        self.sentences: list[str] = []
        self.done = False
        self.error: BaseException | None = None
        self._cond = asyncio.Condition()
        self.notes: list[str] = []
        self._stream = session.svc.llm.stream_sentences(messages, notes=self.notes)
        self.task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        try:
            async for sentence in self._stream:
                async with self._cond:
                    self.sentences.append(sentence)
                    self._cond.notify_all()
        except asyncio.CancelledError:
            raise
        except BaseException as e:  # noqa: BLE001 - 抢跑失败不能带崩会话
            self.error = e
        finally:
            async with self._cond:
                self.done = True
                self._cond.notify_all()

    async def iterate(self):
        """给 _run_turn 当 llm_stream 用：先吐已经攒下的，再等后续的。"""
        idx = 0
        while True:
            async with self._cond:
                await self._cond.wait_for(lambda: len(self.sentences) > idx or self.done)
                if len(self.sentences) <= idx:
                    break
                sentence = self.sentences[idx]
                idx += 1
            yield sentence

    async def cancel(self) -> None:
        if not self.task.done():
            self.task.cancel()
        with contextlib.suppress(BaseException):
            await self.task
        with contextlib.suppress(BaseException):
            await self._stream.aclose()


class DeviceSession:
    def __init__(self, ws: web.WebSocketResponse, device_id: str, client_id: str, services):
        self.ws = ws
        self.device_id = device_id
        self.client_id = client_id
        self.svc = services
        self.cfg = services.cfg

        self.session_id = uuid.uuid4().hex
        self.log = logging.LoggerAdapter(log, {})
        self._prefix = f"[{device_id or '?'} {self.session_id[:8]}]"

        self.history: list[dict] = []
        self.last_activity = time.monotonic()

        self.listening = False
        self.listen_mode = "auto"
        self.pcm_buffer = bytearray()
        self.decoder = OpusDecoder(self.cfg.uplink_sample_rate, 1)
        self.encoder = OpusEncoder(self.cfg.downlink_sample_rate, 1)
        self.vad = VadSegmenter(
            sample_rate=self.cfg.uplink_sample_rate,
            silence_ms=self.cfg.vad_silence_ms,
            min_speech_ms=self.cfg.vad_min_speech_ms,
            max_utterance_ms=self.cfg.vad_max_utterance_ms,
            aggressiveness=self.cfg.vad_aggressiveness,
        )

        # 边听边认：流式 provider 在 listen start 时开一条 ASRStream，帧到就喂
        self.asr_stream = None
        self._asr_streaming = bool(getattr(services.asr, "streaming", False))
        self.turn_task: asyncio.Task | None = None
        self._tts_prewarm: asyncio.Task | None = None
        self._turn_t: dict = {}
        # LLM 抢跑
        self._spec: _SpeculativeTurn | None = None
        self._spec_tries = 0
        self._spec_asr: asyncio.Task | None = None   # 整段识别抢跑：在路上的那次识别
        self._spec_gen = 0                           # 人又开口就 +1，在路上的结果作废
        self._partial = ""
        self._partial_at = 0.0
        # 断句：停顿时判断说完了没有（见 _turn_hold）
        self._turn_gen = 0
        self._pause_checked = False
        self._last_sil = 0
        self._hold_model = 0
        self._hold_text = 0
        self._text_done = False
        self.speaking = False
        self._mcp_id = 0
        self._mcp_pending: dict[int, asyncio.Future] = {}
        self.device_hello: dict | None = None
        self._closed = False

        # 只听用户：到 _deadline 还没等到一句正经话就挂断（见 _window_watchdog）
        self._deadline: float | None = None
        self._window_due = False      # 到点时正忙（有人在说 / 在识别 / 在答），这句处理完再定
        self._processing = False      # _finish_utterance 进行中（识别要几百毫秒，别在这时挂）
        self._hanging_up = False
        self._announcing = False      # 正在念播报：这期间盒子发来的 listen start 不理，免得电视声把播报打断
        self._last_audio_at = 0.0     # 最后一帧上行音频的时刻
        self._bg: set[asyncio.Task] = set()
        saved = _SAVED_HISTORY.pop(device_id, None) if device_id else None
        if saved and time.monotonic() - saved[0] < self.cfg.session_idle_seconds:
            self.history = saved[1]

    # ---------- 小工具 ----------

    def info(self, msg, *args):
        log.info(f"{self._prefix} " + msg, *args)

    def warn(self, msg, *args):
        log.warning(f"{self._prefix} " + msg, *args)

    async def send_json(self, obj: dict) -> None:
        if self.ws.closed:
            return
        try:
            await self.ws.send_str(json.dumps(obj, ensure_ascii=False))
        except (ConnectionResetError, RuntimeError) as e:
            self.warn("send_json 失败: %s", e)

    async def send_audio(self, opus_payload: bytes) -> None:
        if self.ws.closed:
            return
        try:
            await self.ws.send_bytes(
                protocol.pack_audio(opus_payload, self.cfg.protocol_version)
            )
        except (ConnectionResetError, RuntimeError) as e:
            self.warn("send_audio 失败: %s", e)

    # ---------- 主循环 ----------

    async def run(self) -> None:
        idle_task = asyncio.create_task(self._idle_watchdog())
        window_task = asyncio.create_task(self._window_watchdog())
        if self.history:
            self.info("接上次的上下文（%d 条）", len(self.history))
        try:
            async for msg in self.ws:
                self.last_activity = time.monotonic()
                if msg.type == WSMsgType.TEXT:
                    await self._on_text(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await self._on_binary(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    self.warn("WS 错误: %s", self.ws.exception())
                    break
        finally:
            self._closed = True
            idle_task.cancel()
            window_task.cancel()
            await self._close_asr_stream()
            await self._reset_speculation()
            await self._cancel_turn("connection closed")
            if self.device_id and self.history:
                _SAVED_HISTORY[self.device_id] = (self.last_activity, list(self.history))
                self.info("连接关闭，上下文留 %d 条（%d 秒内回来还接得上）",
                          len(self.history), self.cfg.session_idle_seconds)
            else:
                self.info("连接关闭")
            self.history.clear()

    def _window_on(self) -> bool:
        return self.listen_mode in ("auto", "realtime")

    def _arm_window(self, seconds: float) -> None:
        self._deadline = time.monotonic() + seconds if seconds > 0 else None
        self._window_due = False

    def _busy(self) -> bool:
        return (
            self._processing
            # VAD 判「有人在说」但音频 3 秒没来了：盒子已经不送了（切去说话 / 连接卡了），不能一直算忙
            or (self.vad.triggered and time.monotonic() - self._last_audio_at < 3)
            or (self.turn_task is not None and not self.turn_task.done())
        )

    async def _window_watchdog(self) -> None:
        """只听用户。

        盒子没有回声消除，所以每次答完都会自动重新开麦（listen start mode=auto），
        这时电视、音乐、旁边人聊天全都会被收进来——识别越准，它越会一本正经地回答电视。
        （换 Qwen3-ASR 前统计过：sherpa 判空的 209 段里，Qwen3-ASR 能认出字的有 208 段，
        全是歌词和电视对白。）所以：答完 followup_window_s、刚唤醒 connect_window_s 内
        没等到一句被接受的话，就挂断连接。设备收到断开会自己回待机（application.cc
        OnAudioChannelClosed），跟官方服务端的 close_connection_no_voice_time 一个做法。
        到点时正好有声音在进来，就等这一句处理完再定：接受了照常答，没接受就挂。
        """
        while not self._closed:
            await asyncio.sleep(0.5)
            if self._deadline is None or not self._window_on():
                continue
            if time.monotonic() < self._deadline:
                continue
            if self._busy():
                self._window_due = True
                continue
            await self._hang_up("%.0f 秒没等到你说话" % (
                self.cfg.followup_window_s if self.history else self.cfg.connect_window_s))
            return

    async def _hang_up(self, why: str) -> None:
        if self._hanging_up or self._closed or self.ws.closed:
            return
        self._hanging_up = True
        self.info("%s，挂断，盒子回待机", why)
        if self.speaking:
            self.speaking = False
            await self.send_json(protocol.msg_tts_stop(self.session_id))
        with contextlib.suppress(Exception):
            await self.ws.close()

    def _hang_up_soon(self, why: str) -> None:
        """在本轮任务里面要挂断时用：直接 await 会被 run() 收尾时的 _cancel_turn 打断。"""
        t = asyncio.create_task(self._hang_up(why))
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    def _window_expired(self) -> bool:
        return self._window_on() and (
            self._window_due
            or (self._deadline is not None and time.monotonic() >= self._deadline)
        )

    async def _idle_watchdog(self) -> None:
        while not self._closed:
            await asyncio.sleep(30)
            if (
                self.history
                and time.monotonic() - self.last_activity > self.cfg.session_idle_seconds
            ):
                self.info("空闲超过 %ds，清空上下文", self.cfg.session_idle_seconds)
                self.history.clear()

    # ---------- 上行 text ----------

    async def _on_text(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.warn("非法 JSON: %r", raw[:200])
            return
        mtype = data.get("type")
        if mtype != "mcp":
            self.info("← %s", raw[:300])

        if mtype == "hello":
            await self._on_hello(data)
        elif mtype == "listen":
            await self._on_listen(data)
        elif mtype == "abort":
            await self._on_abort(data)
        elif mtype == "mcp":
            await self._on_mcp(data)
        elif mtype == "goodbye":
            # v2.2.4 的 WS 通道并不发 goodbye（websocket_protocol.cc:78-81），
            # 这里留兜底：收到就当断开处理。
            self.info("收到 goodbye（v2.2.4 WS 理论上不发），按断开处理")
            await self._cancel_turn("goodbye")
            await self.ws.close()
        else:
            self.warn("未处理的消息类型 %r", mtype)

    async def _on_hello(self, data: dict) -> None:
        self.device_hello = data
        hello = protocol.msg_hello(
            self.session_id, self.cfg.downlink_sample_rate, self.cfg.frame_ms
        )
        await self.send_json(hello)
        self.info(
            "→ hello 已回（session_id=%s, 下行 %d Hz/%d ms, 帧协议 v%d）",
            self.session_id,
            self.cfg.downlink_sample_rate,
            self.cfg.frame_ms,
            self.cfg.protocol_version,
        )
        if (data.get("features") or {}).get("mcp"):
            asyncio.create_task(self._mcp_bootstrap())
        self._arm_window(self.cfg.connect_window_s)

    async def _on_listen(self, data: dict) -> None:
        state = data.get("state")
        if state == "start" and self._announcing:
            # 盒子被叫醒时会紧跟 detect 发一个 listen start；播报念完直接挂断，不开麦
            self.listen_mode = data.get("mode", "auto")
            self.info("正在念播报，这次 listen start 先不收音")
            return
        if state == "start":
            self.listen_mode = data.get("mode", "auto")
            self.listening = True
            self.pcm_buffer.clear()
            self.vad.reset()
            self._reset_hold()
            await self._reset_speculation()
            await self._open_asr_stream()
            self.info(
                "开始收音（mode=%s, vad=%s, asr=%s%s）",
                self.listen_mode, self.vad.backend, self.svc.asr.name,
                "/边听边认" if self.asr_stream is not None else "",
            )
        elif state == "stop":
            self.listening = False
            self.info("设备发来 listen stop，缓冲 %.0f ms", self._buffered_ms())
            await self._finish_utterance(reason="device-stop")
        elif state == "detect":
            text = (data.get("text") or "").strip()
            self.info("唤醒词 detect: %r", text)
            if text == _ANNOUNCE_WAKE:
                await self._start_announce()
            elif text == _VOICE_WAKE:
                await self._start_voice_switch()
            elif text:
                await self._start_turn(_WAKE_GREETING)
        else:
            self.warn("未知 listen state %r", state)

    async def _on_abort(self, data: dict) -> None:
        self.info("收到 abort（reason=%s），立刻停 TTS", data.get("reason"))
        await self._reset_speculation()
        await self._cancel_turn("abort")

    # ---------- 上行 binary（音频） ----------

    async def _on_binary(self, frame: bytes) -> None:
        try:
            payload = protocol.unpack_audio(frame, self.cfg.protocol_version)
            pcm = self.decoder.decode(payload, self.cfg.frame_samples_uplink)
        except Exception as e:
            self.warn("Opus 解码失败: %s", e)
            return
        if not self.listening:
            return
        self._last_audio_at = time.monotonic()
        self.pcm_buffer.extend(pcm)
        if self.asr_stream is None and self._asr_streaming:
            await self._open_asr_stream()  # 设备没发 listen start 就直接送音频的兜底
        if self.asr_stream is not None:
            try:
                await self.asr_stream.feed(pcm)
            except Exception as e:
                self.warn("流式 ASR feed 失败，本段退回整段识别: %s", e)
                await self._close_asr_stream()
            else:
                await self._maybe_speculate()
        end = self.vad.feed(pcm)
        if not end:
            self._turn_hold()
        if not end and not self._asr_streaming:
            await self._maybe_speculate_whole()
        # 说完了但设备还没发 stop 的那点尾静音，正好用来握 Edge 的连接（~0.9s）
        if (
            not end
            and self.vad.triggered
            and self.vad.speech_ms >= self.cfg.vad_min_speech_ms
            and self.vad.silence_after_speech_ms >= self.cfg.tts_prewarm_silence_ms
        ):
            self._start_tts_prewarm()
        if end:
            self.info("VAD 判定句尾，缓冲 %.0f ms", self._buffered_ms())
            self.listening = False
            await self._finish_utterance(reason="vad")

    def _buffered_ms(self) -> float:
        return len(self.pcm_buffer) / 2 / self.cfg.uplink_sample_rate * 1000

    def _level_of(self, pcm: bytes) -> str:
        """这段上行音频到底有多响。

        识别为空有两种完全不同的原因：麦克风根本没送声音（峰值贴近 0），
        和送来了但认不出（峰值正常）。不打出来就只能靠猜，2026-09-07 就为这个卡过。
        """
        if not pcm:
            return "无音频"
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x * x)))
        # 一段正常说话的峰值在几千到三万；峰值 < 300 基本等于静音
        verdict = "≈静音，麦克风没送到声音" if peak < 300 else ("偏小" if peak < 2000 else "音量正常")
        return f"峰值 {peak:.0f} / RMS {rms:.0f}，{verdict}"

    def _maybe_dump(self, pcm: bytes, tag: str) -> None:
        """XZ_DUMP_UTTERANCE=1 时把这段上行音频落成 wav，方便自己听。"""
        if not os.environ.get("XZ_DUMP_UTTERANCE") or not pcm:
            return
        d = os.environ.get("XZ_DUMP_DIR") or "/tmp/xz-utterance"
        try:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{time.strftime('%H%M%S')}-{tag}.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.cfg.uplink_sample_rate)
                w.writeframes(pcm)
            self.info("上行音频已存 %s", path)
        except Exception as e:  # noqa: BLE001 - 诊断功能不许拖垮会话
            self.warn("存 wav 失败: %s", e)

    # ---------- LLM 抢跑 ----------

    async def _reset_speculation(self) -> None:
        self._partial = ""
        self._partial_at = 0.0
        self._spec_tries = 0
        await self._drop_whole_spec()

    async def _drop_whole_spec(self) -> None:
        self._spec_gen += 1
        task, self._spec_asr = self._spec_asr, None
        if task is not None and not task.done():
            task.cancel()
        spec, self._spec = self._spec, None
        if spec is not None:
            await spec.cancel()

    async def _maybe_speculate_whole(self) -> None:
        """整段识别也抢跑：尾静音够 llm_speculative_whole_ms 就把已收的音频先拿去认
        （GX10 上 0.1 s），认出来就起 LLM；人又开口就全撤。

        VAD 要等 800 ms 静音才判句尾，这段时间 LLM 本来干等着。抢跑命中时句尾那一刻
        直接用抢跑的文本（后面只有静音，不用再认一遍），省下 ~0.45 s。
        """
        if not self.cfg.llm_speculative or self.cfg.llm_speculative_whole_ms <= 0:
            return
        if self._spec is not None or self._spec_asr is not None:
            if self.vad.triggered and self.vad.silence_after_speech_ms == 0:
                self.info("又开口了，撤掉抢跑")
                await self._drop_whole_spec()
            return
        if not self.vad.triggered or self.vad.speech_ms < self.cfg.vad_min_speech_ms:
            return
        if self.vad.silence_after_speech_ms < self.cfg.llm_speculative_whole_ms:
            return
        if self._spec_tries >= 6:  # 一段话里来回停顿，最多抢六次（停顿等得久了，停的次数也多）
            return
        self._spec_tries += 1
        self._spec_asr = asyncio.create_task(
            self._spec_transcribe(bytes(self.pcm_buffer), self._spec_gen))

    async def _spec_transcribe(self, pcm: bytes, gen: int) -> None:
        try:
            text = await self.svc.asr.transcribe(pcm, self.cfg.uplink_sample_rate)
        except Exception as e:  # noqa: BLE001 - 抢跑失败不影响正常流程
            self.warn("抢跑识别失败：%s", e)
            text = ""
        if gen != self._spec_gen:
            return  # 路上人又开口了
        self._spec_asr = None
        if text and self._spec is None and not self._closed:
            self._spec = _SpeculativeTurn(self, text, self._build_messages(text))
            self.info("LLM 抢跑（整段识别，第 %d 次，尾静音 %d ms）：%r",
                      self._spec_tries, self.vad.silence_after_speech_ms, text)
        kind = _dangling(text) if self.listening else ""
        if kind:
            # 只听到「嗯」，或者停在「然后 / 就是」上：多等一会儿，后半句多半还在路上
            self._hold_text = (self.cfg.turn_filler_ms if kind == "filler"
                               else self.cfg.turn_open_ms)
            self._apply_hold()
            self.info("「%s」%s，静音 %d ms 再收尾", text,
                      "只是语气词" if kind == "filler" else "像没说完", self.vad.silence_ms)
        elif self.listening and re.search(r"[？?]$", (text or "").strip()):
            self._text_done = True
            self._apply_hold()

    async def _maybe_speculate(self) -> None:
        """人已经停口、partial 也不动了，就先把 LLM 发出去（还没到 listen stop）。

        两个条件缺一不可：
          - **VAD 看到尾静音**：光看 partial 不变不行——zipformer 是按 chunk 吐字的，
            说话中途 partial 常有 200–400 ms 不动，只按这个判会在句子刚起头就抢跑；
          - **partial 稳定**：解码比声音晚一点，等它追上再定文本。
        """
        if not self.cfg.llm_speculative or self.asr_stream is None:
            return
        partial = (self.asr_stream.partial or "").strip()
        now = time.monotonic()
        if partial != self._partial:
            self._partial = partial
            self._partial_at = now
            if self._spec is not None and self._spec.user_text != partial:
                # 人还在说，抢跑用的文本已经过时：撤掉，等下次停口再抢
                spec, self._spec = self._spec, None
                await spec.cancel()
            return
        if not partial or self._spec is not None:
            return
        if self.vad.speech_ms < self.cfg.vad_min_speech_ms:
            return
        if self.vad.silence_after_speech_ms < self.cfg.llm_speculative_silence_ms:
            return
        if (now - self._partial_at) * 1000 < self.cfg.llm_speculative_stable_ms:
            return
        if self._spec_tries >= 3:  # 一段话里最多抢跑三次，免得来回停顿把 flash 打爆
            return
        self._spec_tries += 1
        self._spec = _SpeculativeTurn(self, partial, self._build_messages(partial))
        self.info(
            "LLM 抢跑（第 %d 次，尾静音 %d ms）：%r",
            self._spec_tries, self.vad.silence_after_speech_ms, partial,
        )

    async def _adopt_speculation(self, text: str) -> _SpeculativeTurn | None:
        """最终文本和抢跑一致就接着用，否则撤掉重来。"""
        spec, self._spec = self._spec, None
        if spec is None:
            return None
        if spec.user_text == text:
            self.info("LLM 抢跑命中，提前 %.0f ms 起跑", (time.monotonic() - spec.started_at) * 1000)
            return spec
        self.info("LLM 抢跑落空（%r ≠ %r），重来", spec.user_text, text)
        await spec.cancel()
        return None

    async def _open_asr_stream(self) -> None:
        """流式 provider：开一条新的 ASRStream（整段式 provider 什么都不做）。"""
        await self._close_asr_stream()
        if not self._asr_streaming:
            return
        try:
            self.asr_stream = self.svc.asr.create_stream()
        except Exception as e:
            self.warn("开流式 ASR 失败，本轮退回整段识别: %s", e)
            self.asr_stream = None

    async def _close_asr_stream(self) -> None:
        stream, self.asr_stream = self.asr_stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.close()

    # ---------- 断句：停顿时判断「说完了没有」 ----------

    def _reset_hold(self) -> None:
        self._turn_gen += 1
        self._pause_checked = False
        self._last_sil = 0
        self._hold_model = self._hold_text = 0
        self._text_done = False
        self.vad.silence_ms = self.cfg.vad_silence_ms

    def _apply_hold(self) -> None:
        model = self._hold_model or self.cfg.vad_silence_ms
        if self._text_done:
            # 识别出来是个带问号的完整问句：模型说没说完也不等（9/24 测试里上海话问句被它判成没说完）
            model = min(model, self.cfg.turn_complete_ms)
        self.vad.silence_ms = max(model, self._hold_text)

    def _turn_hold(self) -> None:
        """停顿 turn_check_ms 时问一次断句模型；又开口了就恢复默认，下一个停顿再问。

        只看静音的话，想词时停顿过 800 ms 就被当成说完了（9/24「我没说完就结束了」）。
        模型觉得没说完就把收尾静音放宽到 turn_incomplete_ms；觉得说完了就按 turn_complete_ms。
        """
        v = self.vad
        if not v.triggered:
            return
        sil = v.silence_after_speech_ms
        resumed, self._last_sil = sil < self._last_sil, sil
        if resumed:
            if self._pause_checked or self._hold_model or self._hold_text or self._text_done:
                self._turn_gen += 1
                self._pause_checked = False
                self._hold_model = self._hold_text = 0
                self._text_done = False
                self._apply_hold()
            return
        if (self._pause_checked or self.svc.turn is None
                or v.speech_ms < self.cfg.vad_min_speech_ms or sil < self.cfg.turn_check_ms):
            return
        self._pause_checked = True
        t = asyncio.create_task(self._check_turn(bytes(self.pcm_buffer), self._turn_gen))
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _check_turn(self, pcm: bytes, gen: int) -> None:
        t0 = time.monotonic()
        try:
            p = await asyncio.to_thread(self.svc.turn.predict, pcm)
        except Exception as e:  # noqa: BLE001 - 模型出错就按默认静音断句
            self.warn("断句模型出错：%s", e)
            return
        if gen != self._turn_gen or not self.listening:
            return
        done = p >= self.cfg.turn_threshold
        self._hold_model = self.cfg.turn_complete_ms if done else self.cfg.turn_incomplete_ms
        self._apply_hold()
        self.info("停顿 %d ms，断句模型 %.2f（%.0f ms）→ %s，静音 %d ms 收尾",
                  self.vad.silence_after_speech_ms, p, (time.monotonic() - t0) * 1000,
                  "像说完了" if done else "像没说完", self.vad.silence_ms)

    def _last_reply_asked(self) -> bool:
        """盒子上一句是不是在问主人（问了的话，「嗯」「对」就是回答，得接着答）。"""
        for m in reversed(self.history):
            if m.get("role") == "assistant":
                c = m.get("content")
                return isinstance(c, str) and bool(_ASKED.search(c.strip()[-20:]))
        return False

    async def _rearm_listening(self, reason: str) -> None:
        """这一段没能进 LLM（太短 / 识别为空）时，把收音重新打开。

        不做这件事会死锁，用户看到的症状就是「一直是聆听中，没有回答」：
        句尾是我们这边的 VAD 判的，设备根本不知道这段被丢了，它还停在
        listening 状态继续往上送音频；而服务端已经把 self.listening 置 False，
        _on_binary 的第一行就把音频全扔了，两边永远等不到对方。
        设备主动发 listen stop 的那种（manual 模式）它在等服务端说话，
        给它一对空的 tts start/stop 才能退回 idle 并重新开麦。
        """
        if reason == "device-stop":
            await self.send_json(protocol.msg_tts_start(self.session_id))
            await self.send_json(protocol.msg_tts_stop(self.session_id))
            return
        self.listening = True
        self.pcm_buffer.clear()
        self.vad.reset()
        self._reset_hold()
        await self._open_asr_stream()
        self.info("这段作废，继续收音")

    async def _finish_utterance(self, reason: str) -> None:
        self._processing = True
        try:
            await self._finish_utterance_inner(reason)
        finally:
            self._processing = False

    async def _after_rejected(self, reason: str, in_turn: bool = False) -> None:
        """这一句没进对话（太短 / 识别为空 / 模型判定不是跟它说的）。窗口到点了就挂，否则接着听。"""
        if self._window_expired():
            why = "到点了，最后这句也不是跟我说的"
            if in_turn:
                self._hang_up_soon(why)
            else:
                await self._hang_up(why)
            return
        await self._rearm_listening(reason)

    async def _finish_utterance_inner(self, reason: str) -> None:
        self._turn_reason = reason
        t_stop = time.monotonic()  # = listen stop / VAD 判尾的时刻，延迟都从这里算
        pcm = bytes(self.pcm_buffer)
        self.pcm_buffer.clear()
        dur_ms = len(pcm) / 2 / self.cfg.uplink_sample_rate * 1000
        self.vad.reset()
        self._reset_hold()
        stream, self.asr_stream = self.asr_stream, None
        if dur_ms < self.cfg.vad_min_speech_ms:
            self.info("这段太短（%.0f ms），丢弃（%s）", dur_ms, reason)
            if stream is not None:
                with contextlib.suppress(Exception):
                    await stream.close()
            await self._after_rejected(reason)
            return
        # 声纹跟 ASR 收尾一起算（线程里几十毫秒），要用时再等
        vp = getattr(self.svc, "voiceprint", None)
        sr = self.cfg.uplink_sample_rate
        vp_score = asyncio.create_task(asyncio.to_thread(vp.score, pcm, sr)) if vp is not None else None
        # 还要等 LLM ~1s，先把 Edge 的连接预热起来（握手 ~0.9s 藏进等待里）
        self._start_tts_prewarm()
        text = ""
        if stream is not None:
            # 边听边认：说话期间已经解完，这里只收尾巴
            try:
                text = await stream.finalize()
            except Exception as e:
                self.warn("流式 ASR 收尾失败，退回整段识别: %s", e)
            finally:
                with contextlib.suppress(Exception):
                    await stream.close()
        if stream is None and self._spec_asr is not None:
            # 抢跑的识别还在路上（它只差几十毫秒），等它，别再发一次
            with contextlib.suppress(BaseException):
                await asyncio.shield(self._spec_asr)
        if not text and stream is None and self._spec is not None:
            # 抢跑起了而且之后没人再开口（开口会被撤掉）：后面全是静音，文本就是它
            text = self._spec.user_text
        if not text and stream is None:
            text = await self.svc.asr.transcribe(pcm, self.cfg.uplink_sample_rate)
        asr_ms = (time.monotonic() - t_stop) * 1000
        self.info(
            "ASR(%s, %s) stop 后 %.0f ms → %r",
            reason, "流式收尾" if stream is not None else "整段", asr_ms, text,
        )
        if not text:
            self.info("识别为空，不进 LLM（%s）", self._level_of(pcm))
            self._maybe_dump(pcm, "empty")
            await self._reset_speculation()
            await self._close_tts_prewarm()
            await self._after_rejected(reason)
            return
        if (self.cfg.turn_ignore_fillers and _dangling(text) == "filler"
                and not self._last_reply_asked()):
            # 多等了两秒半还是只有「嗯」：不是在跟它说事，别回「好了」，接着听
            self.info("只有「%s」，不当一句话，接着听", text)
            await self._reset_speculation()
            await self._close_tts_prewarm()
            await self._after_rejected(reason)
            return
        score = None
        if vp is not None:
            score = await vp_score
            if score is not None:
                self.info("声纹 %.2f（门槛 %.2f，%s）", score, vp.threshold, vp.mode)
            if vp.mode == "enroll":
                await asyncio.to_thread(vp.save, pcm, sr, text, score)
            elif not vp.passes(score):
                # 电视、旁人说话：屏幕上照样显示听到了什么，说一声为什么没理
                self.info("不像你的声音，不回答：%r", text)
                await self.send_json(protocol.msg_stt(text, self.session_id))
                await self.send_json(
                    protocol.msg_tts_sentence_start("（不像你的声音，没理会）", self.session_id))
                await self._reset_speculation()
                await self._close_tts_prewarm()
                await self._after_rejected(reason)
                return
        if self.cfg.keep_recent > 0:
            # 9/26 用户：「为啥不存，至少存三段吧」。只存过了声纹的（本人），滚动留最近几段
            t = asyncio.create_task(asyncio.to_thread(
                keep_recent, self.cfg.recent_dir, self.cfg.keep_recent, pcm, sr, text, score))
            self._bg.add(t)
            t.add_done_callback(self._bg.discard)
        if _VOICE_CMD.search(text):
            # 讲「换只声音」跟长按 BOOT 一样：换下一种声音，用新声音报一声（固件没更新前也能换）
            self.info("要换声音：%r", text)
            await self._reset_speculation()
            await self._close_tts_prewarm()
            await self._start_voice_switch()
            return
        self._turn_t = {"stop": t_stop, "asr_ms": asr_ms, "logged": False}
        spec = await self._adopt_speculation(text)
        self._spec_tries = 0
        # 先取走预热连接再 _start_turn：它的"取消上一轮"清扫不能吃掉这条连接
        await self._start_turn(
            text, asr_started=t_stop, prewarm=self._take_tts_prewarm(), spec=spec
        )

    # ---------- 一轮对话 ----------

    async def _start_turn(
        self, user_text: str, asr_started: float | None = None, prewarm=None, spec=None
    ) -> None:
        await self._cancel_turn("new turn")
        self.turn_task = asyncio.create_task(
            self._run_turn(user_text, asr_started, prewarm, spec)
        )

    async def _cancel_turn(self, why: str) -> None:
        task = self.turn_task
        self.turn_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self.info("已取消上一轮（%s）", why)
        await self._close_tts_prewarm()
        if self.speaking:
            self.speaking = False
            await self.send_json(protocol.msg_tts_stop(self.session_id))

    # ---------- TTS 连接预热（edge 专用，别的 provider 没这方法就跳过） ----------

    def _start_tts_prewarm(self) -> None:
        tts = self.svc.tts
        if getattr(tts, "prewarm", None) is None or self._tts_prewarm is not None:
            return
        self._tts_prewarm = asyncio.create_task(tts.prewarm())

    def _take_tts_prewarm(self) -> asyncio.Task | None:
        """把预热任务交给本轮（**不等它**）。

        ASR 变成边听边认之后这里绝不能 await：Edge 握手 ~0.9s，等它就等于把省下来的
        识别时间原样还回去。任务一直跑到第一句要合成时才 await（那时 LLM 已经花了 ~1s，
        握手早完成了）。取走后 self._tts_prewarm 置空，_cancel_turn 的清扫就不会误关它。
        """
        task, self._tts_prewarm = self._tts_prewarm, None
        return task

    async def _await_prewarm(self, task):
        """真正要用连接时才等；失败就退回常规连接。"""
        if task is None:
            return None
        try:
            return await task
        except Exception as e:
            self.warn("TTS 预热失败（改用常规连接）: %s", e)
            return None

    async def _discard_prewarm(self, task) -> None:
        conn = await self._await_prewarm(task)
        if conn is not None:
            with contextlib.suppress(Exception):
                await conn.close()

    async def _close_tts_prewarm(self) -> None:
        task, self._tts_prewarm = self._tts_prewarm, None
        if task is None:
            return
        try:
            conn = await task
        except Exception:
            return
        if conn is not None:
            with contextlib.suppress(Exception):
                await conn.close()

    async def _run_turn(
        self, user_text: str, asr_started: float | None, prewarm=None, spec=None
    ) -> None:
        t_turn = time.monotonic()
        # asr_started = listen stop / VAD 判尾的时刻，用它报「listen stop→首帧」
        t_listen_stop = asr_started or t_turn
        if self._turn_t.get("stop") != t_listen_stop:
            # 唤醒词 detect 直接给文本，没有 ASR 这一段
            self._turn_t = {"stop": t_listen_stop, "asr_ms": 0.0, "logged": False}
        self._turn_t["asr_done"] = t_turn
        self._turn_t["first_sentence_at"] = None
        await self.send_json(protocol.msg_stt(user_text, self.session_id))
        # tts start 等第一句真要念了再发：模型判「不是跟我说的」时，
        # 盒子一直停在聆听，什么都不用收拾（以前先发 start 再撤，盒子会闪一下「说话中」）

        messages = self._build_messages(user_text)
        reply_parts: list[str] = []
        first_sentence_at: float | None = None
        first_audio_at: float | None = None

        # 抢跑命中就直接接上那条已经在跑的流，否则现起
        notes: list[str] = spec.notes if spec is not None else []
        llm_stream = (spec.iterate() if spec is not None
                      else self.svc.llm.stream_sentences(messages, notes=notes))
        llm_done = False
        ignored = False
        synth_tasks: set[asyncio.Task] = set()
        # prewarm 是「还在握手的 Task」，不是连接：第一句专用，后续句靠预取重叠

        async def next_item() -> _SpeechItem | None:
            """等 LLM 切出下一句，立刻起 TTS 任务（不等合成完）。"""
            nonlocal llm_done, prewarm, ignored
            if llm_done:
                return None
            async for sentence in llm_stream:
                if not sentence:
                    continue
                if _IGNORE.search(sentence):
                    ignored = True
                    llm_done = True
                    return None
                item = _SpeechItem(dialect_line(sentence, self.cfg.voice_dialect))
                task = asyncio.create_task(self._synth_into(item, prewarm))
                prewarm = None  # 只有第一句能吃到预热连接
                synth_tasks.add(task)
                task.add_done_callback(synth_tasks.discard)
                return item
            llm_done = True
            return None

        current: _SpeechItem | None = None
        prefetch: asyncio.Task | None = None
        hang_up_after = False
        try:
            current = await next_item()
            while current is not None:
                # 发第 N 句的同时预取第 N+1 句：LLM 流 + TTS 连接都在后台跑
                prefetch = asyncio.create_task(next_item())
                if first_sentence_at is None:
                    first_sentence_at = time.monotonic()
                    self._turn_t["first_sentence_at"] = first_sentence_at
                    self.info("LLM 首句 %.2fs: %r", first_sentence_at - t_turn, current.sentence)
                reply_parts.append(current.sentence)
                if not self.speaking:
                    await self.send_json(protocol.msg_tts_start(self.session_id))
                    self.speaking = True
                await self.send_json(
                    protocol.msg_tts_sentence_start(current.sentence, self.session_id)
                )
                sent = await self._speak_audio(current, t_turn, t_listen_stop)
                if sent and first_audio_at is None:
                    first_audio_at = time.monotonic()
                    self.info("首句音频已下发，端到端 %.2fs", first_audio_at - t_turn)
                current = await prefetch
                prefetch = None
            if self.cfg.one_turn and any(p not in _FILLERS for p in reply_parts):
                # 一问一答（9/25 用户：「回答完一句以后，就不要再听我说话了，直到我按一下」）：
                # 跟播报一样不发 tts stop（盒子收到会转进聆听），等尾巴放完就挂断，从「说话」直接回待机
                await asyncio.sleep(_ANNOUNCE_TAIL_S)
                self.speaking = False
                hang_up_after = True
        except asyncio.CancelledError:
            self.info("本轮被打断")
            raise
        finally:
            # 收尾：停预取、停所有 TTS 连接、关 LLM 流（abort 时全都要能立刻取消）
            if prewarm is not None:
                await self._discard_prewarm(prewarm)
                prewarm = None
            if prefetch is not None:
                prefetch.cancel()
            for task in list(synth_tasks):
                task.cancel()
            if prefetch is not None:
                with contextlib.suppress(BaseException):
                    await prefetch
            for task in list(synth_tasks):
                with contextlib.suppress(BaseException):
                    await task
            with contextlib.suppress(BaseException):
                await llm_stream.aclose()
            if spec is not None:
                await spec.cancel()
            if self.speaking:
                self.speaking = False
                await self.send_json(protocol.msg_tts_stop(self.session_id))

        reply = "".join(reply_parts)
        if ignored and all(p in _FILLERS for p in reply_parts):
            # 电视 / 歌词 / 别人聊天：不进上下文，不续窗口，屏幕上说一声为什么没理
            self.info("模型判定不是跟我说的，略过（%.2fs）：%r", time.monotonic() - t_turn, user_text)
            await self.send_json(
                protocol.msg_tts_sentence_start("（不像跟我说的，没理会）", self.session_id)
            )
            await self._after_rejected(getattr(self, "_turn_reason", "vad"), in_turn=True)
            return
        if reply:
            # 这一轮查到的资料跟着存：追问「那后天呢」「那上海呢」要靠它，不然模型只能编
            seen = f"{user_text}\n\n（当时查到的资料）\n{notes[-1][:600]}" if notes else user_text
            self.history.append({"role": "user", "content": seen})
            # 垫话不进上下文：存进去模型会学着每句都先来一句「我看一下啊」（9/24 一轮说了两遍）
            said = "".join(p for p in reply_parts if p not in _FILLERS) or reply
            self.history.append({"role": "assistant", "content": said})
            self._trim_history()
            if not hang_up_after:
                self._arm_window(self.cfg.followup_window_s)
        self.last_activity = time.monotonic()
        self.info("本轮结束，总耗时 %.2fs，回复 %r", time.monotonic() - t_turn, reply)
        if hang_up_after:
            self._hang_up_soon("一问一答，答完不再听")

    # ---------- Claude Code 任务播报（app/announce.py） ----------

    async def _start_voice_switch(self) -> None:
        """音量加减一起按（固件 2.2.4.16 起，之前是长按 BOOT；都发 listen detect text=__voice__）或讲「换只声音」：
        轮到下一种声音，用新声音报一声就挂。"""
        v = voices.cycle()
        say = f"册那，调好了！现在是{voices.name(v)}，侬听听看好伐？"  # 9/27 用户：上海话「换好了」讲「调（dio）好了」
        self.info("换声音 → %s（%s）", v, voices.name(v))
        await self._cancel_turn("voice")
        self._announcing = True
        self.listening = False
        self.turn_task = asyncio.create_task(self._run_announce(say, say))

    async def _start_announce(self) -> None:
        """盒子是被收件箱叫醒的：把排队的播报取出来直接念，不过 LLM。"""
        hub = getattr(self.svc, "announce", None)
        self._announcing = True   # 等改写那几秒里盒子发来的 listen start 也不收音
        self.listening = False
        say, text = await hub.take_speech(self.device_id) if hub is not None else ("", "")
        if not say:
            self._announcing = False
            await self._hang_up("被叫醒来听播报，但队列已经空了（过期或已念过）")
            return
        self.info("播报：%r", text)
        if say != text:
            self.info("播报念方言口语版：%r", say)
        await self._cancel_turn("announce")
        self._announcing = True
        self.listening = False
        self.turn_task = asyncio.create_task(self._run_announce(say, text))

    async def _run_announce(self, say: str, text: str) -> None:
        """念 say（方言口语版）；上下文里记 text（原文，细节全）。"""
        t0 = time.monotonic()
        self._turn_t = {}   # 不记延迟账：这一轮没有 ASR / LLM
        sentences = [m.strip() for m in _ANNOUNCE_SPLIT.findall(say) if m.strip()] or [say]
        synth: list[asyncio.Task] = []

        def start(i: int) -> _SpeechItem:
            item = _SpeechItem(sentences[i])
            synth.append(asyncio.create_task(self._synth_into(item)))
            return item

        try:
            nxt = start(0)
            for i in range(len(sentences)):
                cur = nxt
                nxt = start(i + 1) if i + 1 < len(sentences) else None  # 念这句时合成下一句
                if not self.speaking:
                    await self.send_json(protocol.msg_tts_start(self.session_id))
                    self.speaking = True
                await self.send_json(protocol.msg_tts_sentence_start(cur.sentence, self.session_id))
                await self._speak_audio(cur, t0, t0)
            # 念完不发 tts stop：盒子是自动收音模式，收到 stop 会转进聆听。播报就是播报（9/25 用户定），
            # 等尾巴放完直接挂断，盒子从「说话」直接回待机
            await asyncio.sleep(_ANNOUNCE_TAIL_S)
            self.speaking = False
        except asyncio.CancelledError:
            self.info("播报被打断")
            raise
        finally:
            self._announcing = False
            for task in synth:
                task.cancel()
            for task in synth:
                with contextlib.suppress(BaseException):
                    await task
            if self.speaking:
                self.speaking = False
                await self.send_json(protocol.msg_tts_stop(self.session_id))
        # 记进上下文：挂断后上下文会留一阵，用户再喊醒问「哪个项目？」「什么失败了？」模型才接得上
        self.history.append({"role": "user", "content": "（系统：你刚刚主动播报了一条 Claude Code 任务提醒，原话如下）"})
        self.history.append({"role": "assistant", "content": text})
        self._trim_history()
        self.last_activity = time.monotonic()
        self.info("播报念完，%.2fs", time.monotonic() - t0)
        self._hang_up_soon("播报不等回话")

    def _build_messages(self, user_text: str) -> list[dict]:
        return list(self.history) + [{"role": "user", "content": user_text}]

    def _trim_history(self) -> None:
        keep = self.cfg.llm_history_turns * 2
        if len(self.history) > keep:
            del self.history[: len(self.history) - keep]

    async def _synth_into(self, item: _SpeechItem, prewarm=None) -> None:
        """一句话 → PCM 块流进 item（独立 task，好取消）。

        edge：synth_stream 流式（块到就 push）；prewarm 是**还在握手的 Task**，
        到这里才 await —— LLM 那 ~1s 里握手已经完成，等于零成本。
        voxtts：整句 wav 转 PCM 一次 push。
        """
        tts = self.svc.tts
        try:
            conn = await self._await_prewarm(prewarm)
        except asyncio.CancelledError:
            # 握手还没完就被打断：让它自己跑完再关，别把连接漏在 Edge 那边
            if prewarm is not None:
                asyncio.create_task(self._discard_prewarm(prewarm))
            raise
        stream = getattr(tts, "synth_stream", None)
        if stream is None and conn is not None:
            with contextlib.suppress(Exception):
                await conn.close()
        say = fix_readings(item.sentence)  # 只改念法，item.sentence 照旧进日志和上下文
        if say != item.sentence:
            self.info("读音纠正：%r", say)
        agen = stream(say, prewarm=conn) if stream is not None else None
        try:
            if agen is not None:
                async for chunk in agen:
                    await item.push(chunk)
            else:
                wav, _rtt = await tts.synth(say)
                if wav:
                    pcm, src_rate = wav_to_pcm16_mono(wav)
                    if src_rate != self.cfg.downlink_sample_rate:
                        pcm = resample_pcm16(pcm, src_rate, self.cfg.downlink_sample_rate)
                    if pcm:
                        await item.push(pcm)
            await item.finish()
        except Exception as e:
            await item.finish(error=e)
        finally:
            if agen is not None:
                with contextlib.suppress(BaseException):
                    await agen.aclose()

    async def _speak_audio(
        self, item: _SpeechItem, t_turn: float, t_listen_stop: float
    ) -> bool:
        """按 60 ms 实时节奏下发一句的音频；块到就编就发，不等整句。

        顺序仍受 _run_turn 控制：sentence_start → 音频 → 下一句。
        """
        frame_bytes = self.cfg.frame_samples_downlink * 2
        prebuffer = 3  # 先攒 3 帧再起实时节奏，抗网络抖动（与旧实现一致）
        pend = bytearray()
        sent = 0
        idx = 0
        pace_start: float | None = None
        first_logged = False

        def _delay_for_next() -> float:
            target = pace_start + max(0, sent - prebuffer + 1) * self.cfg.frame_ms / 1000.0
            return target - time.monotonic()

        while True:
            chunk = await item.wait_chunk(idx)
            if chunk is None:
                break
            idx += 1
            pend.extend(chunk)
            while len(pend) >= frame_bytes:
                frame = bytes(pend[:frame_bytes])
                del pend[:frame_bytes]
                if pace_start is None:
                    pace_start = time.monotonic()
                else:
                    delay = _delay_for_next()
                    if delay > 0:
                        await asyncio.sleep(delay)
                await self.send_audio(
                    self.encoder.encode(frame, self.cfg.frame_samples_downlink)
                )
                sent += 1
                if not first_logged:
                    first_logged = True
                    self._log_turn_latency()

        if item.error and sent == 0:
            self.warn("TTS 失败，跳过这句: %s", item.error)
            return False
        if item.error:
            self.warn("TTS 中途出错，已发 %d 帧后截断: %s", sent, item.error)
        if pend:
            # 尾帧不足一帧：补零发出去，保证时长与文本对齐
            if pace_start is not None:
                delay = _delay_for_next()
                if delay > 0:
                    await asyncio.sleep(delay)
            frame = bytes(pend) + b"\x00" * (frame_bytes - len(pend))
            await self.send_audio(
                self.encoder.encode(frame, self.cfg.frame_samples_downlink)
            )
            sent += 1
        return sent > 0

    def _log_turn_latency(self) -> None:
        """每轮一行：asr / llm_first / tts_first / total，四段相加 = total。

        asr        = listen stop → 识别文本就绪（边听边认时只剩尾巴）
        llm_first  = 识别文本 → LLM 切出第一句
        tts_first  = 第一句 → 第一帧下行音频（Edge 合成 + 编码）
        total      = listen stop → 第一帧音频（= 设备侧感受到的首句延迟）
        """
        d = self._turn_t
        if not d or d.get("logged"):
            return
        d["logged"] = True
        now = time.monotonic()
        stop = d.get("stop", now)
        asr_ms = d.get("asr_ms", 0.0)
        asr_done = d.get("asr_done", stop)
        first_sentence_at = d.get("first_sentence_at") or now
        self.info(
            "turn latency: asr=%.0fms llm_first=%.0fms tts_first=%.0fms total=%.0fms",
            asr_ms,
            (first_sentence_at - asr_done) * 1000,
            (now - first_sentence_at) * 1000,
            (now - stop) * 1000,
        )

    # ---------- MCP（设备是 server，我们是 client） ----------

    async def _mcp_bootstrap(self) -> None:
        try:
            init = await self._mcp_call(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "xiaozhi-selfhost", "version": "0.1.0"},
                },
                timeout=8,
            )
            self.info("MCP initialize → %s", json.dumps(init, ensure_ascii=False)[:400])

            tools: list[dict] = []
            cursor = ""
            for _ in range(10):  # 单条 payload 上限 8000 字节，要按 nextCursor 翻页
                params = {"withUserTools": False}
                if cursor:
                    params["cursor"] = cursor
                page = await self._mcp_call("tools/list", params, timeout=10)
                tools.extend(page.get("tools") or [])
                cursor = page.get("nextCursor") or ""
                if not cursor:
                    break
            self.info("MCP tools/list 共 %d 个工具：", len(tools))
            for t in tools:
                self.info("    - %s : %s", t.get("name"), (t.get("description") or "")[:100])
            self.mcp_tools = tools
        except asyncio.TimeoutError:
            self.warn("MCP 握手超时（设备可能没开 MCP）")
            return
        except Exception as e:
            self.warn("MCP 握手失败: %s", e)
            return
        await self._log_device_network()
        await self._reboot_if_update_pending()

    async def _log_device_network(self) -> None:
        """每次连上记一笔盒子走的是 Wi-Fi 还是 4G、信号怎样（9/23 用户换了 4G 模组）。"""
        try:
            res = await self._mcp_call("tools/call", {"name": "self.get_device_status", "arguments": {}}, timeout=5)
            text = ((res.get("content") or [{}])[0] or {}).get("text") or "{}"
            status = json.loads(text)
            net = status.get("network")
            if isinstance(net, dict):
                net = {k: v for k, v in net.items() if k not in ("ssid", "iccid", "imei")}  # 不往日志里写
                self.info("盒子网络：%s", json.dumps(net, ensure_ascii=False)[:200])
        except Exception as e:  # noqa: BLE001 - 只是记日志
            self.info("没拿到盒子状态：%s", type(e).__name__)

    async def _reboot_if_update_pending(self) -> None:
        """有新固件等着这台装，就让它重启一次——盒子只在开机时查版本。"""
        from . import firmware
        m = firmware.pending_for(self.cfg, self.device_id)
        if not m or not firmware.claim_reboot(self.cfg, self.device_id, m["version"]):
            return
        self.warn("有新固件 %s 等着装，让盒子重启去升级", m["version"])
        try:
            await self._mcp_call("tools/call", {"name": "self.reboot", "arguments": {}}, timeout=5)
        except asyncio.TimeoutError:
            # 9/23 实测：盒子收到就重启了，来不及回话；9 秒后它的开机 OTA 请求就到了
            self.info("盒子没回话，多半已经在重启")
        except Exception as e:  # noqa: BLE001
            self.warn("让盒子重启没成：%s（下次开机照样会查到新版本）", e)

    async def _mcp_call(self, method: str, params: dict, timeout: float = 10.0) -> dict:
        self._mcp_id += 1
        req_id = self._mcp_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._mcp_pending[req_id] = fut
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        await self.send_json(protocol.msg_mcp(payload, self.session_id))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._mcp_pending.pop(req_id, None)

    async def _on_mcp(self, data: dict) -> None:
        payload = data.get("payload") or {}
        req_id = payload.get("id")
        fut = self._mcp_pending.get(req_id) if isinstance(req_id, int) else None
        if fut and not fut.done():
            if "error" in payload:
                fut.set_exception(RuntimeError(str(payload["error"])))
            else:
                fut.set_result(payload.get("result") or {})
            return
        self.info("MCP 未匹配消息: %s", json.dumps(payload, ensure_ascii=False)[:300])
