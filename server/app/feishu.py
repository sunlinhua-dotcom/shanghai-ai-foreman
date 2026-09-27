"""飞书入口：把飞书 App 当成盒子之外的第二张嘴。

为什么这么接：飞书输入框自带的「语音转文字」跑在客户端，免费，而且比我们
本地的 sherpa-onnx 准。所以不要去调飞书那个按量计费的 speech_to_text 接口
（免费版根本不让调），只要让用户在飞书里说话转成文字发给机器人，我们收到的
就已经是文本了，走同一套 LLM + 联网搜索，再把答案回到聊天窗口。

连接方式是飞书的长连接（WebSocket），机器人主动连出去，所以 EX10 在 NAT
后面不用开端口、不用隧道、不用公网回调地址。

一个坑：只有「文本消息」带得到文字。用户如果是按住说话发了条语音条，
到我们这儿是 msg_type=audio + 一个 file_key，没有文字——那条路要转写就得
花钱了。所以收到语音条时提示一句「请用输入框里的语音转文字」。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict

from .llm.claude import SEARCH_FILLER, ClaudeLLM
from .persona import build_system_prompt
from .tools import _http

log = logging.getLogger("xz.feishu")

_BASE = "https://open.feishu.cn/open-apis"
_ASCII_EDGE = re.compile(r"[0-9A-Za-z]")

_AUDIO_HINT = (
    "我这边只收得到文字哦。发语音条的话我听不见——"
    "请点输入框右边的话筒，用「语音转文字」说完再发，"
    "那个是飞书本地转的，又快又准，也不花钱。"
)
_UNSUPPORTED_HINT = "这种消息我还看不懂，发文字给我就行。"


def _join(sentences: list[str]) -> str:
    """把流式切出来的句子拼回一段话。

    句子在切的时候被 strip 过，中文直接拼没问题；两边都是英文/数字时
    补一个空格，免得 "RTX" "5090" 粘成一坨。
    """
    out = ""
    for s in sentences:
        if not s:
            continue
        if out and _ASCII_EDGE.match(out[-1]) and _ASCII_EDGE.match(s[0]):
            out += " "
        out += s
    return out.strip()


class _Chat:
    """一个飞书会话的上下文。和盒子那边 DeviceSession 的 history 一个套路。"""

    def __init__(self) -> None:
        self.history: list[dict] = []
        self.lock = asyncio.Lock()


class FeishuBridge:
    def __init__(self, cfg, loop: asyncio.AbstractEventLoop) -> None:
        self.cfg = cfg
        self.loop = loop
        # 单独一个 LLM 实例：人设一样，但收尾规则换成「打字看的」那套，
        # 不能沿用盒子的「两句、四十字以内」。
        self.llm = ClaudeLLM(cfg, build_system_prompt(cfg, channel="chat"))
        self.chats: dict[str, _Chat] = {}
        self.seen: OrderedDict[str, float] = OrderedDict()
        self._token: str = ""
        self._token_exp: float = 0.0
        self._token_lock = asyncio.Lock()
        self._thread: threading.Thread | None = None
        self._stopping = False

    # ---------- 生命周期 ----------

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_ws, name="feishu-ws", daemon=True
        )
        self._thread.start()

    async def stop(self) -> None:
        self._stopping = True

    def _run_ws(self) -> None:
        """在自己的线程里跑飞书 SDK 的长连接。

        SDK 那个 ws 客户端在 import 时就 asyncio.get_event_loop() 抓一个 loop
        存成模块级变量，然后 start() 里 run_until_complete。要是在 aiohttp
        的主线程里 import，它抓到的就是正在跑的那个 loop，一 run_until_complete
        直接报「loop already running」。所以先在本线程建好并设成当前 loop，
        再 import——模块级那个 loop 就是本线程的了。
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            import lark_oapi as lark  # noqa: PLC0415  故意延迟到这里

            handler = (
                lark.EventDispatcherHandler.builder("", "")
                .register_p2_im_message_receive_v1(self._on_event)
                .build()
            )
            client = lark.ws.Client(
                self.cfg.feishu_app_id,
                self.cfg.feishu_app_secret,
                event_handler=handler,
                log_level=lark.LogLevel.INFO,
            )
            log.info("飞书长连接启动中…（app_id=%s）", self.cfg.feishu_app_id)
            client.start()   # 阻塞，内部自己重连
        except Exception as e:
            log.error("飞书长连接挂了：%s —— 盒子不受影响，重启服务再试", e)

    # ---------- 事件 ----------

    def _on_event(self, data) -> None:
        """SDK 在 ws 线程里同步调这个。这里只做筛选和转交，别干重活，
        干重活会把长连接的 ack 拖住。"""
        try:
            msg = data.event.message
            sender = data.event.sender
            open_id = getattr(getattr(sender, "sender_id", None), "open_id", "") or ""
            if getattr(sender, "sender_type", "") not in ("", "user"):
                return
            if (msg.chat_type or "") != "p2p":
                return
            allow = self.cfg.feishu_allowed_open_ids
            if allow and open_id not in allow:
                log.warning("飞书：陌生人 %s 来消息，已忽略", open_id[:12])
                return
            if not self._first_time(msg.message_id or ""):
                return

            if (msg.message_type or "") == "text":
                text = (json.loads(msg.content or "{}").get("text") or "").strip()
                if not text:
                    return
                coro = self._reply_turn(msg.chat_id, text)
            elif (msg.message_type or "") == "audio":
                coro = self._send(msg.chat_id, _AUDIO_HINT)
            else:
                coro = self._send(msg.chat_id, _UNSUPPORTED_HINT)

            asyncio.run_coroutine_threadsafe(coro, self.loop)
        except Exception as e:
            log.error("飞书事件处理出错：%s", e)

    def _first_time(self, message_id: str) -> bool:
        """按 message_id 去重（官方明确说别拿 event_id 当去重键）。"""
        if not message_id:
            return True
        now = time.time()
        if message_id in self.seen:
            return False
        self.seen[message_id] = now
        while len(self.seen) > 512:
            self.seen.popitem(last=False)
        return True

    # ---------- 一轮对话 ----------

    async def _reply_turn(self, chat_id: str, text: str) -> None:
        chat = self.chats.setdefault(chat_id, _Chat())
        async with chat.lock:
            t0 = time.monotonic()
            log.info("飞书收到：%s", text[:80])
            messages = list(chat.history) + [{"role": "user", "content": text}]
            pieces: list[str] = []
            try:
                async for sentence in self.llm.stream_sentences(messages):
                    if sentence == SEARCH_FILLER and not pieces:
                        # 这一轮要联网，先把这句发出去当「正在查」的提示。
                        await self._send(chat_id, "我看一下啊…")
                        continue
                    pieces.append(sentence)
            except Exception as e:
                log.error("飞书这轮 LLM 出错：%s", e)
                await self._send(chat_id, f"这轮我卡住了：{e}")
                return

            reply = _join(pieces)
            if not reply:
                reply = "我没想出什么能说的，换个问法？"
            await self._send(chat_id, reply)

            chat.history.append({"role": "user", "content": text})
            chat.history.append({"role": "assistant", "content": reply})
            keep = max(2, self.cfg.feishu_history_turns * 2)
            if len(chat.history) > keep:
                del chat.history[: len(chat.history) - keep]
            log.info("飞书回完，%.2fs，%d 字", time.monotonic() - t0, len(reply))

    # ---------- 发消息 ----------

    async def _token_get(self) -> str:
        async with self._token_lock:
            if self._token and time.time() < self._token_exp:
                return self._token
            sess = await _http()
            async with sess.post(
                f"{_BASE}/auth/v3/tenant_access_token/internal",
                json={"app_id": self.cfg.feishu_app_id,
                      "app_secret": self.cfg.feishu_app_secret},
            ) as r:
                body = await r.json()
            if body.get("code") != 0:
                raise RuntimeError(f"拿 tenant_access_token 失败：{body}")
            self._token = body["tenant_access_token"]
            self._token_exp = time.time() + max(60, int(body.get("expire", 7200)) - 300)
            return self._token

    async def _send(self, chat_id: str, text: str) -> None:
        if not chat_id or not text:
            return
        try:
            token = await self._token_get()
            sess = await _http()
            async with sess.post(
                f"{_BASE}/im/v1/messages",
                params={"receive_id_type": "chat_id"},
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "receive_id": chat_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": text}, ensure_ascii=False),
                    "uuid": str(uuid.uuid4()),
                },
            ) as r:
                body = await r.json()
            if body.get("code") != 0:
                log.error("飞书发消息失败：%s", body)
        except Exception as e:
            log.error("飞书发消息出错：%s", e)
