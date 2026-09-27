"""Anthropic Claude：流式输出 + 按句切分。

按 ~/.claude/skills/claude-api：
- 默认模型可配（本项目默认 claude-sonnet-5，追求低延迟）
- thinking 必须显式配置：Sonnet 5 / Opus 5 上 budget_tokens 已被移除（400），
  只有 {"type":"adaptive"} 和 {"type":"disabled"} 两种；语音场景默认 disabled 换低延迟，
  想要更强推理就把 XZ_LLM_THINKING 改成 adaptive。
- effort 放在 output_config 里，不是顶层。
- 长输出/流式一律用 client.messages.stream()。

Anthropic 兼容网关（智谱 GLM，2026-09-06 实测 glm-5.3-flash）：
- 认证用 ANTHROPIC_AUTH_TOKEN（SDK auth_token=，走 Authorization: Bearer），
  此时不要设 ANTHROPIC_API_KEY。端点写进 ANTHROPIC_BASE_URL。
- thinking / output_config.effort 两个字段都不报 400，但 thinking 必须显式
  {"type":"disabled"}：不传（或 adaptive）时 GLM 默认先思考，max_tokens 会被
  思考吃光、text_stream 迟迟不出正文——语音场景正好维持默认 disabled。
- 流式（messages.stream + text_stream）正常。

工具调用（2026-09-07 接入）：self.tools 由 app/tools.py 提供（当前只有 SearXNG
联网搜索）。stream_sentences() 遇到 stop_reason == "tool_use" 就把攒着的半句冲出去
（「我查一下」先说掉），跑工具，把 tool_result 追加回对话再来一轮，最多
XZ_TOOL_MAX_ROUNDS 轮。最后一轮不再带 tools，保证一定收敛到正文。
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import logging
import re
import time
from typing import AsyncIterator
from zoneinfo import ZoneInfo

import anthropic

from ..tools import (_WEATHER_Q, PREFETCH_NOTE, run_tool, start_prefetch, tool_definitions,
                     wants_search, wants_search_now, weather_lookup, weather_question,
                     web_search)

log = logging.getLogger("xz.llm")

# 判定要联网时立刻说的一句垫话。别写成「我查一下」——模型有时不查，
# 而是反问「你指哪个城市」，那时「我查一下」听着就前言不搭后语。
SEARCH_FILLER = "我看一下啊。"

# 慢车道（挂着工具问模型）干等太久时的垫话。必须是中性的：宽闸门连
# 「今天心情不好」都会命中，那时说「我看一下啊」就答非所问了。
SLOW_FILLER = "嗯，我想想。"

# 模型挂工具、还没开口时垫的一句
TOOL_FILLER = "我查一下啊。"

# 出错时盒子念的句子（念出来比看日志快）
NO_RESULT = "这个我没查到能说的资料，换个问法试试。"
DECLINE = "这个我这边不方便回答，换一个吧。"
NO_KEY = "服务端的接口钥匙还没配好，去配置文件里填一下。"
QUOTA = "大模型这周的额度用完了，得等它重置。"
BUSY = "模型那边这会儿太挤了，过一会儿再问我。"
DOWN = "我这边连不上服务，稍后再试。"
NET = "网络有点卡，没连上，你再说一遍。"

# 上面这些句子不经模型，模型用方言答、它们却是普通话（9/24 主人问「为什么有时候还是会说
# 普通话」）。盒子通道设了 XZ_VOICE_DIALECT 就换成下面的方言版，飞书照旧用普通话。
# 用词跟着模型平时写的走（个、搭、伊、侬、伐、一歇），这些豆包方言音色念得顺；
# 搿、呒没这类生僻写法没验过，先不用。
DIALECT_LINES = {
    "上海话": {
        SEARCH_FILLER: "我帮侬看一看。",
        SLOW_FILLER: "嗯，我想想看。",
        TOOL_FILLER: "我帮侬查查看。",
        NO_RESULT: "这个我没查到好讲个资料，侬换个讲法试试看。",
        DECLINE: "这个我不大好回答，换一个好伐。",
        NO_KEY: "服务端个接口钥匙还没配好，侬到配置文件里去填一下。",
        QUOTA: "大模型这个礼拜个额度用光了，要等伊重置。",
        BUSY: "模型那边现在交关挤，过一歇再来问我。",
        DOWN: "我这边连不上服务，等一歇再试试看。",
        NET: "网络有点卡，没连上，侬再讲一遍。",
    },
}


def dialect_line(line: str, dialect: str) -> str:
    """固定句换成方言版；不是固定句、或这种方言没配，原样返回。"""
    return DIALECT_LINES.get(dialect, {}).get(line, line)

# 文字版搜索（XZ_LLM_TEXT_SEARCH=1）：模型要查的时候只说这一句，服务端代查。
# 智谱网关一挂原生 tools，首字就从 1.2 秒拖到 4~6 秒（9/23 实测），而绝大多数话根本不用查。
TEXT_SEARCH_RULE = (
    "你能上网：输出暗号就是在上网查，所以绝不要说自己不能联网、查不了。"
    "需要实时信息才能答、而对话里又没有现成资料时，只输出一句暗号，别的一个字都不要说："
    "问天气输出【天气：城市名】；新闻、价格、行情、比分、最近发布的东西等输出【搜索：关键词】。"
    "系统查完会把资料发给你，你再用口语回答。不需要查的直接回答，绝不输出暗号。"
)
_SEARCH_MARK = re.compile(r"^\s*[【\[]\s*(搜索|天气)\s*[:：]\s*(.*?)\s*[】\]]")
_SEARCH_MARK_ANY = re.compile(r"[【\[]\s*(搜索|天气)\s*[:：]")
_WEEK = "一二三四五六日"
_TZ = ZoneInfo("Asia/Shanghai")


# 节日日子由服务端算好给模型：flash 自己算常错（9/23 把「离春节还有多少天」算成 147，实际 136）。
# 公历节日每年固定；农历的写死 2026–2030，过了 2030 这几个就不报了。
_LUNAR = {
    "清明节": ("2026-04-05", "2027-04-05", "2028-04-04", "2029-04-04", "2030-04-05"),
    "端午节": ("2026-06-19", "2027-06-09", "2028-05-28", "2029-06-16", "2030-06-05"),
    "中秋节": ("2026-09-25", "2027-09-15", "2028-10-03", "2029-09-22", "2030-09-12"),
    "春节": ("2026-02-17", "2027-02-06", "2028-01-26", "2029-02-13", "2030-02-03"),
}
_SOLAR = {"元旦": (1, 1), "劳动节": (5, 1), "国庆节": (10, 1)}


def _upcoming_holidays(today: _dt.date, n: int = 4) -> str:
    items = []
    for name, days in _LUNAR.items():
        nxt = next((d for d in map(_dt.date.fromisoformat, days) if d >= today), None)
        if nxt:
            items.append((nxt, name))
    for name, (m, d) in _SOLAR.items():
        nxt = _dt.date(today.year, m, d)
        items.append((nxt if nxt >= today else _dt.date(today.year + 1, m, d), name))
    out = []
    for day, name in sorted(items)[:n]:
        left = (day - today).days
        when = f"{day.year}年{day.month}月{day.day}日" if day.year != today.year else f"{day.month}月{day.day}日"
        out.append(f"{name}{when}（{'就是今天' if left == 0 else f'还有{left}天'}）")
    return "、".join(out)


def now_line() -> str:
    """每轮现算的当前时间。不给这一句，「离春节还有多少天」模型只会说不知道今天几号。"""
    t = _dt.datetime.now(_TZ)
    return (f"现在是北京时间 {t.year}年{t.month}月{t.day}日 星期{_WEEK[t.weekday()]} "
            f"{t.hour}点{t.minute:02d}分。接下来的节日：{_upcoming_holidays(t.date())}。")


# 句末标点：先按这些切
_SENTENCE_END = "。！？!?；;…"
# 第一句允许在这些次级停顿切，抢首句延迟
_SOFT_BREAK = "，,、：:"

_STRIP_FOR_TTS = re.compile(r"[*#`_~\[\]<>\\|]+")
# 英文字母 / 数字：硬切时不能切在这种连续串中间（"RTX" 切成 "R"+"TX" 会念成两截）
_ASCII_WORD = re.compile(r"[0-9A-Za-z]")


def clean_for_tts(text: str) -> str:
    """去掉会被念出来的符号（模型偶尔还是会吐 markdown）。"""
    return _STRIP_FOR_TTS.sub("", text).strip()


class SentenceSplitter:
    """流式文本 → 句子。首句放宽到逗号级并封顶，后续句号级。

    首句是首帧延迟的最后一段：遇到逗号 / 顿号 / 句号即切，到 first_max_chars
    还没标点就硬切。后面的句子仍按句末标点整句切（听感优先）。
    """

    def __init__(
        self,
        first_min_chars: int = 2,
        max_chars: int = 60,
        first_max_chars: int = 15,
    ):
        self.buf = ""
        self.count = 0
        self.first_min_chars = max(0, first_min_chars)
        self.max_chars = max_chars
        self.first_max_chars = max(1, first_max_chars)

    def push(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while True:
            idx = self._find_break()
            if idx < 0:
                break
            piece = self.buf[: idx + 1].strip()
            self.buf = self.buf[idx + 1 :]
            piece = clean_for_tts(piece)
            if piece:
                out.append(piece)
                self.count += 1
        return out

    def _find_break(self) -> int:
        hard = min(
            (self.buf.find(c) for c in _SENTENCE_END if self.buf.find(c) >= 0),
            default=-1,
        )
        if self.count == 0:
            # 首句：软停顿（逗号/顿号/：）和句末标点一视同仁，谁先出现切谁；
            # first_min_chars 起才找，免得被开头的 "嗯，" 切成一个字。
            soft = min(
                (
                    i
                    for i in (self.buf.find(c, self.first_min_chars) for c in _SOFT_BREAK)
                    if i >= 0
                ),
                default=-1,
            )
            cands = [i for i in (hard, soft) if i >= 0]
            if cands:
                return min(cands)
            # 一个标点都没有：到上限就硬切，别让首帧一直等
            if len(self.buf) >= self.first_max_chars:
                return self._safe_cut(self.first_max_chars - 1)
            return -1
        if hard >= 0:
            return hard
        if len(self.buf) >= self.max_chars:
            return self._safe_cut(self.max_chars - 1)
        return -1

    def _safe_cut(self, idx: int) -> int:
        """把硬切点往前挪出英文单词 / 数字串，整串挪不开就照原样切。"""
        if idx + 1 >= len(self.buf):
            # 缓冲正好停在字母 / 数字上，还不知道这个词写完没有：等下一个字符再说
            return -1 if _ASCII_WORD.match(self.buf[idx]) else idx
        i = idx
        while i > 0 and _ASCII_WORD.match(self.buf[i]) and \
                _ASCII_WORD.match(self.buf[i + 1]):
            i -= 1
        return i if i > 0 else idx

    def flush(self) -> str:
        piece = clean_for_tts(self.buf)
        self.buf = ""
        if piece:
            self.count += 1
        return piece


async def split_sentences_stream(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    sp = SentenceSplitter()
    async for delta in chunks:
        for s in sp.push(delta):
            yield s
    tail = sp.flush()
    if tail:
        yield tail


_FILLER_BARE = {re.sub(r"[，,。.！!\s]", "", dialect_line(f, d))
                for f in (SEARCH_FILLER, SLOW_FILLER, TOOL_FILLER) for d in ("", *DIALECT_LINES)}


def _is_filler(sentence: str) -> bool:
    return re.sub(r"[，,。.！!\s]", "", sentence or "") in _FILLER_BARE


def _city_key(c) -> str:
    return re.sub(r"(市|的)?(天气|气温|温度).*$|市$", "", (c or "").strip())


class _Primed:
    """本地流为了掐首字超时已经读走了第一段字，这里把它补回 text_stream 开头。"""

    def __init__(self, stream, first):
        self._s, self._first = stream, first
        self.text_stream = self._text()   # 同一个生成器对象：调用处会在里层接着读

    async def _text(self):
        if self._first:
            yield self._first
        async for t in self._s.text_stream:
            yield t

    async def get_final_message(self):
        return await self._s.get_final_message()


class _LocalFirst:
    """本地 TP3 优先、云端兜底，接口照旧是 client.messages.stream(**params)。

    本地是 TP3 集群上的 vLLM（GLM-5.3-Flash，Anthropic 兼容端点），首字约 0.2 s，
    也没有智谱云端的 1301 内容审核（「这个我这边不方便回答」就是它，9/24 真机）。
    TP3 不归这个服务管，随时可能关机、换模型或者被别的活占满：开头连不上、报错、
    首字超过 XZ_LLM_LOCAL_FIRST_S 都这一轮改走云端，并且一分钟内不再试本地。
    已经开始出字之后才断的不回退——前半句已经念出去了。
    """

    DOWN_S = 60.0

    def __init__(self, local, cloud, cfg):
        self.local, self.cloud = local, cloud
        self.model = cfg.llm_local_model
        self.first_s = float(getattr(cfg, "llm_local_first_s", 2.5))
        self._down_until = 0.0
        self._down = False
        self.messages = self

    def stream(self, **params):
        return _FallbackStream(self, params)

    def local_params(self, params: dict) -> dict:
        p = dict(params)
        p["model"] = self.model
        p.pop("output_config", None)
        # GLM-5.3 的对话模板默认先思考；vLLM 不看 thinking 字段，得从模板参数里关
        eb = dict(p.get("extra_body") or {})
        eb["chat_template_kwargs"] = {"enable_thinking": False}
        p["extra_body"] = eb
        return p

    @property
    def local_ok(self) -> bool:
        return time.monotonic() >= self._down_until

    def mark_down(self, why: str) -> None:
        if not self._down:
            log.warning("本地大脑不可用（%s），%d 秒内改走云端", why, self.DOWN_S)
        self._down = True
        self._down_until = time.monotonic() + self.DOWN_S

    def mark_up(self) -> None:
        if self._down:
            log.info("本地大脑恢复")
        self._down = False


class _FallbackStream:
    def __init__(self, owner: _LocalFirst, params: dict):
        self.owner, self.params = owner, params
        self._mgr = None
        self._local = False

    async def _try_local(self):
        o = self.owner
        mgr = o.local.messages.stream(**o.local_params(self.params))
        try:
            stream = await mgr.__aenter__()
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 or e.status_code == 404:   # 404 = 模型换了
                o.mark_down(f"HTTP {e.status_code}")
            else:   # 4xx 多半是这一句本身的事，不拉黑本地
                log.warning("本地大脑拒了这一句（HTTP %s），这句走云端", e.status_code)
            return None
        except anthropic.APIConnectionError as e:
            o.mark_down(type(e).__name__)
            return None
        try:
            first = await asyncio.wait_for(stream.text_stream.__anext__(), o.first_s)
        except StopAsyncIteration:
            first = None
        except BaseException as e:
            with contextlib.suppress(Exception):
                await mgr.__aexit__(None, None, None)
            if not isinstance(e, Exception):
                raise   # 取消（抢跑作废、被打断）照常往上抛
            o.mark_down(f"{o.first_s:g} 秒没出字" if isinstance(e, TimeoutError)
                        else type(e).__name__)
            return None
        o.mark_up()
        self._mgr, self._local = mgr, True
        return _Primed(stream, first)

    async def __aenter__(self):
        if self.owner.local_ok:
            primed = await self._try_local()
            if primed is not None:
                return primed
        self._mgr = self.owner.cloud.messages.stream(**self.params)
        return await self._mgr.__aenter__()

    async def __aexit__(self, et, e, tb):
        if (self._local and et is not None
                and not issubclass(et, (asyncio.CancelledError, GeneratorExit))):
            self.owner.mark_down(f"中途断了：{et.__name__}")
        if self._mgr is not None:
            return await self._mgr.__aexit__(et, e, tb)
        return False


class ClaudeLLM:
    def __init__(self, cfg, system_prompt: str):
        kwargs = {}
        if cfg.anthropic_auth_token:
            # Bearer 型网关（智谱 GLM 等）：SDK auth_token= → Authorization: Bearer …
            kwargs["auth_token"] = cfg.anthropic_auth_token
        elif cfg.anthropic_api_key:
            kwargs["api_key"] = cfg.anthropic_api_key
        if cfg.anthropic_base_url:
            kwargs["base_url"] = cfg.anthropic_base_url
        # 必须有超时：9/23 实测智谱有一次请求两分钟连响应头都没回，SDK 默认等 10 分钟，
        # 盒子就一直卡在「说话中」。连不上 5 秒、两段数据之间 20 秒算超时，SDK 自己重试两次
        self.client = anthropic.AsyncAnthropic(
            timeout=anthropic.Timeout(20.0, connect=5.0), max_retries=2, **kwargs)
        local_url = getattr(cfg, "llm_local_base_url", "")
        if local_url:
            # 两个 key 都显式给：不给的话 SDK 会从环境变量捡智谱的 token 发给本地
            local = anthropic.AsyncAnthropic(
                base_url=local_url, api_key="local", auth_token="local",
                timeout=anthropic.Timeout(15.0, connect=2.0), max_retries=0)
            self.client = _LocalFirst(local, self.client, cfg)
            log.info("大脑：本地 %s 优先（%s），出错退云端 %s",
                     cfg.llm_local_model, local_url, cfg.llm_model)
        self.cfg = cfg
        self.system_prompt = system_prompt
        self.tools: list[dict] = tool_definitions(cfg)
        self.text_search = bool(self.tools) and getattr(cfg, "llm_text_search", True)
        if self.text_search:
            rule = TEXT_SEARCH_RULE
            if getattr(cfg, "home_city", ""):
                # 不写「主人住在…」：默认城市只是偏好，不等于住址
                rule += f"问天气没说城市，就按{cfg.home_city}查。"
            else:
                rule += "问天气没说城市，就先问主人在哪个城市。"
            self.system_prompt = f"{self.system_prompt}\n- {rule}"
        if self.tools:
            log.info("已启用工具：%s（%s）", ", ".join(t["name"] for t in self.tools),
                     "文字暗号，不挂原生 tools" if self.text_search else "原生 tools")

    def _thinking_param(self) -> dict:
        mode = (self.cfg.llm_thinking or "disabled").lower()
        if mode in ("adaptive", "on", "enabled"):
            return {"type": "adaptive"}
        return {"type": "disabled"}

    async def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        """一问一答拿整段文字：不挂工具、不思考，给播报改写这类后台小活用。本地优先、云端兜底照旧。"""
        params = dict(
            model=self.cfg.llm_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "disabled"},
        )
        out: list[str] = []
        async with self.client.messages.stream(**params) as stream:
            async for delta in stream.text_stream:
                out.append(delta)
        return "".join(out).strip()

    async def stream_sentences(self, messages: list[dict],
                               notes: list[str] | None = None) -> AsyncIterator[str]:
        """流式产出「可以直接送 TTS 的一句话」，中途可能去查一次网。

        notes：传个列表进来，这一轮查到的资料会追加进去，调用方存进上下文——
        不存的话追问「那后天呢」模型只能凭自己上一句话去编（9/23 实测编出了温度）。
        """
        state: dict = {"notes": notes}
        convo = list(messages)
        # 闸门（关键词，见 tools.wants_search）在调模型之前就判得出这句要不要联网。
        gate = (bool(self.tools)
                and int(getattr(self.cfg, "tool_max_rounds", 0)) > 0
                and wants_search(self.cfg, convo))
        # 窄闸门：确实在问外面的事实，值得先查了再答（见 tools.wants_search_now）。
        fast = (gate and getattr(self.cfg, "search_direct", True)
                and wants_search_now(self.cfg, convo))
        try:
            if self.text_search and self.tools:
                city = weather_question(convo)
                home = getattr(self.cfg, "home_city", "")
                if city is not None and (city or home):
                    # 问天气就先在后台查着：9/24「今朝上海天气好吗」模型没说【天气：上海】
                    # 直接开口，温度是它编的。它不说暗号就拿这份数重答（见 _stream_rounds）
                    state["force_weather"] = city or home
                    state["weather_task"] = asyncio.create_task(
                        weather_lookup(self.cfg, state["force_weather"]))
            if fast and self.text_search:
                # 文字暗号模式：不先垫话，让模型先判——电视里的「新闻联播」也会命中窄闸门，
                # 以前先念一句「我看一下啊」才被判成【忽略】。但窄闸门命中（明摆着问外面的事）
                # 就在后台先拿原话去查：模型说【忽略】就撤；说暗号就用这份；它要是不说暗号
                # 直接开口（flash 常这样，还会说「我没联网」，9/23 真机），就当它要查，拿这份重答。
                pf = start_prefetch(self.cfg, convo)
                if pf is not None:
                    state["prefetch"] = pf
                    state["force_query"] = pf.query
                fast = False
            if fast:
                # 这一轮要先查再答，几秒跑不掉。先出一声垫话把静默填上——
                # 以前这几秒完全没动静，用户的原话就是「太慢」。
                # 只在窄闸门里说：宽闸门连「今天心情不好」都算命中，
                # 那种时候冒一句「我看一下啊」纯属莫名其妙。
                yield SEARCH_FILLER
                state["spoke"] = True
            if fast:
                # 快车道：用用户原话直接查，结果塞进对话再问模型。省掉
                # 「挂工具问一轮 → 模型吐 tool_call → 带结果再问一轮」整个来回，
                # 而且这一轮不挂工具，首字延迟也跟着降回闲聊水平（实测省 2~3 秒）。
                pf = start_prefetch(self.cfg, convo)
                if pf is not None:
                    state["prefetch"] = pf
                    hit = await pf.take(pf.query)
                    state.pop("prefetch", None)
                    if hit:
                        convo = convo + [{"role": "user",
                                          "content": hit + PREFETCH_NOTE}]
                        gate = False   # 结果已经在手上，不用再给模型工具
                        state["searched"] = True
            agen = self._stream_rounds(convo, state, gate and not self.text_search)
            first_task = None
            try:
                wait_ms = 0 if fast else int(getattr(self.cfg, "llm_filler_after_ms", 0))
                if (gate or self.text_search) and wait_ms > 0:
                    # 挂上工具那一轮的首字延迟能拖到十秒（日志里真出现过
                    # 'LLM 首句 10.37s'），这段时间盒子那头一点动静都没有，
                    # 用户的原话就是「回复很慢」。等超过阈值就先垫一句。
                    first_task = asyncio.ensure_future(agen.__anext__())
                    done, _ = await asyncio.wait({first_task}, timeout=wait_ms / 1000)
                    if not done:
                        log.info("慢车道干等 %d ms 还没出字，先垫一句", wait_ms)
                        yield SLOW_FILLER
                        state["spoke"] = True
                    try:
                        yield await first_task
                    except StopAsyncIteration:
                        return
                    finally:
                        first_task = None
                async for sentence in agen:
                    yield sentence
            finally:
                if first_task is not None and not first_task.done():
                    first_task.cancel()
                    with contextlib.suppress(BaseException):
                        await first_task
                with contextlib.suppress(Exception):
                    await agen.aclose()
        finally:
            # 抢跑最后没用上（模型压根没查，或者这一轮提前收工）就撤掉
            pf = state.pop("prefetch", None)
            if pf is not None:
                pf.cancel()
            wt = state.pop("weather_task", None)
            if wt is not None and not wt.done():
                wt.cancel()

    async def _stream_rounds(self, convo: list[dict], state: dict,
                             gate: bool) -> AsyncIterator[str]:
        rounds = max(0, int(getattr(self.cfg, "tool_max_rounds", 0))) + 1

        for round_idx in range(rounds):
            params = dict(
                model=self.cfg.llm_model,
                max_tokens=self.cfg.llm_max_tokens,
                system=f"{self.system_prompt}\n\n{now_line()}",
                messages=convo,
                thinking=self._thinking_param(),
            )
            if self.cfg.llm_effort:
                params["output_config"] = {"effort": self.cfg.llm_effort}
            # 最后一轮不给工具，逼模型直接说话，避免来回打转；
            # 闲聊也不挂工具，挂上会平白多一两秒首字延迟（见 tools.wants_search）。
            if self.tools and not self.text_search and round_idx < rounds - 1 and \
                    (round_idx > 0 or gate):
                params["tools"] = self.tools

            # 首句遇到逗号/顿号/句号即切、上限 XZ_LLM_FIRST_MAX_CHARS 字（默认 15）；
            # 只影响第一句，后续仍是句末标点整句切。
            sp = SentenceSplitter(
                first_min_chars=self.cfg.llm_first_min_chars,
                first_max_chars=self.cfg.llm_first_max_chars,
            )
            # 垫话在 stream_sentences 里已经说过了，别再重复一次
            spoke = bool(state.get("spoke"))
            final = None
            if round_idx == 0 and gate and "prefetch" not in state:
                # 慢车道（快车道关掉时）：让抢跑跟模型的思考并行
                pf = start_prefetch(self.cfg, convo)
                if pf is not None:
                    state["prefetch"] = pf
            # 文字版搜索：开头先攒着，看是不是【搜索：…】。是就别念，拿去查
            marker_buf = "" if (self.text_search and not state.get("searched")
                                and round_idx < rounds - 1) else None
            query = None
            kind = "搜索"
            try:
                async with self.client.messages.stream(**params) as stream:
                    async for delta in stream.text_stream:
                        if marker_buf is not None:
                            marker_buf += delta
                            head = marker_buf.lstrip()
                            if not head:
                                continue
                            m = _SEARCH_MARK.match(head)
                            if m:
                                # 不 break：模型按规定说完暗号就停了，把流读完让 SDK 正常收尾
                                # （break 出 async for 会留下没关的生成器，退出时 asyncio 报错）
                                kind, query = m.group(1), m.group(2).strip()
                                marker_buf = None
                                async for _ in stream.text_stream:
                                    pass
                                break
                            if head[0] in "【[" and len(head) < 40 and not re.search(r"[】\]]", head):
                                continue  # 还没等到右括号
                            if ((state.get("force_query") or "weather_task" in state)
                                    and round_idx == 0
                                    and head[0] not in "【[" and not head.startswith("忽略")):
                                if "忽略".startswith(head):
                                    continue  # 可能是不带括号的「忽略」，再等一个字
                                # 窄闸门命中、模型却没说暗号就开口：这段不念，拿后台查到的重答
                                if "weather_task" in state:
                                    kind, query = "天气", state["force_weather"]
                                else:
                                    kind, query = "搜索", state["force_query"]
                                marker_buf = None
                                log.info("模型没说暗号就开口（%r），改成先查%s再答", head[:12], kind)
                                break
                            delta, marker_buf = marker_buf, None
                        for sentence in sp.push(delta):
                            if _SEARCH_MARK_ANY.search(sentence):
                                continue  # 不该出现的暗号别念出来
                            if state.get("spoke") and _is_filler(sentence):
                                continue  # 垫话已经说过了，模型照着又说一遍的不念
                            spoke = True
                            yield sentence
                    if query is None:
                        final = await stream.get_final_message()
            except anthropic.APIStatusError as e:
                log.error("Claude API 错误 %s: %s", e.status_code, e)
                tail = sp.flush()
                if tail:
                    yield tail
                if e.status_code == 400 and "1301" in str(e):
                    # 智谱内容审核：多半是搜回来的网页里有敏感词，不是网络问题
                    yield (NO_RESULT if state.get("searched") else DECLINE)
                elif e.status_code in (401, 403):
                    # 让盒子把问题念出来，比看日志快
                    yield NO_KEY
                elif e.status_code == 429:
                    # 智谱的 429 多半不是「太密」，而是 1310「每周/每月额度用完」
                    # （9/10 日志里 6 次全是它），SDK 已经自己重试过两次了
                    if "1310" in str(e) or "上限" in str(e):
                        yield QUOTA
                    else:
                        yield BUSY
                else:
                    yield DOWN
                return
            except anthropic.APIConnectionError as e:
                log.error("Claude 连接失败: %s", e)
                yield NET
                return

            if query is not None:
                if not spoke and not state.get("spoke"):
                    yield SEARCH_FILLER
                    state["spoke"] = True
                out = None
                wt = state.pop("weather_task", None)
                if kind == "搜索" and wt is not None and _WEATHER_Q.search(query):
                    kind = "天气"   # 用【搜索：上海天气】问天气的，也走天气接口
                if kind == "天气":
                    # 不退回网页搜索：搜出来的是一堆「XX天气预报_2345」标题，没有数，
                    # 还踩过一次智谱内容审核（1301）
                    if wt is not None and _city_key(query) == _city_key(state.get("force_weather")):
                        out = await wt
                        log.info("模型要查天气「%s」，后台已经查好", query)
                    else:
                        log.info("模型要查天气「%s」", query)
                        out = await weather_lookup(self.cfg, query)
                if wt is not None and not wt.done():
                    wt.cancel()
                if kind == "天气":
                    home = getattr(self.cfg, "home_city", "")
                    if not out and home:
                        out = await weather_lookup(self.cfg, home)
                    if not out:
                        out = "没查到这个地方的天气（可能没说城市，或者城市名不对）。请只问一句用户在哪个城市。"
                pf = state.pop("prefetch", None)
                if out and pf is not None:
                    pf.cancel()   # 天气已经有数了，抢跑的网页搜索用不上
                if not out and pf is not None:
                    out = await pf.take(query)
                    if out:
                        log.info("模型要查「%s」，抢跑命中", query)
                if not out:
                    log.info("模型要查「%s」", query)
                    out = await web_search(self.cfg, query)
                state["searched"] = True
                if state.get("notes") is not None:
                    state["notes"].append(out)
                # 资料直接拼进用户那句话，比「助手说暗号 + 用户贴资料」两段式快
                # （9/23 实测 glm-5.3 首字中位 1.07 s 对 1.53 s）
                last = convo[-1] if convo else {}
                if last.get("role") == "user" and isinstance(last.get("content"), str):
                    convo[-1] = {"role": "user",
                                 "content": f"{last['content']}\n\n（系统替你查到的资料）\n{out}"}
                else:
                    convo.append({"role": "assistant", "content": f"【{kind}：{query}】"})
                    convo.append({"role": "user", "content": f"（系统替你查到的资料）\n{out}"})
                continue
            if marker_buf:
                # 整轮都没等到右括号（比如只回了个「【」），当普通文字冲出去
                for sentence in sp.push(marker_buf):
                    spoke = True
                    yield sentence

            stop = getattr(final, "stop_reason", None)

            if stop == "refusal":
                log.warning("Claude 拒答：%s", getattr(final, "stop_details", None))
                tail = sp.flush()
                if tail:
                    yield tail
                yield DECLINE
                return

            if stop == "tool_use":
                # 把半句先说完，查询期间盒子不至于干等着没声音
                tail = sp.flush()
                if tail:
                    spoke = True
                    yield tail
                calls = [b for b in (final.content or [])
                         if getattr(b, "type", None) == "tool_use"]
                if not calls:
                    log.warning("stop_reason=tool_use 但没有 tool_use 块，收工")
                    return
                if not spoke:
                    yield TOOL_FILLER
                convo.append({"role": "assistant",
                              "content": _assistant_blocks(final.content)})
                results = []
                for call in calls:
                    args = call.input if isinstance(call.input, dict) else {}
                    out = None
                    pf = state.pop("prefetch", None)
                    if pf is not None and call.name == "web_search":
                        q = (args.get("query") or "").strip()
                        out = await pf.take(q)
                        log.info("抢跑%s「%s」", "命中" if out else "没对上", q)
                    elif pf is not None:
                        pf.cancel()
                    if out is None:
                        log.info("工具调用 %s(%s)", call.name, args)
                        out = await run_tool(self.cfg, call.name, args)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": out,
                    })
                convo.append({"role": "user", "content": results})
                continue

            tail = sp.flush()
            if tail:
                spoke = True
                yield tail
            if not spoke:
                log.warning("这一轮模型什么都没说（stop_reason=%s）", stop)
            return


def _assistant_blocks(content) -> list[dict]:
    """把 SDK 的返回块转成纯 dict，顺手丢掉空 text（有的网关会 400）。"""
    out: list[dict] = []
    for b in content or []:
        kind = getattr(b, "type", None)
        if kind == "text":
            if (b.text or "").strip():
                out.append({"type": "text", "text": b.text})
        elif kind == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name,
                        "input": b.input if isinstance(b.input, dict) else {}})
    return out
