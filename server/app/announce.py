"""Claude Code 任务播报：别处 POST 进来 → 按设备排队 → 盒子待机时长轮询收件箱，
有货就自己「唤醒」连上来（固件 WakeWordInvoke("__announce__")，WS 里是
listen detect text=__announce__），会话里把队列合成一段念掉。

- 第一条到了先等 merge_s 秒再放行，期间来的合成一段念；
- ttl_s 秒还没念出去就扔（盒子离线 / 一直在对话）；
- quiet 时段来的直接丢，盒子那段时间也不轮询（X-Poll-After 让它睡到时段结束）；
- 盒子有 WS 会话（正在对话）时不放行，等它回待机再念。
队列只在内存里：服务重启就清空，播报本来就是「过时就没意义」的东西。
"""
from __future__ import annotations

import asyncio
import collections
import datetime as dt
import logging
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

log = logging.getLogger("xz.announce")

WAKE_TEXT = "__announce__"

_MAX_QUEUE = 20          # 每台设备最多压这么多条，多了丢最老的
_MAX_TEXT = 160          # 单条最多念这么多字
_MAX_READ = 4            # 一次最多念几条，其余报个数
_MAX_OFFERS = 4          # 连着叫醒这么多次都没来领，说明盒子那边有问题，扔掉
_RATE_WINDOW_S = 600.0
_RATE_MAX = 30           # 10 分钟最多收 30 条（token 泄漏或上游死循环时兜底）
_ORD = ("第一条", "第二条", "第三条", "第四条")
_CN = "零一两三四五六七八九十"
_REWRITE_S = 12.0        # 改写最多等这么久：入队就在后台跑，合并窗口 10 秒里一般早就好了
_REWRITE_WAIT_S = 3.0    # 盒子来领时还没改好，最多再等这么久，过了就念原文
_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*")

# 播报原文是 ClaudeVoice 写的书面普通话，豆包照着念常常串成普通话（9/25 探针：播报 20 段里 8 段）。
# 先改写成方言口语再念，同样的句子 20 段里 0 段串（tools/dialect_probe.py 的 rw1）。
_REWRITE_RULES = """你把一条发给主人的任务提醒改写成{dialect}口语，交给语音合成直接念出来。
规则：
- 意思不变：称呼、项目名、结果、要主人做的事都要留着；「目录」两个字照原样留在项目名前面；「具体是」这类套话可以去掉。
- 用{dialect}平时讲话的说法和写法{words}，像当地人讲话，不要书面语。
- 只用常见的方言字，不要用搿、呒没、㑚这类生僻字。
- 项目名、英文、命令、文件名、数字照原样写，不要翻译。
- 可以加一两个口头禅或感叹、加标点和语气词带出情绪{flavor}；除此以外不加原文没有的内容，不要客套。
- 只输出改写后的话，不要引号，不要解释。"""
_REWRITE_EXAMPLES = {
    "上海话": """
例子：
原文：老孙，目录 ESP 盒子 的任务完成了。具体是，strix 重启完成，虚拟内存设置已核对。
改写：老孙，目录 ESP 盒子个任务做好了，老卵！strix 重新开好了，虚拟内存个设置也对过了。
原文：布莱特，目录 临时工作区 需要你授权。请确认是否启用此功能。
改写：布莱特，目录 临时工作区要侬授权一下噢，侬看看这个功能要勿要开？
原文：老孙，目录 NAS 的任务失败了。原因是网络中断，需要你检查网线。
改写：册那！老孙，目录 NAS 个任务没弄好，网络断脱了，侬去看一看网线呀。""",
}


def rewrite_prompt(dialect: str, words: str = "", flavor: str = "") -> str:
    return _REWRITE_RULES.format(dialect=dialect, words=f"（比如{words}）" if words else "",
                                 flavor=f"（{flavor}）" if flavor else "") \
        + _REWRITE_EXAMPLES.get(dialect, "")


def check_rewrite(orig: str, out: str, project: str = "") -> str:
    """改写结果像样就返回清理过的；不像样返回空串，调用处念原文。"""
    out = (out or "").strip()
    if not out or "\n" in out:
        return ""   # 多行 = 带了解释
    out = re.sub(r"^(改写|改写后|输出)\s*[:：]\s*", "", out).strip("「」“”\"' ")
    out = re.sub(r"\s+", " ", out)
    if not out or re.search(r"[【】#*]", out):
        return ""
    if len(out) > max(len(orig) * 1.3, len(orig) + 12) or len(out) < len(orig) * 0.35:
        return ""   # 长出一截是在加戏，短得太多是丢了内容
    low, flat = out.lower(), out.replace(" ", "")
    if any(w.lower() not in low for w in _ASCII_WORD.findall(orig) if len(w) >= 2):
        return ""   # 英文名、命令漏了或被翻译了
    proj = (project or "").replace(" ", "")
    if proj and proj in orig.replace(" ", "") and proj not in flat:
        return ""   # 项目名丢了
    if "目录" in orig and "目录" not in out:
        # 「目录」两个字要留（9/25 用户定）：模型省掉了就补回项目名前面，找不到项目名就念原文
        m = re.search(r"\s*".join(map(re.escape, proj)), out) if proj else None
        if not m:
            return ""
        out = f"{out[:m.start()]}目录 {out[m.start():]}"
    return _clean(out)


@dataclass
class _Item:
    at: float       # time.monotonic()
    text: str
    kind: str
    project: str
    say: str = ""                        # 改写成方言口语的版本；空 = 念原文
    job: asyncio.Task | None = None      # 改写任务


def _parse_quiet(spec: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", spec or "")
    if not m:
        return None
    a = int(m[1]) * 60 + int(m[2])
    b = int(m[3]) * 60 + int(m[4])
    return None if a == b else (a, b)


def _clean(text: str) -> str:
    """压空白；太长就在句号（其次逗号）处截断，别念到半截。"""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= _MAX_TEXT:
        return text
    cut = text[:_MAX_TEXT]
    end = max(cut.rfind(p) for p in "。！？；!?;")
    if end >= 40:
        return cut[: end + 1]
    end = max(cut.rfind(p) for p in "，,、")
    return (cut[:end] if end >= 40 else cut) + "。"


def _cn(n: int) -> str:
    return _CN[n] if 0 <= n < len(_CN) else str(n)


def compose(texts: list[str], spoken: bool = False) -> str:
    """多条合成一段口播。单条原样念。spoken=True：方言口语版，数字写成汉字。"""
    uniq: list[str] = []
    for t in texts:
        if t and t not in uniq:
            uniq.append(t)
    if not uniq:
        return ""
    if len(uniq) == 1:
        return uniq[0]
    parts = [f"有{_cn(len(uniq)) if spoken else len(uniq)}条提醒。"]
    for i, t in enumerate(uniq[:_MAX_READ]):
        parts.append(f"{_ORD[i]}，{t.rstrip('。！？!?.')}。")
    if len(uniq) > _MAX_READ:
        more = len(uniq) - _MAX_READ
        # 念的那版用上海话讲法：「手机上」讲「手机浪」，「看吧」讲「看好了」（9/27 用户要上海话语义对）
        parts.append(f"还有{_cn(more)}条，侬到手机浪看好了。" if spoken else f"还有{more}条，去手机上看吧。")
    return "".join(parts)


class AnnounceHub:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tz = ZoneInfo(cfg.announce_tz or "Asia/Shanghai")
        self.quiet = _parse_quiet(cfg.announce_quiet)
        self._q: dict[str, list[_Item]] = {}
        self._offers: dict[str, int] = {}
        self._last_offer: dict[str, float] = {}
        self._live: dict[str, int] = {}
        self._recent: collections.deque[float] = collections.deque()
        self._cond = asyncio.Condition()
        # 原文 → 方言口语；main 里配了 XZ_VOICE_DIALECT 才接上大模型，None = 原样念
        self.rewrite: Callable[[str], Awaitable[str]] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.announce_token)

    # ---------- 时段 ----------

    def quiet_left_s(self, now: dt.datetime | None = None) -> float:
        """现在在静默时段里就返回离结束还有几秒，否则 0。"""
        if not self.quiet:
            return 0.0
        now = now or dt.datetime.now(self.tz)
        a, b = self.quiet
        m = now.hour * 60 + now.minute + now.second / 60
        inside = (a <= m < b) if a < b else (m >= a or m < b)
        if not inside:
            return 0.0
        left = (b - m) % (24 * 60)
        return max(left * 60, 1.0)

    # ---------- 队列 ----------

    def _keys(self, device: str) -> tuple[str, ...]:
        return (device, "*") if device != "*" else ("*",)

    def _purge(self, device: str) -> None:
        cutoff = time.monotonic() - self.cfg.announce_ttl_s
        for k in self._keys(device):
            q = self._q.get(k)
            if not q:
                continue
            fresh = [it for it in q if it.at >= cutoff]
            if len(fresh) != len(q):
                log.info("播报过期扔掉 %d 条（%s，%.0f 秒没念出去）",
                         len(q) - len(fresh), k, self.cfg.announce_ttl_s)
            if fresh:
                self._q[k] = fresh
            else:
                self._q.pop(k, None)
                self._offers.pop(k, None)
                self._last_offer.pop(k, None)

    def _items(self, device: str) -> list[_Item]:
        out: list[_Item] = []
        for k in self._keys(device):
            out.extend(self._q.get(k, ()))
        return sorted(out, key=lambda it: it.at)

    def _ready_at(self, device: str) -> float | None:
        items = self._items(device)
        return items[0].at + self.cfg.announce_merge_s if items else None

    async def add(self, text: str, kind: str = "", project: str = "",
                  device: str | None = None) -> tuple[int, dict]:
        text = _clean(text)
        if not text:
            return 400, {"ok": False, "error": "empty text"}
        quiet = self.quiet_left_s()
        if quiet:
            log.info("静默时段，播报不排队：[%s/%s] %s", kind, project, text)
            return 200, {"ok": True, "queued": False, "reason": "quiet"}
        now = time.monotonic()
        while self._recent and now - self._recent[0] > _RATE_WINDOW_S:
            self._recent.popleft()
        if len(self._recent) >= _RATE_MAX:
            log.warning("播报太密（%d 分钟 %d 条），这条不收：%s",
                        _RATE_WINDOW_S // 60, _RATE_MAX, text)
            return 429, {"ok": False, "error": "too many"}
        self._recent.append(now)

        allowed = self.cfg.allowed_device_ids
        if device:
            device = device.strip().lower()
            if allowed and device not in allowed:
                return 404, {"ok": False, "error": "unknown device"}
            targets = [device]
        else:
            targets = list(allowed) or ["*"]
        item = _Item(now, text, kind, project)
        async with self._cond:
            for dev in targets:
                q = self._q.setdefault(dev, [])
                q.append(item)
                if len(q) > _MAX_QUEUE:
                    del q[: len(q) - _MAX_QUEUE]
            self._cond.notify_all()
        log.info("播报入队 → %s：[%s/%s] %s", ",".join(targets), kind, project, text)
        if self.rewrite is not None:
            item.job = asyncio.create_task(self._rewrite(item))
        return 200, {"ok": True, "queued": True, "devices": len(targets)}

    async def _rewrite(self, item: _Item) -> None:
        t0 = time.monotonic()
        try:
            raw = await asyncio.wait_for(self.rewrite(item.text), _REWRITE_S)
        except Exception as e:  # noqa: BLE001
            log.warning("播报改写失败（%s），念原文", type(e).__name__)
            return
        out = check_rewrite(item.text, raw, item.project)
        if not out:
            log.warning("播报改写结果不像样，念原文：%r", (raw or "")[:160])
            return
        item.say = out
        log.info("播报改写 %.1fs：%s → %s", time.monotonic() - t0, item.text, out)

    def _pop(self, device: str) -> list[_Item]:
        device = (device or "").lower()
        self._purge(device)
        items = self._items(device)
        for k in self._keys(device):
            self._q.pop(k, None)
            self._offers.pop(k, None)
            self._last_offer.pop(k, None)
        return items

    def take(self, device: str) -> str:
        """取走这台设备的全部播报，合成一段原文。"""
        return compose([it.text for it in self._pop(device)])

    async def take_speech(self, device: str) -> tuple[str, str]:
        """会话里收到 __announce__ 时调：取走全部播报，返回（要念的话, 原文）。

        改写还没回来的最多再等 _REWRITE_WAIT_S 秒，过了就那条念原文。
        """
        items = self._pop(device)
        jobs = [it.job for it in items if it.job is not None and not it.job.done()]
        if jobs:
            await asyncio.wait(jobs, timeout=_REWRITE_WAIT_S)
        orig = compose([it.text for it in items])
        if not any(it.say for it in items):
            return orig, orig
        return compose([it.say or it.text for it in items], spoken=True), orig

    # ---------- 会话进出 ----------

    async def session_opened(self, device: str) -> None:
        device = (device or "").lower()
        async with self._cond:
            self._live[device] = self._live.get(device, 0) + 1
            self._cond.notify_all()

    async def session_closed(self, device: str) -> None:
        device = (device or "").lower()
        async with self._cond:
            n = self._live.get(device, 0) - 1
            if n > 0:
                self._live[device] = n
            else:
                self._live.pop(device, None)
            self._cond.notify_all()

    # ---------- 长轮询 ----------

    async def poll(self, device: str, wait_s: float,
                   alive: Callable[[], bool] | None = None) -> tuple[int, int]:
        """返回 (HTTP 状态, 建议下次隔几秒再来)。200 = 有播报，叫它醒来连上来。

        alive：连接还在不在。盒子重启 / 换网时旧的长轮询会在这里挂到超时，
        它拿到 200 也送不到，不能算一次叫醒。
        """
        device = (device or "").lower()
        quiet = self.quiet_left_s()
        if quiet:
            return 204, int(min(quiet, 3600))
        deadline = time.monotonic() + wait_s
        async with self._cond:
            live_at_start = self._live.get(device, 0)
            while True:
                now = time.monotonic()
                self._purge(device)
                live = self._live.get(device, 0)
                if live > live_at_start:
                    # 等的这会儿有人把它叫醒说话了：先放掉这条连接，对话完它会再来
                    return 204, 5
                ready = self._ready_at(device)
                if ready is not None and now >= ready and live == 0:
                    if alive is not None and not alive():
                        return 204, 1
                    key = device if self._q.get(device) else "*"
                    n = self._offers.get(key, 0)
                    if now - self._last_offer.get(key, 0.0) > 5:   # 同一时刻好几条连接只算一次
                        n = self._offers[key] = n + 1
                        self._last_offer[key] = now
                    if n > _MAX_OFFERS:
                        log.warning("叫醒 %d 次都没来领（%s），这批播报扔掉", _MAX_OFFERS, device)
                        for k in self._keys(device):
                            self._q.pop(k, None)
                            self._offers.pop(k, None)
                            self._last_offer.pop(k, None)
                        continue
                    log.info("收件箱：%s 有 %d 条播报，叫它醒（第 %d 次）",
                             device, len(self._items(device)), n)
                    return 200, 0
                timeout = deadline - now
                if ready is not None and ready > now:
                    timeout = min(timeout, ready - now + 0.05)
                if timeout <= 0:
                    return 204, 1
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout)
                except asyncio.TimeoutError:
                    pass
