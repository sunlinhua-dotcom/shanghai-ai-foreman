"""LLM 可调用的工具：目前只有联网搜索。

为什么放在自建后端而不是走小智官方的 MCP：这台盒子已经连的是我们自己的
xiaozhi-protocol 服务，LLM 请求也在我们手里，直接在服务端做函数调用最短路——
不用把搜索服务暴露到公网，也不依赖灵矽控制台。

两个后端，用 XZ_SEARCH_PROVIDER 切：
  tavily  （默认）云端搜索 API，自带一段合成好的 answer，中文问题也答得准；
          需要 TAVILY_API_KEY，key 只在 .env 里，绝不进代码和日志。
  searxng 自建实例（XZ_SEARXNG_URL），建议走内网 / tailnet 直连，
          不经代理、不上公网。Tavily 额度用完或不想外发查询词时切回它。
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import email.utils
import hashlib
import json
import logging
import re
import time

import aiohttp

from .config import SERVER_DIR

log = logging.getLogger("xz.tools")
_BJ = _dt.timezone(_dt.timedelta(hours=8))

# 每次 aiohttp.ClientSession(...) 都要重做 DNS 解析 + TLS 握手。实测同一台 EX10
# 上，新建 session 查一次 Tavily 要 2.1~2.6s，复用连接只要 0.5~1.0s——白扔 1.3 秒，
# 而一轮搜索总共才几秒。所以整个进程共用一个 session，连接留着不关。
_session: aiohttp.ClientSession | None = None
_session_lock = asyncio.Lock()


async def _http() -> aiohttp.ClientSession:
    global _session
    if _session is not None and not _session.closed:
        return _session
    async with _session_lock:
        if _session is None or _session.closed:
            # trust_env=False：这台机器有代理环境变量，搜索一律直连。
            _session = aiohttp.ClientSession(
                trust_env=False,
                connector=aiohttp.TCPConnector(limit=8, ttl_dns_cache=300,
                                               keepalive_timeout=120),
            )
    return _session


async def close_http() -> None:
    """进程退出时收尾，避免 aiohttp 打「Unclosed session」。"""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


WEB_SEARCH = {
    "name": "web_search",
    "description": (
        "联网搜索，返回若干条网页摘要。遇到你不知道、可能已经变了、或者需要今天/最近"
        "信息的问题就调用它：新闻、天气、价格、比分、股价、版本号、某人现在在做什么、"
        "任何带时间的事实。只查一次，拿到结果直接用口语回答，别念网址。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，用中文或英文都行，写成搜索引擎里会打的样子。",
            },
        },
        "required": ["query"],
    },
}


def _provider_ready(cfg) -> bool:
    if cfg.search_provider == "tavily":
        # 多个 key 存在 TAVILY_API_KEYS 里；一个都没有也还能退回 SearXNG
        return bool(cfg.tavily_api_key or getattr(cfg, "tavily_api_keys", None) or cfg.searxng_url)
    return bool(cfg.searxng_url)


def tool_definitions(cfg) -> list[dict]:
    if not (cfg.search_enabled and _provider_ready(cfg)):
        return []
    return [WEB_SEARCH]


# 挂上 tools 会把首字延迟从 ~1s 抬到 ~3s（GLM 网关实测），而绝大多数闲聊根本不用查。
# 所以先用一层关键词闸门：像是要查东西的句子才把工具递给模型，剩下的走快车道。
# 宁可多放行（多花 1~2 秒），不要漏（漏了就是瞎编）。XZ_SEARCH_ALWAYS=1 可以关掉闸门。
_SEARCH_HINT = re.compile(
    r"查一下|查查|查下|搜一下|搜搜|搜索|搜下|帮我查|上网|网上|百度|谷歌|google|"
    r"今天|今日|今晚|现在|目前|当前|最新|最近|刚刚|昨天|明天|后天|本周|这周|"
    r"今年|去年|明年|几点|几号|日期|星期几|礼拜几|多久|截止|"
    r"天气|气温|下雨|降雨|台风|空气质量|限行|路况|"
    r"新闻|头条|发生了什么|进展|舆论|"
    r"股价|股票|行情|汇率|油价|金价|币价|比特币|涨了|跌了|"
    r"多少钱|价格|售价|房价|票价|优惠|"
    r"比分|赛程|排名|冠军|夺冠|票房|上映|发售|发布会|新款|版本号|最新版",
    re.IGNORECASE,
)


# 上面那张表故意放得宽（宁可多挂一次工具），代价是「今天心情不好」「我最近
# 老失眠」这种纯闲聊也会命中——只是个时间副词而已。快车道要先查再答，命中
# 空气就白等两三秒，所以它只认下面这张「问的确实是外面的事实」的窄表：得有
# 实打实的话题词或者明说要查，光有「今天 / 现在 / 最近」不算。
_SEARCH_TOPIC = re.compile(
    r"查一下|查查|查下|搜一下|搜搜|搜索|搜下|帮我查|上网查|百度|谷歌|google|"
    r"天气|气温|下雨|降雨|台风|空气质量|限行|路况|"
    r"新闻|头条|发生了什么|票房|上映|发售|发布会|"
    r"发布|发表|上线|开源|官宣|新版本|最新版|上新|新出|新款|"
    r"有什么新|有没有新|出了什么|评测|获奖|排行榜|销量|市值|财报|收购|融资|"
    r"股价|股票|行情|汇率|油价|金价|币价|比特币|"
    r"多少钱|价格|售价|房价|票价|"
    r"比分|赛程|排名|冠军|夺冠|版本号|最新版",
    re.IGNORECASE,
)
# （「几点 / 几号 / 星期几」以前也在这张表里。9/23 起提示词里每轮都带当前时间，
#   这类问题模型自己就能答，先查再答反倒白等两秒。）


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = [b.get("text", "") for b in c
                     if isinstance(b, dict) and b.get("type") == "text"]
            if parts:
                return " ".join(parts)
        return ""
    return ""


# --- 上海话 → 普通话 ---------------------------------------------------------
# 9/24 起盒子说上海话，主人也跟着讲，ASR 照音写成「今朝」「伐」「高头」「哪能」。
# 关键词闸门、Tavily 的时间范围、抢跑的查询词都只认普通话：「今朝有啥新闻」被判成近一周，
# 「今朝习近平去美国了伐」闸门一个词都不中，模型写进暗号的「今朝最新AI新闻」也原样发给了 Tavily。
# 只换跟「查什么、查哪天」有关的词；侬 / 阿拉 / 伊这类代词不换（阿拉伯、伊朗会被换坏）。
_WU2PTH = (
    ("啥辰光", "什么时候"), ("辰光", "时候"), ("今朝", "今天"), ("昨日", "昨天"),
    ("后日", "后天"), ("前日", "前天"), ("夜到", "晚上"), ("上半天", "上午"), ("下半天", "下午"),
    ("啥价钿", "什么价格"), ("几钿", "多少钱"), ("价钿", "价格"), ("几化", "多少"),
    ("落雨", "下雨"), ("落雪", "下雪"), ("哪能", "怎么样"), ("啥地方", "哪里"), ("啥", "什么"),
    ("阿是", "是不是"), ("高头", "上面"), ("里向", "里面"),
)
# 「明朝」也是明代：后面跟着天气 / 时间 / 问句才当「明天」
_WU_MINGZHAO = re.compile(
    r"明朝(?=天气|会|要|下雨|下雪|有|几|早|上午|下午|夜|晚|去|来|礼拜|星期|冷|热|温度|吗|伐|[，,。?？!！\s]|$)")
_WU_LIBAI = re.compile(r"礼拜(?=[一二三四五六日天几])")
_WU_Q_END = re.compile(r"(?<![砍讨征攻步北杀挞])伐(?=[\s，,。.！!？?]|$)")
_FILLER_WORD = re.compile(r"(?:^|(?<=[，,。\s]))(?:呃|嗯|啊|额|唔|哎|诶|那个)(?=[，,。\s]|$)")
_ASK_PREFIX = re.compile(
    r"^(?:请|麻烦)?(?:你|您|侬)?(?:帮我|帮我们|替我|给我)?(?:上网|网上)?"
    r"(?:搜索|搜一下|搜一搜|搜搜|搜下|查一下|查一查|查查|查下|看一下|看一看|看看|看下|找一下|找找)"
    r"(?:一下)?看?")
_EDGE_PUNCT = re.compile(r"^[\s，,。.！!？?、:：]+|[\s，,。.！!？?、:：]+$")


def normalize_query(text: str) -> str:
    """上海话的时间 / 问句词换成普通话，去掉句首「帮我搜一下」和夹在中间的「呃」。"""
    raw = (text or "").strip()
    q = raw
    for wu, pth in _WU2PTH:
        q = q.replace(wu, pth)
    q = _WU_MINGZHAO.sub("明天", q)
    q = _WU_LIBAI.sub("星期", q)
    q = _WU_Q_END.sub("吗", q)
    q = _FILLER_WORD.sub("", q)
    q = _EDGE_PUNCT.sub("", q)
    q = _EDGE_PUNCT.sub("", _ASK_PREFIX.sub("", q))
    return q or raw


def user_query(messages: list[dict]) -> str:
    return normalize_query(_last_user_text(messages))


def wants_search(cfg, messages: list[dict]) -> bool:
    """这一轮要不要把搜索工具递给模型。"""
    if getattr(cfg, "search_always", False):
        return True
    text = user_query(messages)
    # 宽表必须是窄表的超集：窄表命中却宽表不中（「苹果发布新手机了吗」——
    # 没有时间副词），这一轮连工具都不会挂上，模型只能凭记忆瞎答。
    return bool(_SEARCH_HINT.search(text) or _SEARCH_TOPIC.search(text))


def wants_search_now(cfg, messages: list[dict]) -> bool:
    """这句话该不该「先查了再答」——快车道的闸门，比 wants_search 严得多。"""
    if not getattr(cfg, "search_enabled", True):
        return False
    return bool(_SEARCH_TOPIC.search(user_query(messages)))


def _clean(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text[:limit]


_TAIL_INSTRUCTION = (
    "请用中文、用一两句口语把答案说出来。不要念网址，不要罗列条目，"
    "不要说「搜索结果显示」「根据搜到的消息」这类话，也别再追问要不要接着查——"
    "直接当成你知道的事说。条目开头括号里是发布日期：用户问哪天就说哪天的，"
    "没有那天的就明说，别把旧消息当新的说。查不到就一句「这个我没查到」。"
)

# Tavily 默认 topic=general 不看日期：9/23 实测「昨天有什么 AI 新闻」念成了去年 10 月的
# 旧闻，「比特币现在多少钱」报了个过时的价（真价 8.6 万，它说 8.1 万）。按查询词分三类：
#   价格 / 行情      → general + 近一天（报价页天天更新，正文里就有数）
#   新闻 / 最近 / 昨天 → news + 近一周（news 结果带发布日期，模型自己挑那天的）
#   今天 / 刚刚      → news + 近一天
_PRICE_Q = re.compile(r"价格|多少钱|股价|币价|金价|油价|汇率|行情|报价|市值|涨跌|大盘|股市|A股|美股|港股|比分")
_TODAY_Q = re.compile(r"今天|今日|今早|今晚|今夜|刚刚|刚才|此刻")
_RECENT_Q = re.compile(
    r"新闻|消息|资讯|热点|头条|最新|最近|近期|这几天|这两天|本周|这周|昨天|昨日|昨晚|前天|"
    r"发布|官宣|动态|进展|发生了什么|出了什么|\d{1,2}月\d{1,2}[日号]")
_MONTH_Q = re.compile(r"这个月|本月|上个月|上月")


def _time_scope(query: str) -> tuple[str, str | None]:
    """查询词 → (Tavily 的 topic, time_range)。不是时效类的就 ("general", None)。"""
    if _PRICE_Q.search(query):
        return "general", "day"
    if _MONTH_Q.search(query):
        return "news", "month"
    if _RECENT_Q.search(query):
        # 「今天的新闻」只要近一天；「昨天的」得放一周——近一天只盖得住昨天的后半天
        if _TODAY_Q.search(query) and not re.search(r"昨|前天", query):
            return "news", "day"
        return "news", "week"
    if _TODAY_Q.search(query):
        return "news", "day"
    return "general", None


def _pub_day(s: str) -> str:
    """news 结果的 published_date（「Tue, 22 Sep 2026 04:00:00 GMT」）→ 北京时间的日期。"""
    with contextlib.suppress(Exception):
        d = email.utils.parsedate_to_datetime(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return d.astimezone(_BJ).date().isoformat()
    return ""


def _render(cfg, query: str, items: list[dict], answer: str = "") -> str:
    """把若干条 {title, body, site} 拼成给模型读的纯文本。"""
    if not items and not answer:
        return f"没有搜到「{query}」的结果。请告诉用户没查到，不要编造。"
    today = time.strftime("%Y年%m月%d日")
    lines = [f"今天是{today}。以下是「{query}」的搜索结果："]
    if answer:
        # Tavily 的 answer 常常是英文，模型照着用中文复述即可。
        lines.append(f"要点：{_clean(answer, 400)}")
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. {it['title']}｜{it['body']}（来源 {it['site']}）")
    lines.append(_TAIL_INSTRUCTION)
    return "\n".join(lines)


def _kid(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _next_month_utc() -> float:
    d = _dt.datetime.now(_dt.timezone.utc)
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return _dt.datetime(y, m, 1, tzinfo=_dt.timezone.utc).timestamp()


class _KeyRing:
    """几个 Tavily 账号按顺序用：第 1 个这个月用完就从第 2 个起，第 2 个用完从第 3 个起，每月自动回到第 1 个。

    「用完」= Tavily 回 432 / 433，或者 /usage 接口（不扣额度）显示已用 ≥ 额度。记到下个月 1 号（UTC），
    存在 tmp/tavily_keys.json（只存 key 哈希的前 12 位，不存 key），重启不丢——以前只记在内存里，
    每次重启后的第一次搜索都要先撞一下用完的 key，白等 1 秒。Tavily 没写额度按哪天刷新，所以
    tavily_probe_loop 每 6 小时用 /usage 复查一次，刷新了就提前放回来。
    429（太密）歇 60 秒，401 / 403（key 失效）歇一天。日志里只写「第几个 key」，key 本身绝不落日志。
    """

    MONTHLY = (432, 433)
    REST = {429: 60, 401: 24 * 3600, 403: 24 * 3600}

    def __init__(self, path) -> None:
        self.path = path
        self._until: dict[str, float] = {}
        self._why: dict[str, int] = {}
        with contextlib.suppress(Exception):
            now = time.time()
            for kid, v in json.loads(self.path.read_text(encoding="utf-8")).items():
                if float(v["until"]) > now:
                    self._until[kid], self._why[kid] = float(v["until"]), int(v.get("why", 0))

    def _save(self) -> None:
        with contextlib.suppress(Exception):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: {"until": u, "why": self._why.get(k, 0)}
                                       for k, u in self._until.items()}), encoding="utf-8")
            tmp.replace(self.path)

    def order(self, keys: list[str]) -> list[str]:
        now = time.time()
        return [k for k in keys if self._until.get(_kid(k), 0) <= now]

    def rest(self, key: str, status: int) -> None:
        kid = _kid(key)
        self._until[kid] = (_next_month_utc() if status in self.MONTHLY
                            else time.time() + self.REST.get(status, 60))
        self._why[kid] = status
        self._save()

    def exhausted(self, key: str) -> bool:
        kid = _kid(key)
        return self._why.get(kid) in self.MONTHLY and self._until.get(kid, 0) > time.time()

    def clear(self, key: str) -> None:
        kid = _kid(key)
        if self._until.pop(kid, None) is not None:
            self._why.pop(kid, None)
            self._save()


_TAVILY_KEYS = _KeyRing(SERVER_DIR / "tmp" / "tavily_keys.json")


def _tavily_keys(cfg) -> list[str]:
    return [k for k in (getattr(cfg, "tavily_api_keys", None) or [cfg.tavily_api_key]) if k]


def _usage_numbers(d: dict) -> tuple[float, float]:
    """/usage 的回包 → (已用, 额度)。key 自己有上限就看 key 的，否则看账户套餐的。"""
    k, a = d.get("key") or {}, d.get("account") or {}
    for used, limit in ((k.get("usage"), k.get("limit")), (a.get("plan_usage"), a.get("plan_limit"))):
        if isinstance(limit, (int, float)) and limit > 0 and isinstance(used, (int, float)):
            return float(used), float(limit)
    return 0.0, 0.0


async def tavily_probe(cfg) -> None:
    """用 /usage（不扣额度）看每个 key 这个月还剩没剩：用完的跳到月底，额度刷新了的放回来。"""
    keys = _tavily_keys(cfg)
    url = cfg.tavily_base_url.rstrip("/") + "/usage"
    sess = await _http()
    for i, key in enumerate(keys, 1):
        try:
            async with sess.get(url, headers={"Authorization": f"Bearer {key}"},
                                timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    continue
                used, limit = _usage_numbers(await r.json(content_type=None))
        except Exception:  # noqa: BLE001 - 查不到就下次再查
            continue
        if not limit:
            continue
        if used >= limit and not _TAVILY_KEYS.exhausted(key):
            _TAVILY_KEYS.rest(key, 432)
            log.info("Tavily 第 %d/%d 个 key 本月已用 %d / %d，这个月跳过它", i, len(keys), used, limit)
        elif used < limit and _TAVILY_KEYS.exhausted(key):
            _TAVILY_KEYS.clear(key)
            log.info("Tavily 第 %d/%d 个 key 额度刷新了（已用 %d / %d），又从它用起", i, len(keys), used, limit)


async def tavily_probe_loop(cfg, every_s: float = 6 * 3600) -> None:
    if cfg.search_provider != "tavily" or not _tavily_keys(cfg):
        return
    while True:
        with contextlib.suppress(Exception):
            await tavily_probe(cfg)
        await asyncio.sleep(every_s)


async def _tavily_post(cfg, payload: dict) -> tuple[dict | None, str]:
    """POST 一次 /search，key 轮着试。返回 (结果, 失败时给模型的话)；key 绝不进日志。"""
    url = cfg.tavily_base_url.rstrip("/") + "/search"
    keys = _tavily_keys(cfg)
    for key in _TAVILY_KEYS.order(keys):
        label = f"第 {keys.index(key) + 1}/{len(keys)} 个 key"
        try:
            timeout = aiohttp.ClientTimeout(total=cfg.search_timeout_s)
            sess = await _http()
            async with sess.post(url, json=payload, timeout=timeout,
                                 headers={"Authorization": f"Bearer {key}"}) as resp:
                if resp.status in _KeyRing.MONTHLY:
                    _TAVILY_KEYS.rest(key, resp.status)
                    log.warning("Tavily %s 本月额度用完（HTTP %s），这个月起用下一个", label, resp.status)
                    continue
                if resp.status in _KeyRing.REST:
                    # 限流 / 被拒：这个 key 歇着，马上换下一个
                    _TAVILY_KEYS.rest(key, resp.status)
                    log.warning("Tavily %s 不能用（HTTP %s），换下一个", label, resp.status)
                    continue
                if resp.status != 200:
                    log.warning("Tavily 返回 %s", resp.status)
                    return None, f"搜索失败：搜索服务返回 {resp.status}。请告诉用户你查不到，不要编造。"
                return await resp.json(content_type=None), ""
        except asyncio.TimeoutError:
            log.warning("Tavily 超时（%.1fs）", cfg.search_timeout_s)
            return None, "搜索超时。请告诉用户网络慢没查到，不要编造。"
        except Exception as e:  # noqa: BLE001 - 工具失败绝不能拖垮一轮对话
            log.warning("Tavily 出错：%s", type(e).__name__)
            return None, "搜索失败。请告诉用户你查不到，不要编造。"
    log.error("Tavily %d 个 key 全都额度满或被拒，敲 tavily-key 换", len(keys))
    return None, "搜索额度用完了。请告诉用户你查不到，不要编造。"


async def tavily_search(cfg, query: str) -> str:
    """查 Tavily。失败也返回文本，不抛异常。"""
    topic, rng = _time_scope(query)
    payload = {
        "query": query,
        "max_results": cfg.search_max_results,
        "search_depth": cfg.tavily_search_depth,
        "include_answer": cfg.tavily_include_answer,
        "topic": topic,
    }
    if rng:
        payload["time_range"] = rng
    started = time.monotonic()
    data, err = await _tavily_post(cfg, payload)
    if data is not None and topic == "news" and not data.get("results"):
        # news 库以主流媒体为主，冷门话题可能一条没有：退回 general，时间范围不变
        payload["topic"] = "general"
        data, err = await _tavily_post(cfg, payload)
    if data is None:
        return err

    raw = (data.get("results") or [])[: cfg.search_max_results]
    answer = data.get("answer") or ""
    items = []
    for r in raw:
        day = _pub_day(r.get("published_date") or "")
        url = r.get("url", "") or ""
        items.append({
            "title": _clean(r.get("title", ""), 80),
            "body": (f"（{day}）" if day else "") + _clean(r.get("content", ""), cfg.search_snippet_chars),
            "site": _clean(url.split("/")[2] if "//" in url else "", 40),
        })
    log.info("Tavily「%s」[%s%s] → %d 条%s，%.2fs", query, payload["topic"],
             f"/{rng}" if rng else "", len(items), "（含要点）" if answer else "",
             time.monotonic() - started)
    return _render(cfg, query, items, answer)


async def searxng_search(cfg, query: str) -> str:
    """查 SearXNG，返回一段给模型读的纯文本。失败也返回文本，不抛异常。"""
    url = cfg.searxng_url.rstrip("/") + "/search"
    params = {
        "q": query,
        "format": "json",
        "language": cfg.search_language,
        "safesearch": "1",
    }
    if getattr(cfg, "searxng_engines", ""):
        params["engines"] = cfg.searxng_engines
    started = time.monotonic()
    try:
        timeout = aiohttp.ClientTimeout(total=cfg.search_timeout_s)
        sess = await _http()
        async with sess.get(url, params=params, timeout=timeout) as resp:
            if resp.status != 200:
                log.warning("SearXNG 返回 %s", resp.status)
                return f"搜索失败：搜索服务返回 {resp.status}。请告诉用户你查不到，不要编造。"
            data = await resp.json(content_type=None)
    except asyncio.TimeoutError:
        log.warning("SearXNG 超时（%.1fs）", cfg.search_timeout_s)
        return "搜索超时。请告诉用户网络慢没查到，不要编造。"
    except Exception as e:  # noqa: BLE001
        log.warning("SearXNG 出错：%s", e)
        return "搜索失败。请告诉用户你查不到，不要编造。"

    # 搜狗微信会翻出 2018 年的「比特币今日价格」，模型会当成今天的念。带日期的只留一年内的，
    # 日期写进正文，让模型自己分新旧
    cutoff = _dt.date.today() - _dt.timedelta(days=365)
    kept = []
    for r in data.get("results") or []:
        day = None
        with contextlib.suppress(Exception):
            day = _dt.date.fromisoformat((r.get("publishedDate") or "")[:10])
        if day is not None and day < cutoff:
            continue
        kept.append((day, r))
        if len(kept) >= cfg.search_max_results + 1:
            break
    log.info("SearXNG「%s」→ %d 条，%.2fs", query, len(kept), time.monotonic() - started)
    items = [{
        "title": _clean(r.get("title", ""), 80),
        "body": (f"（{day}）" if day else "") + _clean(r.get("content", ""), cfg.search_snippet_chars),
        "site": _clean(r.get("engine", ""), 20),
    } for day, r in kept]
    return _render(cfg, query, items)


# --- 天气：Open-Meteo（免费、免 key，9/23 实测 EX10 出去 0.8 s） -------------------
# 搜索引擎给的天气是一堆「XX 天气预报_2345」的页面标题，没有数；直接查预报接口准得多。

_WMO = {
    0: "晴", 1: "晴间多云", 2: "多云", 3: "阴", 45: "有雾", 48: "有雾凇",
    51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨", 56: "冻毛毛雨", 57: "冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒", 80: "阵雨", 81: "阵雨", 82: "强阵雨",
    85: "阵雪", 86: "强阵雪", 95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨伴冰雹",
}


async def weather_lookup(cfg, city: str) -> str:
    """城市名 → 今天起三天的预报，给模型读的纯文本。失败返回空串（调用方改走搜索）。"""
    city = re.sub(r"(市|的)?(天气|气温|温度).*$", "", (city or "").strip()) or getattr(cfg, "home_city", "")
    if not city:
        return ""
    started = time.monotonic()
    try:
        sess = await _http()
        timeout = aiohttp.ClientTimeout(total=cfg.search_timeout_s)
        async with sess.get("https://geocoding-api.open-meteo.com/v1/search",
                            params={"name": city, "count": 1, "language": "zh"},
                            timeout=timeout) as r:
            geo = ((await r.json(content_type=None)).get("results") or [None])[0]
        if not geo:
            log.info("天气：找不到城市「%s」", city)
            return ""
        async with sess.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": geo["latitude"], "longitude": geo["longitude"],
            "current": "temperature_2m,apparent_temperature,weather_code,relative_humidity_2m,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "Asia/Shanghai", "forecast_days": 3,
        }, timeout=timeout) as r:
            d = await r.json(content_type=None)
    except Exception as e:  # noqa: BLE001
        log.warning("天气查询失败（%s）：%s", city, type(e).__name__)
        return ""
    cur, day = d.get("current") or {}, d.get("daily") or {}
    lines = [f"{geo.get('name', city)}（{geo.get('admin1', '')}）的天气，数据来自气象预报接口："]
    if cur:
        lines.append(
            f"现在 {_WMO.get(cur.get('weather_code'), '')}，{cur.get('temperature_2m')} 度，"
            f"体感 {cur.get('apparent_temperature')} 度，湿度 {cur.get('relative_humidity_2m')}%，"
            f"风速每小时 {cur.get('wind_speed_10m')} 公里。")
    for i, name in enumerate(("今天", "明天", "后天")):
        try:
            lines.append(
                f"{name}（{day['time'][i]}）{_WMO.get(day['weather_code'][i], '')}，"
                f"{day['temperature_2m_min'][i]} 到 {day['temperature_2m_max'][i]} 度，"
                f"降水概率 {day['precipitation_probability_max'][i]}%。")
        except (KeyError, IndexError, TypeError):
            break
    lines.append("请挑用户问的那天，用一两句口语说：天气、温度范围，要下雨就提醒带伞。温度说整数。")
    log.info("天气「%s」→ %s，%.2fs", city, geo.get("name"), time.monotonic() - started)
    return "\n".join(lines)


# --- 热榜：「今天有什么新闻」这种泛问 -----------------------------------------
# Tavily 对泛问只会随便捞三条（9/24：「今朝最新的新闻」回的是文化类消息，当天头条是中美元首
# 会晤），热榜才是「今天大家在看什么」。NewsNow（github.com/ourongxing/newsnow，小智社区的
# xiaozhi-esp32-server 也用它）把澎湃 / 财联社 / 微博热搜聚成一个 JSON 接口，EX10 直连 1 s 左右。
_NEWS_Q = re.compile(r"新闻|头条|热点|热搜|大事|要闻|发生了什么|出了什么事")
_NEWS_GENERIC = re.compile(
    r"\d{2,4}年|\d{1,2}月|\d{1,2}[日号]|[0-9０-９]+|今天|今日|今早|今晚|昨天|最近|近期|这几天|这两天|"
    r"最新|最热|实时|国内外|国内|国际|全球|世界|中国|有什么|有哪些|什么|哪些|几条|一些|重要|大|"
    r"新闻|头条|热点|热搜|大事|要闻|发生了|出了|事|给我|讲讲|说说|播报|听听|一下|看|是|有|讲|说|"
    r"的|了|吗|呢|吧|啊|呀")
_NEWS_NAMES = {"thepaper": "澎湃新闻", "cls-hot": "财联社", "weibo": "微博热搜", "toutiao": "今日头条",
               "zhihu": "知乎热榜", "baidu": "百度热搜", "ithome": "IT之家", "36kr": "36氪"}


def _generic_news(query: str) -> bool:
    return bool(_NEWS_Q.search(query)) and len(_NEWS_GENERIC.sub("", _norm_query(query))) <= 1


async def hot_news(cfg) -> str:
    """几个热榜的头几条标题，给模型挑着说。全都拿不到返回空串（调用方改走搜索）。"""
    base = cfg.newsnow_url.rstrip("/")
    started = time.monotonic()
    sess = await _http()

    async def one(sid: str) -> list[str]:
        try:
            async with sess.get(f"{base}/api/s", params={"id": sid},
                                headers={"User-Agent": "Mozilla/5.0"},
                                timeout=aiohttp.ClientTimeout(total=4)) as r:
                if r.status != 200:
                    return []
                d = await r.json(content_type=None)
            return [_clean(i.get("title", ""), 60) for i in (d.get("items") or [])[:6] if i.get("title")]
        except Exception:  # noqa: BLE001 - 一个源挂了不影响别的
            return []

    sids = [x.strip() for x in cfg.newsnow_sources.split(",") if x.strip()]
    got = await asyncio.gather(*(one(x) for x in sids))
    lines = [f"{_NEWS_NAMES.get(sid, sid)}：" + "；".join(ts) for sid, ts in zip(sids, got) if ts]
    log.info("热榜 %s → %d 个源有数，%.2fs", ",".join(sids), len(lines), time.monotonic() - started)
    if not lines:
        return ""
    today = time.strftime("%Y年%m月%d日")
    return "\n".join([f"今天是{today}。以下是刚抓到的几个热榜标题（每个源按热度排）：", *lines,
                      "请挑两三条最重要的大事，用一两句口语说给用户听，先说最大的那件。"
                      "不要念来源名字，不要逐条罗列，不要说「根据热榜」。"])


def _usable(out: str) -> bool:
    # 失败文本都以「搜索」开头（鉴权 / 额度 / 超时 / 出错），没结果是「没有搜到」
    return bool(out) and not out.startswith(("搜索", "没有搜到"))


async def _tavily_hedged(cfg, query: str) -> str:
    """先问 Tavily；XZ_SEARCH_HEDGE_S 秒还没回就同时问 SearXNG，谁先拿到能用的结果用谁。"""
    tav = asyncio.create_task(tavily_search(cfg, query))
    sx = None
    try:
        hedge = float(getattr(cfg, "search_hedge_s", 0) or 0)
        if cfg.searxng_url and hedge > 0:
            done, _ = await asyncio.wait({tav}, timeout=hedge)
            if not done:
                log.info("Tavily %.1f 秒还没回，同时问 SearXNG", hedge)
                sx = asyncio.create_task(searxng_search(cfg, query))
                out = ""
                for fut in asyncio.as_completed((tav, sx)):
                    out = await fut
                    if _usable(out):
                        return out
                return out
        out = await tav
        # 9/23 Tavily 额度用光（432）整整一天没人发现——所有 key 都不行时有 SearXNG 就自动退过去
        if _usable(out) or not cfg.searxng_url:
            return out
        log.warning("Tavily 没查成，这次改用 SearXNG")
        return await searxng_search(cfg, query)
    finally:
        for t in (tav, sx):
            if t is not None and not t.done():
                t.cancel()


async def web_search(cfg, query: str) -> str:
    query = normalize_query(query)
    if getattr(cfg, "news_hotlist", True) and cfg.newsnow_url and _generic_news(query):
        out = await hot_news(cfg)
        if out:
            return out
    if cfg.search_provider == "tavily":
        if getattr(cfg, "tavily_api_keys", None) or cfg.tavily_api_key:
            return await _tavily_hedged(cfg, query)
        log.error("XZ_SEARCH_PROVIDER=tavily 但没有 TAVILY_API_KEY，改用 SearXNG")
    return await searxng_search(cfg, query)


async def run_tool(cfg, name: str, args: dict) -> str:
    if name == "web_search":
        query = (args or {}).get("query", "").strip()
        if not query:
            return "搜索词是空的，没法查。"
        return await web_search(cfg, query)
    log.warning("未知工具：%s", name)
    return f"没有叫 {name} 的工具。"


# --- 抢跑：模型还在想「要不要查」的时候，先拿用户原话去查 ------------------
#
# 一次搜索轮的时间是这么花的：第 1 轮 LLM 挂着工具、决定要调 web_search，
# 3~4 秒；然后才轮到搜索，2 秒；最后第 2 轮读结果作答，1~4 秒。前 3~4 秒
# 网络完全闲着。既然 wants_search() 在调 LLM 之前就已经判定「这句要查」，
# 那就同时把搜索发出去——等模型真的吐出 tool_call，结果往往已经在手上，
# 整整两秒就藏进了那 3~4 秒里。
#
# 模型改写的查询词和用户原话差太远时（比如原话「北京冷不冷」改写成
# 「北京 2026年9月7日 气温」），就丢掉重查，代价只是多一次 API 调用。

_TAIL_PARTICLES = re.compile(r"[的了吗呢吧啊呀嘛]+$|怎么样$|如何$|是多少$|多少$")
# 模型改写时爱加的「时效 / 泛称」词：不改变查的是什么，比对前从模型那边去掉
# （9/23：「比特币现在多少钱」被改写成「比特币最新价格」，八成规则判没命中，白查一次 2.3 s）
_NEUTRAL_Q = re.compile(
    r"\d{2,4}年|\d{1,2}月|\d{1,2}[日号]|[0-9０-９]+|"
    r"最新|最近|近期|实时|今天|今日|昨天|昨日|前天|现在|目前|当前|"
    r"价格|报价|行情|走势|新闻|消息|资讯|动态|进展|情况|多少钱|多少|有哪些|有什么")
# 天气走 Open-Meteo，拿原话抢跑网页搜索纯属白花 Tavily 额度
_WEATHER_Q = re.compile(r"天气|气温|温度|下雨|降雨|下雪|冷不冷|热不热|刮风|带伞|打伞")


_WEATHER_ASK = re.compile(r"[?？吗]|怎么样|如何|会不会|要不要|几度|多少度|冷不冷|热不热|下不下|有没有雨|带伞|穿什么")
_WEATHER_STRIP = re.compile(
    r"今天|明天|后天|大后天|今早|今晚|今夜|早上|上午|中午|下午|晚上|夜里|这两天|这几天|周末|这周|下周|"
    r"星期[一二三四五六日天几]|周[一二三四五六日]|"
    r"天气|气温|温度|预报|下雨|降雨|下雪|刮风|冷不冷|热不热|冷|热|带伞|穿什么|"
    r"怎么样|如何|会不会|要不要|几度|多少度|有没有雨|下不下|好吗|好不好|"
    r"我们这|我这里|我这边|这里|这边|外面|的|会|吗|呢|啊|呀|吧|了|是|现在|目前|好")


def weather_question(messages: list[dict]) -> str | None:
    """这句是在问天气吗：是就返回城市（没说城市返回空串，查的时候用 home_city），不是返回 None。

    9/24 真机：「今朝上海天气好吗？」模型没说【天气：上海】直接开口，「多云，廿三到三十度」
    是它编的。窄闸门命中天气又像个问句，就在后台先查着，模型不说暗号也拿这份数重答。
    """
    q = user_query(messages)
    bare = _norm_query(q)
    # 问句，或者干脆就是「苏州天气」这种
    if not (_WEATHER_Q.search(q) and (_WEATHER_ASK.search(q)
                                      or re.search(r"(天气|气温|温度|预报)$", bare))):
        return None
    city = _WEATHER_STRIP.sub("", bare)
    rep = re.fullmatch(r"(.+?)\1+", city)   # 「上海天气怎么样？上海天气」→ 上海上海
    if rep:
        city = rep.group(1)
    # 剩下的像个地名才用（两到六个字、没有动词），不然按主人所在城市
    if 2 <= len(city) <= 6 and not re.search(r"[要去带穿出门做吃玩]", city):
        return city
    return ""


def _norm_query(q: str) -> str:
    """去掉标点空白和句尾语气词，只留下用来比对的实词。"""
    q = re.sub(r"[\s,，。！!?？、:：;；\"'“”‘’()（）\[\]【】]+", "", (q or "").strip())
    return _TAIL_PARTICLES.sub("", q)


class SearchPrefetch:
    """一次抢跑的搜索。命中就白赚，没命中就取消，不影响正确性。"""

    def __init__(self, cfg, query: str):
        self.query = query
        self._task = asyncio.create_task(web_search(cfg, query))

    def matches(self, query: str) -> bool:
        """抢跑结果能不能拿来当这个查询词的答案。

        宽一点没关系：抢跑查的是用户的原话，Tavily 本来就吃自然语言，
        通常比模型改写的关键词还全。要防的只是「跑题」——多轮对话里
        用户说「那东京呢」，模型查「东京天气」，抢跑那句是废话。
        判据：模型查询词里的字，八成以上在用户原话里出现过。
        """
        a, b = _norm_query(normalize_query(self.query)), _norm_query(normalize_query(query))
        if not a or not b:
            return False
        if a in b or b in a:
            return True
        core = set(_NEUTRAL_Q.sub("", b))
        if not core:
            return True   # 模型的词全是「最新新闻」这类，没有比原话更具体的东西
        hit = sum(1 for c in core if c in a)
        return hit / len(core) >= 0.8

    async def take(self, query: str) -> str | None:
        """查询词对得上就返回抢跑结果（可能还要等它跑完），否则返回 None。"""
        if not self.matches(query):
            self.cancel()
            return None
        try:
            return await self._task
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - 抢跑失败就当没抢过
            log.warning("抢跑搜索出错，改走正常调用：%s", type(e).__name__)
            return None

    def cancel(self) -> None:
        if not self._task.done():
            self._task.cancel()


PREFETCH_NOTE = (
    "\n（以上是系统按你刚才那句话自动查的。如果跟问题没关系，就当没看见，"
    "正常聊天回答，别提搜索这回事。）"
)


def start_prefetch(cfg, messages: list[dict]) -> SearchPrefetch | None:
    """闸门已经放行时调用：用最后一句用户原话开跑。"""
    if not getattr(cfg, "search_prefetch", True):
        return None
    query = user_query(messages)
    if not query or len(query) > 120 or _WEATHER_Q.search(query):
        return None
    log.info("抢跑搜索「%s」", query)
    return SearchPrefetch(cfg, query)
