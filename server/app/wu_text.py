"""普通话写法 → 上海话写法（轻量词表替换），给豆包「用上海话讲」再加一道保险。

豆包小何 2.0 的上海话只靠语音指令，喂书面普通话时常常直接念成普通话（9/25 探针：
线上指令下，播报 20 段里 10 段念成普通话）；原文里有侬、伐、啥这类上海话字，它就稳得多。
这里只换一小撮常用、豆包念得准的词；搿、呒没、㑚这类生僻写法没验过，不加。
词形参考吴语学堂 / rime-wugniu 上海话词典、沪江上海话对照表。
"""
from __future__ import annotations

import re

# 原样保护的词写成「X → X」：比它短的规则（你、他、不要）不能拆开它
_WORDS = [
    ("你们", "你们"), ("其他", "其他"), ("要不要", "要勿要"), ("用不用", "用勿用"),
    ("什么时候", "啥辰光"), ("为什么", "为啥"), ("什么", "啥"),
    ("怎么样", "哪能"), ("怎么", "哪能"),
    ("我们", "阿拉"), ("咱们", "阿拉"), ("他们", "伊拉"), ("她们", "伊拉"),
    ("您", "侬"), ("你", "侬"), ("他", "伊"), ("她", "伊"),
    ("今天", "今朝"), ("明天", "明朝"), ("昨天", "昨日"),
    ("不要", "覅"), ("不用", "覅"),
    ("不知道", "勿晓得"), ("知道", "晓得"),
    ("不是", "勿是"), ("不会", "勿会"), ("不能", "勿能"), ("不太", "勿大"), ("不好", "勿好"),
    ("是否", "是勿是"), ("哪里", "啥地方"), ("睡觉", "困觉"), ("下雨", "落雨"),
    ("很", "蛮"),
    ("再说", "再讲"), ("说话", "讲话"), ("说一", "讲一"), ("怎么说", "哪能讲"),
]
_WORDS.sort(key=lambda kv: -len(kv[0]))
_MAP = dict(_WORDS)
_WORD_RE = re.compile("|".join(re.escape(k) for k, _ in _WORDS))

# 「X 的 Y」→「X 个 Y」；这些词里的「的」不换
_DE_KEEP = ("目的", "的确", "的士", "有的", "的话", "端的", "似的")
_DE_RE = re.compile(r"(?<=[一-鿿])的(?=[一-鿿])")
_MA_RE = re.compile(r"吗(?=[？?。！!]|$)")


def _de(text: str) -> str:
    out = []
    for m in _DE_RE.finditer(text):
        i = m.start()
        if any(text[max(0, i - len(w) + 1):i + len(w)].find(w) >= 0 for w in _DE_KEEP):
            continue
        out.append(i)
    if not out:
        return text
    chars = list(text)
    for i in out:
        chars[i] = "个"
    return "".join(chars)


# 读音纠正：只改送去合成的字，日志和上下文照旧用原文。豆包 2.0 的音色不认 SSML，只能换同音字。
# 9/25 试过「目录」→「莫录」（莫、目在上海话里都念 moq）：用户听了说原来的「目录」念上海话时就对，
# 念错的那遍其实是整句串成了普通话（mu lu），换字治不了，已撤掉。先确认是读音问题、不是串普通话，再往这里加。
# 9/27 用户：「新会话发音不对，应该是新为屋」「提交上海话叫第高」（吴语模型，不是串普通话），照他给的字换。
# 「棒棒硬」上海话正写「邦邦硬」，邦邦都念第一声（用户听样音挑的）。
READINGS: dict[str, str] = {"会话": "为屋", "提交": "第高", "棒棒硬": "邦邦硬"}
_READ_RE = re.compile("|".join(re.escape(k) for k in sorted(READINGS, key=len, reverse=True)))

# 9/25 用户听三版「四路」（写「四」、写「4」、换同音的「世」），挑了写「4」的：豆包念「4」更像上海话的 sy。
# 只换当数目用的「四」，后面得跟量词（四路、四桩、第四条）；「四川」「四周」「不三不四」不动，
# 前面挨着别的数字（十四、三四个）也不动，免得把一个数拆成两半。
_NUM = "零一二两三四五六七八九十百千万亿"
_SI_RE = re.compile(rf"(?<![{_NUM}])四(?=[路条项桩个件点次天年月号遍位只张块分秒岁段步种样台页层趟回家部本句批轮组])")


# 9/27 用户：「换好了」上海话讲「调好了」（调念 dio），上海闲话口语里单用的「换」都讲「调」：调一只、调声音、调衣裳、调位子。
# 只换后面跟这些字的单用「换」；切换、交换、替换、兑换、转换、更换、互换、置换这类书面词不动。
_HUAN_RE = re.compile(r"(?<![切交替兑转更互置])换(?=[好掉过成只个一下声衣位])")


# 9/27 用户：「侬在」个「在」上海话讲「了该」。钱乃荣《「辣辣」的写法和用法》正字写「辣辣 / 辣海」（辣念 lak，不写勒、了），
# 新派年轻人多讲「辣该」，照用户的讲法写「辣该」。只换稳妥的几种：「正在」一律换；人称（我侬你伊他她、阿拉伊拉、X们）后面
# 紧跟或隔一个就 / 勿 / 不的「在」；还在、也在、都在、一直在。在乎、在意、在于、在线、在场、在世、在即、在内、在座、在行不动；
# 现在、实在、存在前面不是人称，本来就碰不到。
_ZAI_RE = re.compile(
    r"(?<!真)正在"
    r"|(?:(?<=[我侬你伊他她拉们])(?:就|勿|不)?|(?<=[还也都])|(?<=一直))在(?![乎意于线场世即内座行])"
)


def _zai(m: re.Match) -> str:
    s = m.group(0)
    return "辣该" if s == "正在" else s[:-1] + "辣该"


def fix_readings(text: str) -> str:
    if READINGS:
        text = _READ_RE.sub(lambda m: READINGS[m.group(0)], text)
    text = _HUAN_RE.sub("调", text)
    text = _ZAI_RE.sub(_zai, text)
    return _SI_RE.sub("4", text)


def to_wu(text: str) -> str:
    """普通话常用词换成上海话写法；本来就是上海话的句子基本不动。"""
    text = _WORD_RE.sub(lambda m: _MAP[m.group(0)], text)
    text = _MA_RE.sub("伐", text)
    return _de(text)


# ---------- 上海话数字（只给本地 CosyVoice 用；豆包那条路照旧走 fix_readings）----------
# 9/26 用户：「上海话有两百乃尼代表 222」。规矩：
# - 十位上的 2 念「廿」：28 → 廿八，120 → 一百廿，222 → 两百廿二，第 20 → 第廿
# - 百 / 千 / 万 / 亿 前面的 2 念「两」：200 → 两百，2000 → 两千，2 万 → 两万
# - 末位的 2 念「二」：12 → 十二，32 → 三十二；单独的 2 后面跟量词念「两」（两个、两点钟），否则「二」（二号、二楼）
# - 7:10 → 七点十分；2.5 → 两点五；20% → 百分之廿；2026 年逐位念；版本号 2.2.4 逐段念
_D = "零一二三四五六七八九"
_TWO_MEASURE = re.compile(
    r"(?:个|条|块|分钟|分|秒|天|次|遍|位|只|张|台|件|项|岁|趟|页|本|部|家|层|声|句|段|步|种|样|批|轮|组|路|点|钟头|小时|"
    r"礼拜|号头|公里|斤|瓶|杯|碗|份|套|双|对|辆|首|口|间|笔|封|粒|根|支|把|桩|趟|记|歇)"
)


def _wu_4(n: int) -> str:
    th, r = divmod(n, 1000)
    h, r = divmod(r, 100)
    t, o = divmod(r, 10)
    s = ""
    if th:
        s += ("两" if th == 2 else _D[th]) + "千"
    if h:
        s += ("两" if h == 2 else _D[h]) + "百"
    elif th and (t or o):
        s += "零"
    if t == 1:
        s += "一十" if s else "十"
    elif t == 2:
        s += "廿"
    elif t:
        s += _D[t] + "十"
    elif (th or h) and o and not s.endswith("零"):
        s += "零"
    if o:
        s += _D[o]
    return s or "零"


def _wu_int(n: int) -> str:
    """整数念法（上海话）。单独的 2 返回「二」，要不要改「两」看后面跟什么，由调用方定。"""
    for unit, size in (("亿", 10 ** 8), ("万", 10 ** 4)):
        if n >= size:
            hi, lo = divmod(n, size)
            head = _wu_int(hi)
            s = ("两" if head == "二" else head) + unit
            if lo:
                s += ("零" if lo < size // 10 else "") + _wu_int(lo)
            return s
    return _wu_4(n)


def _digits(s: str) -> str:
    return "".join(_D[int(c)] for c in s)


def _num_sub(m: re.Match) -> str:
    raw = m.group(0)
    if len(raw) >= 7 or (len(raw) > 1 and raw.startswith("0")):  # 电话、编号：逐位念
        return _digits(raw)
    s = _wu_int(int(raw))
    if s == "二" and _TWO_MEASURE.match(m.string, m.end()):
        return "两"
    return s


def wu_numbers(text: str) -> str:
    """把句子里的数字改成上海话的说法，再交给 CosyVoice（它自己的规整会念成普通话的「二十八」）。"""
    def clock(m):
        h, mi = int(m[1]), int(m[2])
        s = ("两" if h == 2 else _wu_int(h)) + "点"
        if mi:
            s += ("零" if mi < 10 else "") + _wu_int(mi) + "分"
        return s

    text = re.sub(r"(?<![\d.])(\d{1,2})[:：](\d{2})(?![\d:])", clock, text)
    text = re.sub(r"(\d+(?:\.\d+)?)%", lambda m: "百分之" + m[1], text)
    text = re.sub(r"(?<![\d.])\d+(?:\.\d+){2,}(?![\d.])",  # 版本号 2.2.4.14
                  lambda m: "点".join(_wu_int(int(p)) for p in m[0].split(".")), text)
    text = re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])",
                  lambda m: ("两" if m[1] == "2" else _wu_int(int(m[1]))) + "点" + _digits(m[2]), text)
    text = re.sub(r"(?<!\d)(\d{4})(?=\s*年)", lambda m: _digits(m[1]), text)
    text = re.sub(r"\d+", _num_sub, text)
    # 模型 / 改写已经写成汉字的：二十八 → 廿八，二百 → 两百
    text = re.sub(r"(?<![零一二两三四五六七八九十])二十", "廿", text)
    text = re.sub(r"(?<![零一二两三四五六七八九十百千万亿])二(?=[百千万亿])", "两", text)
    return text
