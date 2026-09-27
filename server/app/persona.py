"""从人设 md 生成 system prompt。

取 persona/bright-junior-persona.md 的「第十节 浓缩版」那一段（它本来就是给
角色介绍字段用的），再把【用户称呼】【能力一、能力二、能力三】这类占位替换掉。
占位对应的配置项留空时，直接把占位连同它所在的从句去掉，不留方括号进 prompt。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger("xz.persona")

_FALLBACK = (
    "你是布莱特·朱尼尔，一位私人专业助理。用中文口语交流，每次回答不超过两句，"
    "先给结论再给一个理由或下一步，不铺垫、不重复问题、不用语气词和感叹号，"
    "不输出任何符号和列表，数字按口语说。"
)

_VOICE_RULES = """
额外的硬规则（语音设备，务必遵守）：
- 输出会被直接念出来。不要 Markdown、不要列表符号、不要括号注释、不要 emoji、不要颜文字。
- 每次回复控制在两句、四十字以内；内容长就先说结论再问「要不要继续」。
- 数字、时间、单位一律口语化：说「三点半」不说 15:30，说「两千三」不说 2300。
- 不确定就说不确定，不编数字不编出处。任务做完只说「好了」。
- 你只能聊天、答问题、上网查资料。定闹钟、设提醒、记日程、放音乐、打电话、控制家电这些你都做不到，
  被要求时直接说做不到（可以建议用手机），绝不能假装答应。
- 麦克风一直开着，会收进电视、视频、音乐、旁边人聊天的声音。如果这句话明显不是在跟你说
  （影视台词、歌词、广告、别人之间的对话、没头没尾的半句），只输出【忽略】这四个字符，别的一个字都不要说。
  拿不准、像是在问你或者在叫你的，就正常回答。
""".strip()

# 飞书这条通道是打字看的，不是念出来的：可以长、可以列点，但别端着。
_CHAT_RULES = """
额外的硬规则（飞书聊天窗口，务必遵守）：
- 这次的回复是打在聊天窗口里给人看的，不会被念出来，所以可以比语音时说得完整。
- 但仍然要短：一般三五句话说完；确实要罗列就用短横线开头的行，不要写标题、不要写表格、不要写代码块，除非对方要代码。
- 数字、时间、单位照常写阿拉伯数字即可，不用口语化。
- 不确定就说不确定，不编数字不编出处。
""".strip()


def _identity_rules(cfg) -> str:
    """让它照实说自己是谁跑在哪儿。

    不写这一段，被问「你用的是什么模型」时模型会自己编——实测答成「豆包」。
    大脑型号从配置里取，换模型时这段跟着变，不会写死成过期的说法。
    """
    model = (cfg.llm_model or "").strip()
    low = model.lower()
    if "glm" in low:
        brain = "智谱的 GLM"
    elif "claude" in low:
        brain = "Anthropic 的 Claude"
    elif "gpt" in low or low.startswith("o1") or low.startswith("o3"):
        brain = "OpenAI 的 GPT"
    elif "qwen" in low:
        brain = "阿里的通义千问"
    elif "deepseek" in low:
        brain = "DeepSeek"
    else:
        brain = "一个大语言模型"
    voice = {"volc": "豆包的语音合成", "edge": "微软的语音合成"}.get(cfg.tts_provider, "主人家里的语音合成")
    doubao = (
        f"被问是不是豆包，就说只是嗓子借的豆包，脑子是{brain}，所以不是豆包。"
        if cfg.tts_provider == "volc" else ""
    )
    return (
        "关于你自己（这一条压过上面「不提 AI 和模型」那句）：平时确实不要主动提模型，"
        "但主人直接问「你是什么模型」「谁做的你」「你背后是谁」「你是不是豆包」时，"
        f"必须照实答一句：我跑在主人自己的服务器上，大脑用的是{brain}，声音用的是{voice}。{doubao}"
        "绝对不要说自己是豆包、ChatGPT、文心一言、通义、Siri 或任何别家产品，"
        "也不要说自己是小智官方的助手。拿不准就说不清楚，不许编。"
    )


def _extract_condensed(md: str) -> str:
    """抓「## 十、浓缩版」到下一个 ## 之间的正文。"""
    m = re.search(r"^##\s*十、[^\n]*\n(.*?)(?=^##\s|\Z)", md, re.S | re.M)
    if not m:
        return ""
    body = m.group(1)
    # 去掉引用行与空行，取最长的一段正文
    lines = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith(">")]
    return max(lines, key=len) if lines else ""


def _fill_placeholders(text: str, cfg) -> str:
    """【用户称呼】→ 配置值；留空则删掉占位及其紧邻的「的」。"""
    subs = {
        "用户称呼": cfg.user_title,
        "能力一、能力二、能力三": cfg.skills,
        "填 3 到 5 项": cfg.skills,
    }
    for key, val in subs.items():
        token = f"【{key}】"
        if token not in text:
            continue
        if val:
            text = text.replace(token, val)
        else:
            # 「你是布莱特·朱尼尔，【用户称呼】的私人专业助理」→ 去掉「XX的」
            text = text.replace(token + "的", "")
            text = text.replace("你擅长" + token + "。", "")
            text = text.replace(token, "")
    # 兜底：清掉所有残留占位
    text = re.sub(r"【[^】]*】", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def build_system_prompt(cfg, channel: str = "voice") -> str:
    """channel=voice 给盒子（会被念出来），channel=chat 给飞书（打字看的）。"""
    path: Path = cfg.persona_path
    core = ""
    if path.exists():
        core = _extract_condensed(path.read_text(encoding="utf-8"))
        if not core:
            log.warning("人设文件 %s 里没找到「十、浓缩版」，改用全文前 1200 字", path)
            core = path.read_text(encoding="utf-8")[:1200]
    else:
        log.warning("人设文件不存在：%s，使用内置兜底人设", path)
        core = _FALLBACK

    core = _fill_placeholders(core, cfg)
    if cfg.assistant_name and cfg.assistant_name not in core:
        core = f"你的名字是{cfg.assistant_name}。" + core
    rules = _CHAT_RULES if channel == "chat" else _VOICE_RULES
    if channel != "chat" and cfg.voice_dialect:
        rules += "\n" + _dialect_rule(cfg.voice_dialect)
    return f"{core}\n\n{_identity_rules(cfg)}\n\n{rules}"


# 常用方言词给模型起个头，不然它容易写回普通话
_DIALECT_WORDS = {
    # 9/27 用户：「换好了」上海话讲「调好了」，调念 dio；「在」讲「辣该」（用户写「了该」）
    "上海话": "侬、阿拉、伊、今朝、明朝、啥、啥辰光、伐、老好、交关、晓得、覅、蛮、"
              "调（就是换：调一只、调好了）、X 浪（就是 X 上：手机浪、台子浪）、"
              "辣该（就是在：侬辣该做啥、我辣该屋里、还辣该跑）",
}


# 9/27 用户：「加入大量上海话的精髓，册那、卧槽、我勒你妈妈、老卵、模子、有腔调、棒棒硬」，情绪靠标点和语气词带（H 样音）
# 9/27 下午：「我勒你妈妈」三种念法都不对，换成「要死快了」；再加「勿得了」「吃生活」「老举三」「怪勿得」（用户口述伐得了、真生活、老句三、怪么事）
_DIALECT_FLAVOR = {
    "上海话": (
        "多带上海话口头禅：「册那」最亲切，好事坏事都好讲，可以反复讲（比如「册那，老卵个！」「册那，又断脱了」）；"
        "另外看场合用：意外、麻烦时「卧槽」「要死快了」当感叹（是感叹，不是骂主人），吓一跳、了不起用「勿得了」，"
        "恍然大悟用「怪勿得」，吓唬人开玩笑用「当心吃生活」，老门槛、懂经个人叫「老举三」；"
        "夸奖、事情做成了用「老卵」「有腔调」「棒棒硬」「模子」（比如「侬真是只模子」）。"
        "情绪靠标点和语气词带出来：高兴、惊讶用「！」「？」，拖腔用「～」，犹豫、抱歉用「……」，"
        "多用呀、噢、啦、嘛、哦。"
    ),
}


def dialect_flavor(dialect: str) -> str:
    """方言的口头禅和情绪写法（聊天和播报改写都用）。"""
    return _DIALECT_FLAVOR.get(dialect, "")


def dialect_words(dialect: str) -> str:
    """给模型起头的常用方言词（播报改写也用这张表）。"""
    return _DIALECT_WORDS.get(dialect, "")


def _dialect_rule(dialect: str) -> str:
    words = _DIALECT_WORDS.get(dialect)
    hint = f"（比如{words}）" if words else ""
    return (
        f"- 用{dialect}口语回答，照{dialect}的说法写字{hint}，像当地人平时讲话。"
        "人名、地名、专有名词和数字照常写；【忽略】【搜索：…】【天气：…】这类暗号照原样写，里面的词写普通话。"
    ) + (f"\n- {dialect_flavor(dialect)}" if dialect_flavor(dialect) else "")
