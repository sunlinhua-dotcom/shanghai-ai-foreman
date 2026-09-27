"""集中配置：全部从 server/.env + 环境变量读。

硬规矩：ANTHROPIC_API_KEY 只从环境/.env 读，绝不写进任何被提交的文件。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = SERVER_DIR / ".env"

_ENV_MISSING = f"""\
缺少配置文件 {ENV_FILE}

请这样创建（key 不会进版本库，.env 已在 .gitignore 里）：

    cd {SERVER_DIR}
    cp .env.example .env
    # 用编辑器把 ANTHROPIC_API_KEY=sk-ant-... 填上

至少要有 ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN 其中一行。其余项都有默认值，见 .env.example。
"""

_KEY_MISSING = f"""\
{ENV_FILE} 里既没有可用的 ANTHROPIC_API_KEY，也没有 ANTHROPIC_AUTH_TOKEN。

    打开 {ENV_FILE}
    二选一：
        ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxx        # 官方 / x-api-key 网关
        ANTHROPIC_BASE_URL=https://… ANTHROPIC_AUTH_TOKEN=…   # Bearer 型网关（如智谱 GLM）
    然后 ./run.sh restart

凭据只会留在这个文件里，.env 已在 .gitignore 中，不会被提交。
其余配置项见 .env.example。
"""

_PLACEHOLDERS = {"", "sk-ant-...", "placeholder_fill_me", "fill_me", "changeme"}


class ConfigError(RuntimeError):
    pass


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    return int(raw) if raw else default


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    return float(raw) if raw else default


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8095
    token: str = ""
    public_ws_url: str = ""
    relay_ws_url: str = ""            # 经阿里云上海中转（frp）进来的请求回这个语音地址；空 = 不区分
    allowed_device_ids: tuple[str, ...] = ()

    # 音频
    downlink_sample_rate: int = 24000
    uplink_sample_rate: int = 16000  # 固件恒定 16k，见 research §1.3
    frame_ms: int = 60
    protocol_version: int = 1

    # VAD
    vad_silence_ms: int = 800
    vad_min_speech_ms: int = 300
    vad_max_utterance_ms: int = 15000
    vad_aggressiveness: int = 2
    # 断句模型（app/turn.py，pipecat smart-turn v3）：停顿 turn_check_ms 时判断「说完了没有」，
    # 像没说完就把收尾静音放宽到 turn_incomplete_ms；空 = 不用模型，只看静音
    turn_model: str = ""
    turn_threads: int = 1
    turn_check_ms: int = 200
    turn_threshold: float = 0.5
    turn_complete_ms: int = 800      # 模型说「说完了」：静音这么久收尾（默认跟 vad_silence_ms 一样，不抢话）
    turn_incomplete_ms: int = 1600   # 模型说「没说完」（识别出来带问号的除外）
    turn_open_ms: int = 2000         # 识别出来停在逗号或「然后 / 就是」上
    turn_filler_ms: int = 2500       # 只听到「嗯」「啊」：多等一会儿，看后面还有没有话
    turn_ignore_fillers: bool = True  # 等完还是只有语气词就不当一句话（盒子刚问过问题的除外）

    # ASR
    asr_provider: str = "sherpa-onnx"      # sherpa-onnx（流式，默认）| qwen3-http（GX10 上的 Qwen3-ASR）| faster-whisper | null
    asr_http_url: str = "http://127.0.0.1:8091/v1/audio/transcriptions"  # 仅 qwen3-http
    asr_http_timeout: float = 5.0          # 仅 qwen3-http；超时当噪声（多半是模型在复读电视里的歌）
    asr_model_dir: str = ""                # sherpa-onnx 模型目录（GitHub release 解开的那个）
    asr_num_threads: int = 4               # sherpa-onnx onnxruntime 线程数
    asr_tail_padding_ms: int = 300         # 收尾补多少静音，让最后一个字解出来
    asr_model: str = "small"               # 仅 faster-whisper
    asr_language: str = "zh"               # 仅 faster-whisper
    asr_compute_type: str = "int8"
    asr_device: str = "cpu"                # 仅 faster-whisper

    # LLM
    llm_model: str = "claude-sonnet-5"
    llm_max_tokens: int = 1024
    llm_thinking: str = "disabled"
    llm_effort: str = "low"
    llm_history_turns: int = 8
    llm_speculative: bool = True        # 识别结果稳定就先起 LLM，不等 listen stop
    llm_speculative_stable_ms: int = 150   # partial 不变多久算稳定
    llm_speculative_silence_ms: int = 200  # VAD 尾静音多久算「人说完了」
    llm_speculative_whole_ms: int = 350    # 整段识别（GX10）抢跑：尾静音到这么久就先认先问
    llm_first_min_chars: int = 2   # 首句最短：太短的碎片（"嗯，"）没意义
    llm_first_max_chars: int = 15  # 首句上限：到这还没标点就硬切，保首帧延迟
    llm_filler_after_ms: int = 2500  # 干等这么久还没出字就先垫一句；0=关。1.8 s 时电视台词也会被垫一句再判【忽略】
    session_idle_seconds: int = 600
    # 只听用户：盒子答完会自动再开麦，电视 / 音乐会被当成提问。这么久没等到一句正经话就挂断回待机
    followup_window_s: float = 12.0   # 答完之后；0=不挂
    connect_window_s: float = 20.0    # 刚唤醒还没说第一句
    # 一问一答：答完直接挂断回待机，不再听，按键或喊醒才开始下一问（9/25 用户要的，防抢答）
    one_turn: bool = False
    # 声纹（app/voiceprint.py）：模型路径空 = 不开；enroll 只录不拦，on 低于门槛不回答
    voiceprint_model: str = ""
    voiceprint_dir: str = ""
    voiceprint_mode: str = "enroll"
    voiceprint_threshold: float = 0.35
    keep_recent: int = 3  # 滚动存最近几段本人说的话（9/26 用户：「至少存三段」）；0 = 不存
    recent_dir: str = "~/xz-data/recent"
    anthropic_api_key: str = field(default="", repr=False)
    anthropic_auth_token: str = field(default="", repr=False)  # Bearer 型网关（智谱 GLM）
    anthropic_base_url: str = ""
    # 本地大脑（TP3 集群上的 vLLM，Anthropic 兼容端点）：填了就先打本地，连不上 / 出错 /
    # 首字太慢就这一轮退回上面的云端。本地没有智谱云端那道 1301 内容审核，首字也快
    llm_local_base_url: str = ""
    llm_local_model: str = "GLM-5.3-Flash-EXL3"
    llm_local_first_s: float = 2.5   # 本地这么久还没出第一个字就改走云端

    # 联网搜索（tavily 云端 / searxng 自建，见 app/tools.py）
    search_enabled: bool = True
    # tavily = 云端搜索 API（自带合成要点，中文准）；searxng = 自建，查询词不出内网
    search_provider: str = "tavily"
    tavily_api_key: str = field(default="", repr=False)
    # 几个 Tavily 账号轮着用（TAVILY_API_KEYS=逗号分隔；旧的 TAVILY_API_KEY 也算进来）
    tavily_api_keys: list = field(default_factory=list, repr=False)
    tavily_base_url: str = "https://api.tavily.com"
    tavily_search_depth: str = "basic"    # basic 约 2.5s；advanced 更全但更慢更贵
    tavily_include_answer: bool = True
    searxng_url: str = "http://127.0.0.1:8080"
    search_language: str = "zh-CN"
    # SearXNG 只用这几个引擎：Bing 的中文结果是百度百科凑数（「美股今天怎么样」→「美」字词条），
    # 360 和搜狗微信带日期、够新（9/23 实测）。空 = 用实例默认的全部引擎
    searxng_engines: str = "360search,sogou wechat"
    home_city: str = ""   # 问天气没说城市时按这个；空 = 让模型问
    # 5 条比 3 条慢约 0.4s，第 2 轮的提示词也更长；口语回答用不到那么多
    search_max_results: int = 3
    search_snippet_chars: int = 220
    search_timeout_s: float = 8.0
    # Tavily 这么久还没回（9/24 实测 3.4~8.5 s）就同时问一下 SearXNG，谁先拿到结果用谁；0 = 不对冲
    search_hedge_s: float = 3.5
    # 「今天有什么新闻」这类泛问走 NewsNow 热榜（github.com/ourongxing/newsnow），
    # 比让 Tavily 随便捞三条准：9/24 Tavily 回的是「文化类消息」，热榜头条是中美元首会晤
    news_hotlist: bool = True
    newsnow_url: str = "https://newsnow.busiyi.world"
    newsnow_sources: str = "thepaper,cls-hot,weibo"
    search_always: bool = False   # True = 每轮都挂工具，不走关键词闸门
    # 闸门放行时，用用户原话跟第 1 轮 LLM 并行抢跑一次搜索（见 tools.SearchPrefetch）
    search_prefetch: bool = True
    # True = 闸门放行就先查、把结果直接塞进对话问一轮；False = 老的挂工具两轮
    search_direct: bool = True
    tool_max_rounds: int = 2   # 一轮对话里最多让模型查几次
    # 不给模型挂原生工具，改成让它在要查的时候只说一句【搜索：关键词】，服务端代查。
    # 智谱网关一挂 tools，所有模型首字都慢 3~5 秒（9/23 实测 5 个模型），不挂 1.2 秒。
    llm_text_search: bool = True

    # 人设
    persona_file: str = "../persona/bright-junior-persona.md"
    user_title: str = ""
    assistant_name: str = "布莱特"
    skills: str = ""

    # TTS
    tts_provider: str = "edge"  # edge（默认，微软 Edge 神经语音）| voxtts（VoxCPM2）| volc（豆包语音）
    tts_base_url: str = "http://127.0.0.1:4690"
    # 留空用 provider 默认：edge=茉莉（zh-CN-XiaoxiaoNeural），voxtts=my_voice，volc=小何 2.0
    tts_voice: str = ""
    tts_timeout: float = 120.0
    tts_prewarm_silence_ms: int = 250  # 说完后静音这么久就先握 Edge 连接（藏掉 ~0.9s 握手）
    # 豆包语音（火山引擎 seed-tts-2.0）；key 由 doubao-key 写进 .env，出错时退回 Edge 茉莉
    volc_api_key: str = field(default="", repr=False)
    volc_resource_id: str = "seed-tts-2.0"
    volc_instruction: str = ""  # 语音指令（不计费），如「用上海话说」
    volc_speech_rate: int = 0  # -50..100，0 = 原速，100 = 两倍速
    # GX10 上的 CosyVoice3（用户本人的声音讲上海话，本地、不花钱）；空 = 不用
    cosyvoice_url: str = ""
    tts_local_hours: str = "23:00-11:00"  # 这段时间先用本地 CosyVoice，其余时间豆包优先（9/26 用户定）
    tts_cache_dir: str = "~/xz-data/tts-cache"  # 豆包念过的短句存这里，同一句不再花字数；空 = 不存
    # 盒子这条通道让模型直接用方言写回答（如「上海话」），配合豆包的方言指令念得更地道；飞书不受影响
    voice_dialect: str = ""

    ws_heartbeat: float = 30.0

    # 飞书入口：用飞书 App 当另一张嘴（语音转文字在飞书客户端做，不花钱）
    feishu_enabled: bool = False
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_allowed_open_ids: tuple[str, ...] = ()   # 空 = 谁都能聊（不推荐）
    feishu_history_turns: int = 12

    # 局域网直连（可选）：盒子在家时 OTA 回局域网 ws，省掉 Cloudflare 隧道往返
    lan_ws_url: str = ""              # 空 = 关闭（默认）
    lan_public_ips: tuple[str, ...] = ()  # 本机公网出口 IP；留空则启动时自动探测
    lan_trust_private: bool = True    # 直连进来的私网源地址也算「在家」
    lan_ipv6_prefix: int = 64         # IPv6 出口按几位前缀算「同一家」

    # 无线升级（app/firmware.py）：清单和固件放部署目录外，deploy 的 rsync --delete 碰不到
    fw_dir: str = "/srv/xiaozhi-fw"

    # Claude Code 任务播报（app/announce.py）：token 为空 = 关
    announce_token: str = field(default="", repr=False)
    announce_quiet: str = "23:00-08:00"   # 这段时间来的播报直接丢，盒子也不轮询；空 = 不设
    announce_tz: str = "Asia/Shanghai"
    announce_merge_s: float = 10.0        # 第一条到了先等这么久，期间来的合成一段念
    announce_ttl_s: float = 900.0         # 这么久还没念出去就扔
    announce_rewrite: bool = True         # 配了 XZ_VOICE_DIALECT 时，播报先让大模型改写成方言口语再念

    @property
    def lan_enabled(self) -> bool:
        return bool(self.lan_ws_url)

    @property
    def persona_path(self) -> Path:
        p = Path(self.persona_file)
        return p if p.is_absolute() else (SERVER_DIR / p).resolve()

    @property
    def frame_samples_downlink(self) -> int:
        return self.downlink_sample_rate * self.frame_ms // 1000

    @property
    def frame_samples_uplink(self) -> int:
        return self.uplink_sample_rate * self.frame_ms // 1000


def load_config(require_api_key: bool = True) -> Config:
    if ENV_FILE.exists():
        # override=True：server/.env 是本服务自己的配置文件，必须压过外面 shell 里
        # 顺手 export 的同名变量（这台机器全局就 export 了 ANTHROPIC_BASE_URL，
        # override=False 时会被它悄悄劫持，排查起来很费劲）。
        load_dotenv(ENV_FILE, override=True)
    elif require_api_key and not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        raise ConfigError(_ENV_MISSING)

    cfg = Config(
        host=_get("XZ_HOST", "0.0.0.0"),
        port=_get_int("XZ_PORT", 8095),
        token=_get("XZ_TOKEN"),
        public_ws_url=_get("XZ_PUBLIC_WS_URL"),
        relay_ws_url=_get("XZ_RELAY_WS_URL"),
        allowed_device_ids=tuple(
            d.strip().lower() for d in _get("XZ_ALLOWED_DEVICE_IDS").split(",") if d.strip()
        ),
        downlink_sample_rate=_get_int("XZ_DOWNLINK_SAMPLE_RATE", 24000),
        frame_ms=_get_int("XZ_FRAME_MS", 60),
        protocol_version=_get_int("XZ_PROTOCOL_VERSION", 1),
        vad_silence_ms=_get_int("XZ_VAD_SILENCE_MS", 800),
        vad_min_speech_ms=_get_int("XZ_VAD_MIN_SPEECH_MS", 300),
        vad_max_utterance_ms=_get_int("XZ_VAD_MAX_UTTERANCE_MS", 15000),
        vad_aggressiveness=_get_int("XZ_VAD_AGGRESSIVENESS", 2),
        turn_model=_get("XZ_TURN_MODEL", ""),
        turn_threads=_get_int("XZ_TURN_THREADS", 1),
        turn_check_ms=_get_int("XZ_TURN_CHECK_MS", 200),
        turn_threshold=_get_float("XZ_TURN_THRESHOLD", 0.5),
        turn_complete_ms=_get_int("XZ_TURN_COMPLETE_MS", 800),
        turn_incomplete_ms=_get_int("XZ_TURN_INCOMPLETE_MS", 1600),
        turn_open_ms=_get_int("XZ_TURN_OPEN_MS", 2000),
        turn_filler_ms=_get_int("XZ_TURN_FILLER_MS", 2500),
        turn_ignore_fillers=_get("XZ_TURN_IGNORE_FILLERS", "1") not in ("0", "false", "no"),
        asr_provider=_get("XZ_ASR_PROVIDER", "sherpa-onnx"),
        asr_model_dir=_get("XZ_ASR_MODEL_DIR"),
        asr_http_url=_get("XZ_ASR_HTTP_URL", "http://127.0.0.1:8091/v1/audio/transcriptions"),
        asr_http_timeout=_get_float("XZ_ASR_HTTP_TIMEOUT", 5.0),
        asr_num_threads=_get_int("XZ_ASR_NUM_THREADS", 4),
        asr_tail_padding_ms=_get_int("XZ_ASR_TAIL_PADDING_MS", 300),
        asr_model=_get("XZ_ASR_MODEL", "small"),
        asr_language=_get("XZ_ASR_LANGUAGE", "zh"),
        asr_compute_type=_get("XZ_ASR_COMPUTE_TYPE", "int8"),
        asr_device=_get("XZ_ASR_DEVICE", "cpu"),
        llm_model=_get("XZ_LLM_MODEL", "claude-sonnet-5"),
        llm_max_tokens=_get_int("XZ_LLM_MAX_TOKENS", 1024),
        llm_thinking=_get("XZ_LLM_THINKING", "disabled"),
        llm_effort=_get("XZ_LLM_EFFORT", "low"),
        llm_history_turns=_get_int("XZ_LLM_HISTORY_TURNS", 8),
        llm_speculative=_get("XZ_LLM_SPECULATIVE", "1") not in ("0", "false", "no"),
        llm_speculative_stable_ms=_get_int("XZ_LLM_SPECULATIVE_STABLE_MS", 150),
        llm_speculative_silence_ms=_get_int("XZ_LLM_SPECULATIVE_SILENCE_MS", 200),
        llm_speculative_whole_ms=_get_int("XZ_LLM_SPECULATIVE_WHOLE_MS", 350),
        llm_first_min_chars=_get_int("XZ_LLM_FIRST_MIN_CHARS", 2),
        llm_first_max_chars=_get_int("XZ_LLM_FIRST_MAX_CHARS", 15),
        llm_filler_after_ms=_get_int("XZ_LLM_FILLER_AFTER_MS", 2500),
        session_idle_seconds=_get_int("XZ_SESSION_IDLE_SECONDS", 600),
        followup_window_s=_get_float("XZ_FOLLOWUP_WINDOW_S", 12.0),
        connect_window_s=_get_float("XZ_CONNECT_WINDOW_S", 20.0),
        one_turn=_get("XZ_ONE_TURN", "0").lower() in ("1", "true", "yes", "on"),
        voiceprint_model=_get("XZ_VOICEPRINT_MODEL", ""),
        voiceprint_dir=os.path.expanduser(_get("XZ_VOICEPRINT_DIR", "~/xz-data/voiceprint")),
        voiceprint_mode=_get("XZ_VOICEPRINT_MODE", "enroll").lower(),
        voiceprint_threshold=_get_float("XZ_VOICEPRINT_THRESHOLD", 0.35),
        keep_recent=_get_int("XZ_KEEP_RECENT", 3),
        recent_dir=os.path.expanduser(_get("XZ_RECENT_DIR", "~/xz-data/recent")),
        anthropic_api_key=_get("ANTHROPIC_API_KEY"),
        anthropic_auth_token=_get("ANTHROPIC_AUTH_TOKEN"),
        anthropic_base_url=_get("ANTHROPIC_BASE_URL"),
        llm_local_base_url=_get("XZ_LLM_LOCAL_BASE_URL"),
        llm_local_model=_get("XZ_LLM_LOCAL_MODEL", "GLM-5.3-Flash-EXL3"),
        llm_local_first_s=_get_float("XZ_LLM_LOCAL_FIRST_S", 2.5),
        search_enabled=_get("XZ_SEARCH_ENABLED", "1") not in ("0", "false", "no"),
        search_provider=_get("XZ_SEARCH_PROVIDER", "tavily").lower(),
        tavily_api_key=_get("TAVILY_API_KEY"),
        tavily_api_keys=list(dict.fromkeys(
            k.strip() for k in (_get("TAVILY_API_KEYS").replace(" ", ",").split(",")
                                + [_get("TAVILY_API_KEY")]) if k.strip())),
        tavily_base_url=_get("XZ_TAVILY_BASE_URL", "https://api.tavily.com"),
        tavily_search_depth=_get("XZ_TAVILY_DEPTH", "basic"),
        tavily_include_answer=_get("XZ_TAVILY_ANSWER", "1") not in ("0", "false", "no"),
        searxng_url=_get("XZ_SEARXNG_URL", "http://127.0.0.1:8080"),
        search_language=_get("XZ_SEARCH_LANGUAGE", "zh-CN"),
        searxng_engines=_get("XZ_SEARXNG_ENGINES", "360search,sogou wechat"),
        home_city=_get("XZ_HOME_CITY", ""),
        search_max_results=_get_int("XZ_SEARCH_MAX_RESULTS", 3),
        search_snippet_chars=_get_int("XZ_SEARCH_SNIPPET_CHARS", 220),
        search_timeout_s=_get_float("XZ_SEARCH_TIMEOUT_S", 8.0),
        search_hedge_s=_get_float("XZ_SEARCH_HEDGE_S", 3.5),
        news_hotlist=_get("XZ_NEWS_HOTLIST", "1") not in ("0", "false", "no"),
        newsnow_url=_get("XZ_NEWSNOW_URL", "https://newsnow.busiyi.world"),
        newsnow_sources=_get("XZ_NEWSNOW_SOURCES", "thepaper,cls-hot,weibo"),
        search_always=_get("XZ_SEARCH_ALWAYS", "0") in ("1", "true", "yes"),
        search_prefetch=_get("XZ_SEARCH_PREFETCH", "1") not in ("0", "false", "no"),
        search_direct=_get("XZ_SEARCH_DIRECT", "1") not in ("0", "false", "no"),
        tool_max_rounds=_get_int("XZ_TOOL_MAX_ROUNDS", 2),
        llm_text_search=_get("XZ_LLM_TEXT_SEARCH", "1") not in ("0", "false", "no"),
        persona_file=_get("XZ_PERSONA_FILE", "../persona/bright-junior-persona.md"),
        user_title=_get("XZ_USER_TITLE"),
        assistant_name=_get("XZ_ASSISTANT_NAME", "布莱特"),
        skills=_get("XZ_SKILLS"),
        tts_provider=_get("XZ_TTS_PROVIDER", "edge").lower(),
        tts_base_url=_get("XZ_TTS_BASE_URL", "http://127.0.0.1:4690").rstrip("/"),
        tts_voice=_get("XZ_TTS_VOICE", ""),
        tts_timeout=_get_float("XZ_TTS_TIMEOUT", 120.0),
        tts_prewarm_silence_ms=_get_int("XZ_TTS_PREWARM_SILENCE_MS", 250),
        volc_api_key=_get("XZ_VOLC_API_KEY"),
        volc_resource_id=_get("XZ_VOLC_RESOURCE_ID", "seed-tts-2.0"),
        volc_instruction=_get("XZ_VOLC_INSTRUCTION"),
        volc_speech_rate=_get_int("XZ_VOLC_SPEECH_RATE", 0),
        cosyvoice_url=_get("XZ_COSYVOICE_URL"),
        tts_local_hours=_get("XZ_TTS_LOCAL_HOURS", "23:00-11:00"),
        tts_cache_dir=os.path.expanduser(_get("XZ_TTS_CACHE_DIR", "~/xz-data/tts-cache")),
        voice_dialect=_get("XZ_VOICE_DIALECT"),
        ws_heartbeat=_get_float("XZ_WS_HEARTBEAT", 30.0),
        feishu_enabled=_get("XZ_FEISHU_ENABLED", "0") not in ("0", "false", "no", ""),
        feishu_app_id=_get("XZ_FEISHU_APP_ID"),
        feishu_app_secret=_get("XZ_FEISHU_APP_SECRET"),
        feishu_allowed_open_ids=tuple(
            x.strip() for x in _get("XZ_FEISHU_ALLOWED_OPEN_IDS").split(",") if x.strip()
        ),
        feishu_history_turns=_get_int("XZ_FEISHU_HISTORY_TURNS", 12),
        lan_ws_url=_get("XZ_LAN_WS_URL"),
        lan_public_ips=tuple(
            ip.strip() for ip in _get("XZ_LAN_PUBLIC_IPS").split(",") if ip.strip()
        ),
        lan_trust_private=_get("XZ_LAN_TRUST_PRIVATE", "1") not in ("0", "false", "no"),
        lan_ipv6_prefix=_get_int("XZ_LAN_IPV6_PREFIX", 64),
        fw_dir=_get("XZ_FW_DIR", "/srv/xiaozhi-fw"),
        announce_token=_get("XZ_ANNOUNCE_TOKEN"),
        announce_quiet=_get("XZ_ANNOUNCE_QUIET", "23:00-08:00"),
        announce_tz=_get("XZ_ANNOUNCE_TZ", "Asia/Shanghai"),
        announce_merge_s=_get_float("XZ_ANNOUNCE_MERGE_S", 10.0),
        announce_ttl_s=_get_float("XZ_ANNOUNCE_TTL_S", 900.0),
        announce_rewrite=_get("XZ_ANNOUNCE_REWRITE", "1").lower() not in ("0", "false", "no", "off"),
    )

    has_key = cfg.anthropic_api_key.lower() not in _PLACEHOLDERS
    has_token = cfg.anthropic_auth_token.lower() not in _PLACEHOLDERS
    if require_api_key and not (has_key or has_token):
        raise ConfigError(_KEY_MISSING)
    if require_api_key and has_key and not cfg.anthropic_api_key.startswith("sk-ant-"):
        # 不拦：可能是自建网关的 key。但要吼一声，否则只会看到盒子念「钥匙没配好」
        logging.getLogger("xz.config").warning(
            "ANTHROPIC_API_KEY 看着不像真 key（不以 sk-ant- 开头）。"
            "OTA/WS/ASR/TTS 都能跑，但 LLM 会 401，盒子会念「服务端的接口钥匙还没配好」。"
            "去 %s 填上真 key 再 ./run.sh restart。", ENV_FILE
        )
    if cfg.protocol_version not in (1, 2, 3):
        raise ConfigError("XZ_PROTOCOL_VERSION 只能是 1 / 2 / 3")
    if cfg.tts_provider not in ("edge", "voxtts", "volc", "cosyvoice"):
        raise ConfigError("XZ_TTS_PROVIDER 只能是 edge / voxtts / volc / cosyvoice")
    if cfg.tts_provider == "cosyvoice" and not cfg.cosyvoice_url:
        raise ConfigError("XZ_TTS_PROVIDER=cosyvoice 时要填 XZ_COSYVOICE_URL（tools/wu_server.py 的地址）")
    if cfg.asr_provider.lower() in ("qwen3-http", "qwen3", "gx10") and cfg.asr_model_dir:
        d = Path(cfg.asr_model_dir).expanduser()
        d = d if d.is_absolute() else (SERVER_DIR / d)
        cfg.asr_model_dir = str(d.resolve()) if d.is_dir() else ""  # 没有就不回退，GX10 挂了这句算没听清
    if cfg.asr_provider.lower() in ("sherpa-onnx", "sherpa_onnx", "sherpa", "streaming"):
        if not cfg.asr_model_dir:
            raise ConfigError(
                "XZ_ASR_PROVIDER=sherpa-onnx 时必须给 XZ_ASR_MODEL_DIR（模型目录）。\n"
                "见 README §3.2；或改回旧路径 XZ_ASR_PROVIDER=faster-whisper。"
            )
        d = Path(cfg.asr_model_dir).expanduser()
        if not (d if d.is_absolute() else (SERVER_DIR / d)).is_dir():
            raise ConfigError(f"XZ_ASR_MODEL_DIR 指的目录不存在：{cfg.asr_model_dir}")
        cfg.asr_model_dir = str((d if d.is_absolute() else (SERVER_DIR / d)).resolve())
    return cfg
