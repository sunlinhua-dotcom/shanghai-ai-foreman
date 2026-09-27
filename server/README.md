# server/ — 小智协议自建服务端

给跑 [xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) 固件的盒子用的后端：OTA、WebSocket 语音通道、流式识别、大模型、配音、播报收件箱，都在一个 aiohttp 进程里，一个端口。协议按固件 v2.2.4 的源码逐条对过，关键出处写在代码注释里。

```
盒子麦克风 ──16k / 60 ms Opus──▶ VAD 分段 ─▶ sherpa-onnx 流式识别 ─▶ smart-turn 断句
                                                            │
                          ┌── 大模型（流式，按句切）◀──────────┘
                          ▼
            吴语配音 / Edge 神经语音（预热 + 下一句预取）
                          │
                 24k PCM ─▶ Opus 60 ms ──▶ 盒子扬声器
```

## 装和跑

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # 至少填一个大模型的 key
./run.sh fg                   # 前台跑，Ctrl-C 退出；也可以 start / stop / status / logs
./run.sh test                 # 假设备端到端跑一轮（不需要真机）
```

用 Python 3.12，是因为 `ctranslate2` / `onnxruntime` 还没有 3.14 的 wheel。

盒子这边：配网页 →「高级选项」→ OTA 地址填 `http(s)://<你的服务器>/xiaozhi/ota/`。也可以在编译固件时写进 `CONFIG_OTA_URL`，见 [../firmware/README.md](../firmware/README.md)。

## 配置

全部在 `.env` 里（已 gitignore）。`.env.example` 每一项都有注释，这里只列最常改的：

| 变量 | 说明 |
|---|---|
| `ANTHROPIC_API_KEY`，或 `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` | 大模型。Claude 直接填 key；智谱 GLM 这类 Anthropic 兼容网关填后两项，这时 `XZ_LLM_THINKING` 要保持 `disabled` |
| `XZ_LLM_MODEL` | 如 `claude-sonnet-5`、`glm-5.3-flash` |
| `XZ_LLM_LOCAL_BASE_URL` / `XZ_LLM_LOCAL_MODEL` | 可选的本地 vLLM。连不上、出错或首字超过 `XZ_LLM_LOCAL_FIRST_S` 时，这一轮退回云端 |
| `XZ_TOKEN`、`XZ_ALLOWED_DEVICE_IDS` | WS 鉴权和设备 MAC 白名单。**服务开在公网时两个都要填** |
| `XZ_ASR_PROVIDER` / `XZ_ASR_MODEL_DIR` | 默认 `sherpa-onnx`，模型用 `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`（k2-fsa 的 GitHub release） |
| `XZ_TURN_MODEL` | pipecat smart-turn v3 的 onnx。留空就只用静音加语气词、问号规则 |
| `XZ_TTS_PROVIDER` | `edge`（默认，免 key）/ `cosyvoice`（自建吴语配音，出错退 Edge）/ `volc`（豆包语音，可再串吴语配音）/ `voxtts` |
| `XZ_COSYVOICE_URL` | 吴语配音服务地址，如 `http://127.0.0.1:8097` |
| `XZ_VOICE_DIALECT` | 让模型直接用方言写回答，如 `上海话` |
| `XZ_ANNOUNCE_TOKEN` | 播报接口的 Bearer token。留空 = 关掉播报 |
| `XZ_ANNOUNCE_QUIET` / `XZ_ANNOUNCE_TZ` | 静默时段（默认 `23:00-08:00`、`Asia/Shanghai`），填 `off` 就全天都播 |
| `XZ_SEARCH_PROVIDER`、`TAVILY_API_KEY(S)`、`XZ_SEARXNG_URL` | 联网搜索。多个 Tavily key 用逗号分隔，当月额度用完自动换下一个 |
| `XZ_HOME_CITY` | 问天气没说城市时查哪里 |

## 播报：让盒子替别的程序喊一声

```bash
curl -X POST http://127.0.0.1:8095/xiaozhi/announce \
  -H "Authorization: Bearer $XZ_ANNOUNCE_TOKEN" -H 'Content-Type: application/json' \
  -d '{"text": "目录 foo 的任务完成了。", "kind": "done", "project": "foo"}'
```

- 消息按设备排队，第一条到了会等 `XZ_ANNOUNCE_MERGE_S` 秒，期间来的合成一段一起念。
- 盒子待机时长轮询 `GET /xiaozhi/inbox`，有货就自己唤醒（`listen detect text=__announce__`），念完回待机。
- 开着 `XZ_ANNOUNCE_REWRITE` 时，先让大模型把原文改写成一句口语（配合 `XZ_VOICE_DIALECT=上海话` 就是上海话）。
- 超过 `XZ_ANNOUNCE_TTL_S` 还没念出去就丢掉。队列只在内存里，播报本来就是过时就没意义的东西。

接 Claude Code：在它的 Stop / Notification 钩子里调上面这条 curl 就行。

## 吴语配音服务（可选）

`tools/wu_server.py` 包的是 WenetSpeech-Wu 的 CosyVoice2-Wu-SFT，要一块能跑 CosyVoice2 的 GPU。

1. 按 CosyVoice 官方说明装好环境，下载 CosyVoice2-Wu-SFT 权重（请先查看它的许可）。
2. 把 `wu_server.py` 和 `voicing.py` 放进 CosyVoice 目录。参考 `deploy/wu_voices.example.json` 写一份声音表，用 `WU_VOICES` 指过去：
   - `llm_wav` / `llm_text`：一段上海话录音和它的逐字原文，定「怎么讲」；
   - `wav`：一段干净录音，定「什么音色」。
3. 起服务（`deploy/cosyvoice-sh.service` 是 systemd 模板），然后 `.env` 里写 `XZ_TTS_PROVIDER=cosyvoice`、`XZ_COSYVOICE_URL=http://<那台机器>:8097`。

盒子上音量加减一起按，或者讲「换只声音」，就轮到下一种声音。选中的声音记在 `~/xz-data/voice.json`，声音 id 要跟声音表对上（`app/tts/voices.py`）。**只克隆你有权使用的声音。**

## 测试

```bash
./run.sh test --text "今天天气怎么样"     # 换一句测试语音
./run.sh test --abort-after 1.5           # 首帧后 1.5 s 打断
python tools/fake_box.py say:今朝天气哪能   # 照真盒子的节奏走整条链路，量「说完 → 听到第一个字」
python tools/asr_bench.py --synth         # ASR 引擎对照：字错率 + 收尾延迟
```

每轮下发第一帧音频时，日志里打一行 `turn latency: asr=… llm_first=… tts_first=… total=…`，四段相加等于服务端侧的首句延迟。

## 部署

`deploy/xiaozhi-server.service` 是 systemd 模板：服务只听 `127.0.0.1`，公网入口建议用 Cloudflare Tunnel 之类的反代。挂公网之前务必填好 `XZ_TOKEN` 和 `XZ_ALLOWED_DEVICE_IDS`。

## 目录

```
app/
  main.py         入口：OTA + WS + 播报 + 收件箱，同一个端口
  config.py       .env → Config，缺什么就报错并说明怎么填
  ota.py          POST /xiaozhi/ota/
  session.py      一条连接一个会话：状态机、VAD、抢跑、一轮对话、MCP
  announce.py     播报队列、合并、静默时段、上海话改写
  asr/            sherpa-onnx（流式）/ faster-whisper / HTTP ASR
  llm/            Anthropic 流式 + 按句切分 + 搜索工具
  tts/            edge / cosyvoice（吴语）/ volc / voxtts + 兜底与缓存
  turn.py         smart-turn v3 断句（numpy 特征 + onnxruntime）
  tools.py        搜索（Tavily / SearXNG 对冲、NewsNow）、天气、上海话查询词归一
  wu_text.py      上海话读法：数字、多音字
  voiceprint.py   声纹，只听主人说话（可选）
  persona.py      人设 md → system prompt
tools/            假设备、ASR 对照、断句模型核对、配音服务
deploy/           systemd 模板、声音表示例
```
