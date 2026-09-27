# 打桩模子 · 讲上海话的 AI 监工

<p align="center"><img src="docs/eyes.gif" width="320" alt="SDF eyes"><br><sub>盒子的眼睛：<code>scripts/eyesim</code> 在电脑上用固件同一份 <code>eye_engine.cc</code> 渲染。</sub></p>

一只粉色的 ESP32 桌面小盒子。它盯着你手下那一群 AI 编程助手，谁做完了、谁出错了、谁又卡在那里等你授权，它都会**用上海话喊出来**。

> 册那！伊又等侬授权了！

起因很简单：一个人同时开着十几个 AI 干活（Claude Code、Codex、GLM、DeepSeek……），消息多了根本看不过来。于是每个 AI 的事件都推到桌上这只盒子，它会自己醒过来，把消息改写成一句上海话念给你听。平时也能直接跟它聊，唤醒词是「打桩模子」（上海话里指黄牛）。

[English](README.md)

## 有什么

| | |
|---|---|
| **眼睛** | 240×240 屏上的有向距离场（SDF）眼睛引擎，约 30 帧。玻璃胶囊、^ ^、跳动的心、星星、加载圈、一笔画出的对勾，每种眼形都是一个距离场，变形、抗锯齿、辉光和倒角光照都从同一个距离算出来。颜色在 Oklab 空间里混，两个核分工出帧。引擎是纯 C++，`scripts/eyesim` 能在电脑上跑出一模一样的画面。 |
| **唤醒词** | 设备端 MultiNet 自定义词「打桩模子」（dang zang mu zi）。 |
| **播报** | `POST /xiaozhi/announce` 把消息按设备排队。盒子待机时长轮询收件箱，有消息就自己唤醒，把积压的几条合成一句上海话念掉。支持静默时段、过期丢弃和合并窗口。 |
| **耳朵** | sherpa-onnx 流式识别，说完只剩尾巴要解，中位约 9 ms。用 pipecat smart-turn v3 加几条文字规则判断你有没有说完，想词的时候不抢话。 |
| **大脑** | 任意 Anthropic 兼容端点：Claude、智谱 GLM 的 Anthropic 网关、本地 vLLM（连不上自动退云端）都行。按句流式输出；识别稳定了就抢跑发请求，能藏掉几百毫秒。 |
| **嘴巴** | 自建吴语配音服务（`server/tools/wu_server.py`），模型是 WenetSpeech-Wu 的 CosyVoice2-Wu-SFT（西工大 ASLP；权重请自行下载并查看其许可）。每种声音用一段上海话录音定腔调、一段干净录音定音色，零样本克隆；几种声音可以轮换：音量加减一起按，或者讲「换只声音」。配音出错时退到免 key 的微软 Edge 神经语音。 |
| **搜索** | Tavily 查询，慢了就同时问 SearXNG；另有 NewsNow 热榜和 Open-Meteo 天气。上海话问法先转成普通话（今朝→今天，几钿→多少钱……）。 |
| **4G** | 支持 ML307 Cat.1 模块。它的 TLS 跟 Cloudflare 握不上手，所以走 4G 时固件对自己的服务器改用明文，还能另配一个中转地址。 |

## 硬件

目标机是**无名科技 星智 1.54 ML307**（ESP32-S3，1.54 英寸 240×240 屏，ML307 4G），对应 [78/xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) 里的 `xingzhi-cube-1.54tft-ml307`。别的小智板子可以直接用这套服务端，但眼睛界面目前只适配了这一款。

## 快速开始

**服务端**（Python 3.12，Linux 或 macOS）：

```bash
cd server
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # 填一个大模型的 key，其余都有默认值
./run.sh fg                 # 前台运行；./run.sh test 用假设备跑一轮端到端
```

**固件**（ESP-IDF 5.5）：

```bash
cd firmware
./setup.sh                  # 拉上游到固定提交，再把 overlay/ 盖上去
cd xiaozhi-esp32
idf.py menuconfig           # Xiaozhi Assistant → OTA 地址；走 4G 的再填「Self-host」
python scripts/release.py xingzhi-cube-1.54tft-ml307
```

最后把盒子的 OTA 地址填成 `http(s)://<你的服务器>/xiaozhi/ota/`。细节见 [server/README.md](server/README.md) 和 [firmware/README.md](firmware/README.md)。

**推一条播报：**

```bash
curl -X POST https://<你的服务器>/xiaozhi/announce \
  -H "Authorization: Bearer $XZ_ANNOUNCE_TOKEN" -H 'Content-Type: application/json' \
  -d '{"text": "目录 foo 的任务完成了。", "kind": "done", "project": "foo"}'
```

## 许可

MIT，见 [LICENSE](LICENSE)。`firmware/overlay/` 里有改过的上游 xiaozhi-esp32 文件，上游同样是 MIT，许可声明保留在 `firmware/LICENSE-upstream-MIT`。SDF 函数参考 Inigo Quilez 的 2D 距离函数（MIT），颜色计算用 Björn Ottosson 的 Oklab。

个人项目，与小智、乐鑫、Anthropic 及各模型厂商都没有关系。
