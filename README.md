# Shanghai AI Foreman · 打桩模子

<p align="center"><img src="docs/eyes.gif" width="320" alt="SDF eyes"><br><sub>The box's eyes, rendered on a desktop by <code>scripts/eyesim</code> from the firmware's own <code>eye_engine.cc</code>.</sub></p>

A pink ESP32 desk box that watches your AI coding agents and **yells at you in Shanghainese** when they finish, fail, or sit there waiting for your permission.

> 册那！伊又等侬授权了！ — *"Damn, it's waiting for your approval again!"*

It started as a way for one person to keep track of a dozen AI agents running in parallel (Claude Code, Codex, GLM, DeepSeek…). Every agent event goes to a tiny box on the desk. The box wakes up by itself, rewrites the event as one line of Shanghainese, and says it out loud. You can also just talk to it: the wake word is 「打桩模子」, Shanghainese slang for a street tout.

[中文说明](README.zh-CN.md)

## What's inside

| | |
|---|---|
| **Eyes** | A signed-distance-field eye engine on the 240×240 screen at about 30 fps. Every shape (the glass capsule, ^ ^, beating hearts, stars, loading arcs, a one-stroke check mark) is a distance field. Morphing, anti-aliasing, glow and bevel lighting all come from that same distance. Colours blend in Oklab. Frames are rendered across both ESP32-S3 cores. It is plain C++, so `scripts/eyesim` renders the exact same code on a desktop. |
| **Wake word** | A custom MultiNet word 「打桩模子」 (*dang zang mu zi*), recognised on the device. |
| **Announcer** | `POST /xiaozhi/announce` puts a message in a per-device queue. The idle box long-polls an inbox, wakes itself up, and reads the queue out as one merged, Shanghainese-rewritten sentence. It has quiet hours, TTL and merge windows. |
| **Ears** | Streaming ASR with sherpa-onnx (a zipformer transducer). Only the tail is left to decode after you stop talking (about 9 ms median). End-of-turn is decided by pipecat smart-turn v3 plus text rules, so it doesn't cut in while you are thinking of a word. |
| **Brain** | Any Anthropic-compatible endpoint: Claude, or GLM through its Anthropic gateway, or a local vLLM with cloud fallback. It streams sentence by sentence. A speculative start sends the LLM request before the device says "stop listening", which hides a few hundred ms. |
| **Mouth** | A Wu-dialect TTS server (`server/tools/wu_server.py`) around CosyVoice2-Wu-SFT from WenetSpeech-Wu. Each voice is cloned zero-shot from one reference clip for the Shanghainese delivery and one for the timbre. It has several switchable voices: press volume up and down together, or say 「换只声音」. Microsoft Edge neural TTS is the zero-key fallback. |
| **Search** | Tavily with a SearXNG hedge, NewsNow headlines, and Open-Meteo weather. Shanghainese queries are normalised first (今朝→今天, 几钿→多少钱…). |
| **4G** | Works on the ML307 Cat.1 modem. Its TLS stack can't handshake with Cloudflare, so on 4G the firmware talks plain HTTP to your server, with an optional relay host. |

## How it fits together

```
  your agents ──POST /xiaozhi/announce──▶ ┌──────────────── server (Python, aiohttp) ────────────────┐
                                          │  queue ─▶ Shanghainese rewrite (LLM)                     │
  ┌────────── box (ESP32-S3) ──────────┐  │                                                          │
  │ wake word · eyes · Opus 16k/60 ms  │◀─┼─ inbox long-poll / OTA / WebSocket (xiaozhi protocol)    │
  │ vol+ & vol− together = next voice  │  │  VAD ─▶ streaming ASR ─▶ smart-turn ─▶ LLM (streamed)   │
  └────────────────────────────────────┘  │                     └─▶ Wu TTS / Edge TTS ─▶ Opus 24k    │
                                          └──────────────────────────────────────────────────────────┘
```

## Hardware

The target is the **无名科技 星智 1.54 ML307** box (ESP32-S3, 1.54″ 240×240 TFT, ML307 4G), board `xingzhi-cube-1.54tft-ml307` in [78/xiaozhi-esp32](https://github.com/78/xiaozhi-esp32). Other xiaozhi boards can use the server unchanged. The eye UI is only wired up for this board.

## Quick start

**Server** (Python 3.12, Linux or macOS):

```bash
cd server
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # fill in an LLM key; everything else has defaults
./run.sh fg                 # foreground; ./run.sh test runs a fake device end to end
```

**Firmware** (ESP-IDF 5.5):

```bash
cd firmware
./setup.sh                  # clones upstream at the pinned commit and lays overlay/ on top
cd xiaozhi-esp32
idf.py menuconfig           # Xiaozhi Assistant → OTA URL, and "Self-host" for 4G
python scripts/release.py xingzhi-cube-1.54tft-ml307
```

Then point the box's OTA URL at `http(s)://<your-server>/xiaozhi/ota/`. See [server/README.md](server/README.md) and [firmware/README.md](firmware/README.md) for details.

**Announce something:**

```bash
curl -X POST https://<your-server>/xiaozhi/announce \
  -H "Authorization: Bearer $XZ_ANNOUNCE_TOKEN" -H 'Content-Type: application/json' \
  -d '{"text": "The ESP32 task in project foo is done.", "kind": "done", "project": "foo"}'
```

## Layout

```
firmware/   overlay on upstream xiaozhi-esp32 + setup.sh
server/     xiaozhi-protocol server: OTA, WebSocket, ASR, LLM, TTS, announcer (MIT)
persona/    the assistant persona the system prompt is built from
```

## License

MIT, see [LICENSE](LICENSE). `firmware/overlay/` contains modified files from upstream xiaozhi-esp32, which is also MIT; its notice is kept in `firmware/LICENSE-upstream-MIT`. The SDF helpers follow Inigo Quilez's 2D distance functions (MIT), and the colour maths is Björn Ottosson's Oklab.

## Credits

[78/xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) · [Inigo Quilez's distance functions](https://iquilezles.org/articles/distfunctions2d/) · [Oklab](https://bottosson.github.io/posts/oklab/) · [FluxGarage RoboEyes](https://github.com/FluxGarage/RoboEyes) (the first version's inspiration) · [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) · [pipecat smart-turn](https://github.com/pipecat-ai/smart-turn) · [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) · WenetSpeech-Wu CosyVoice2-Wu-SFT (NWPU ASLP; get the weights yourself and check their licence) · [NewsNow](https://github.com/ourongxing/newsnow) · [Open-Meteo](https://open-meteo.com/)

This is a personal project, not affiliated with Xiaozhi, Espressif, Anthropic or any model vendor.
