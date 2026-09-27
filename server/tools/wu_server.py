"""上海话配音服务：WenetSpeech-Wu 的 CosyVoice2-Wu-SFT（西工大，8000 小时吴语训练），多种声音。

部署：放在 ~/cosyvoice3/CosyVoice/ 下，systemd 模板见 deploy/cosyvoice-sh.service。
  POST /tts {"text": "…", "voice": "me"} → 流式 PCM16 单声道（chunked），采样率见 X-Sample-Rate；voice 不认识就用默认
  GET /health → 可用的声音
声音表：WU_VOICES（JSON：id → {name, wav, text, llm_wav?, llm_text?, gain_db?, eq?}），参考音 + 它的原文。按参考音零样本克隆。
  9/27 起可以分开给：llm_wav/llm_text 定「怎么讲」（要上海话录音），wav 定「什么音色、什么音质」（要干净录音，原文不用）。
  几种声音可以共用一段本地人讲的上海话录音（如 voices/nat_sh.wav）定腔调，音色各取各的参考录音。
出声前过 voicing.py 的小喇叭调音（WU_EQ=off 关；eq 按声音改档位），gain_db 把几种声音拉到一样响。
坑（9/27 实测）：官方示例把 <|wuyu|> 加在正文前，每句开头会冒出一两个乱字；加在参考原文最前面就好了。
数字改上海话说法在盒子服务器那头做（wu_numbers），这里原样念。一次只念一句（锁）。
"""
import json
import logging
import os
import re
import sys
import threading
import time

import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "third_party/Matcha-TTS")
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


def _load(path, *a, **k):  # torchaudio 2.9+ 的 load 要 torchcodec，用 soundfile 顶上
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load
from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402
from voicing import Voicing  # noqa: E402

H = os.path.expanduser("~/cosyvoice3")
MODEL_DIR = os.environ.get("WU_MODEL_DIR", f"{H}/pretrained_models/WenetSpeech-Wu/CosyVoice2-Wu-SFT")
VOICES_FILE = os.path.expanduser(os.environ.get("WU_VOICES", f"{H}/voices/voices.json"))
DEFAULT = os.environ.get("WU_DEFAULT_VOICE", "me")
PORT = int(os.environ.get("COSY_PORT", "8097"))
TAG = "<|wuyu|>"
EQ = os.environ.get("WU_EQ", "on") != "off"
EQ_ARGS = {"shelf_db": float(os.environ.get("WU_EQ_SHELF_DB", "5")), "mud_db": float(os.environ.get("WU_EQ_MUD_DB", "-3")),
           "hp_hz": float(os.environ.get("WU_EQ_HP_HZ", "120"))}
LLM_KEYS = ("prompt_text", "prompt_text_len", "llm_prompt_speech_token", "llm_prompt_speech_token_len", "llm_embedding")
FLOW_KEYS = ("flow_prompt_speech_token", "flow_prompt_speech_token_len", "prompt_speech_feat", "prompt_speech_feat_len",
             "flow_embedding")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wu-sh")

t0 = time.time()
model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, load_vllm=False, fp16=True)
SR = model.sample_rate
VOICES = json.load(open(VOICES_FILE, encoding="utf-8"))
for vid, v in VOICES.items():
    v["wav"] = os.path.expanduser(v["wav"])
    if v.get("llm_wav"):  # 腔调和音色分开取
        a = model.frontend.frontend_zero_shot("", TAG + v["llm_text"], os.path.expanduser(v["llm_wav"]), SR, "")
        b = model.frontend.frontend_zero_shot("", TAG, v["wav"], SR, "")
        model.frontend.spk2info[vid] = {**{k: a[k] for k in LLM_KEYS}, **{k: b[k] for k in FLOW_KEYS}}
    else:
        model.add_zero_shot_spk(TAG + v["text"], v["wav"], vid)
if DEFAULT not in VOICES:
    DEFAULT = next(iter(VOICES))
LOCK = threading.Lock()


CENA = "册那"
# 9/27 用户：「册那」是灵魂，两个字都要拖长。模型现场念每遍都不一样（10 遍里有被吞成 0.2 秒的），
# 整段放慢 1.7 倍他又嫌「像慢放，很奇怪」。最后他从 8 段对比里挑了一遍，cena_clip.py 把它切出来、每种声音各做一段
# （voices/cena_<声音>.wav，已调过音）。句子里的「册那」一律播这段，前后的话照常合成。WU_CENA_CLIP=off 关。
CLIPS = {}
_ME_CLIP = f"{H}/voices/cena_me.wav"


def _make_clip(vid, path):
    """新加的声音还没有「册那」片段：拿本人那段的语音 token，按这个声音的音色画一段（跟 cena_clip.py 同一个做法）。"""
    import uuid
    toks = model.frontend.frontend_zero_shot("", "", _ME_CLIP, SR, "")["flow_prompt_speech_token"]
    fl, v, u = model.frontend.spk2info[vid], VOICES[vid], str(uuid.uuid1())
    model.model.hift_cache_dict[u] = None
    x = model.model.token2wav(token=toks, prompt_token=fl["flow_prompt_speech_token"], prompt_feat=fl["prompt_speech_feat"],
                              embedding=fl["flow_embedding"], token_offset=0, uuid=u, finalize=True, speed=1.0)
    model.model.hift_cache_dict.pop(u, None)
    y = Voicing(SR, float(v.get("gain_db", 0)), **{**EQ_ARGS, **v.get("eq", {})})(x.squeeze(0).float().cpu().numpy())
    k = int(SR * 0.015)
    y[-k:] *= np.linspace(1, 0, k)
    sf.write(path, y, SR, subtype="PCM_16")
    log.info("「册那」片段按 %s 的音色新画了一段：%s", vid, path)


if os.environ.get("WU_CENA_CLIP", "on") != "off" and os.path.exists(_ME_CLIP):
    for _vid in VOICES:
        _p = f"{H}/voices/cena_{_vid}.wav"
        if not os.path.exists(_p):
            _make_clip(_vid, _p)
        _y, _sr = sf.read(_p, dtype="float32")
        if _sr == SR:
            CLIPS[_vid] = (np.clip(_y, -1, 1) * 32767).astype("<i2").tobytes()


def speak(text, vid):
    v = VOICES[vid]
    vo = Voicing(SR, float(v.get("gain_db", 0)), **{**EQ_ARGS, **v.get("eq", {})}) if EQ else None

    def synth(t):
        for out in model.inference_zero_shot(t, v.get("text", ""), v["wav"], zero_shot_spk_id=vid, stream=True):
            x = out["tts_speech"].squeeze(0).float().cpu().numpy()
            if vo is not None:
                x = vo(x)
            yield (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()

    clip = CLIPS.get(vid)
    if clip is None or CENA not in text:
        yield from synth(text)
        return
    for i, part in enumerate(text.split(CENA)):
        if i:
            yield clip
            part = part.lstrip("，,。！!？?～~、…—- ")
        if re.search(r"\w", part):
            yield from synth(part)


with LOCK:  # 每种声音预热一句，第一条真请求不吃冷启动
    for vid in VOICES:
        for _ in speak("侬好。", vid):
            pass
log.info("ready: %s voices=%s default=%s sr=%d eq=%s cena=%s load=%.1fs", os.path.basename(MODEL_DIR),
         ",".join(VOICES), DEFAULT, SR, EQ_ARGS if EQ else "off", ",".join(CLIPS) or "off", time.time() - t0)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "model": os.path.basename(MODEL_DIR), "sample_rate": SR, "busy": LOCK.locked(),
                             "default": DEFAULT, "voices": {k: v["name"] for k, v in VOICES.items()}})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/tts":
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            text = str(req.get("text", "")).strip()
            vid = str(req.get("voice") or DEFAULT)
        except Exception:
            return self._json(400, {"error": "bad json"})
        if not text:
            return self._json(400, {"error": "empty text"})
        if vid not in VOICES:
            vid = DEFAULT
        with LOCK:
            t1 = time.time()
            first = None
            n = 0
            self.send_response(200)
            self.send_header("Content-Type", "audio/L16")
            self.send_header("X-Sample-Rate", str(SR))
            self.send_header("X-Voice", vid)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for pcm in speak(text, vid):
                    if first is None:
                        first = time.time() - t1
                    n += len(pcm)
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(pcm), pcm))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                log.warning("client gone: %r", text[:30])
                return
            log.info("tts[%s] %.2fs first %.2fs audio %.1fs: %s", vid, time.time() - t1, first or -1, n / 2 / SR, text[:40])


ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
