"""GX10 上的 CosyVoice3 上海话配音服务（小智盒子夜间 / 豆包出错时用，9/26 起）。

部署：放在 GX10 ~/cosyvoice3/CosyVoice/ 下，systemd cosyvoice-sh（deploy/cosyvoice-sh.service）。
  POST /tts {"text": "…"} → 流式 PCM16 单声道（Transfer-Encoding: chunked），采样率见 X-Sample-Rate
  GET /health
模型：Fun-CosyVoice3-0.5B 的 RL 版 + fp16（9/26 实测 7 句平均 CER 0.017、RTF 0.64，都比 base 好）。
声音：COSY_PROMPT_WAV（你自己的参考录音，约 12 秒）；指令：请用上海话表达。
一次只念一句（锁）。数字改上海话说法在盒子服务器那头做（wu_numbers），这里原样念。
"""
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "third_party/Matcha-TTS")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


def _load(path, *a, **k):  # torchaudio 2.9+ 的 load 要 torchcodec，用 soundfile 顶上
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

MODEL_DIR = os.environ.get("COSY_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B-RL")
PROMPT_WAV = os.path.expanduser(os.environ.get("COSY_PROMPT_WAV", "~/cosyvoice3/my_voice_ref.wav"))
INSTR = os.environ.get("COSY_INSTR", "You are a helpful assistant. 请用上海话表达。<|endofprompt|>")
FP16 = os.environ.get("COSY_FP16", "1") == "1"
PORT = int(os.environ.get("COSY_PORT", "8097"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cosyvoice-sh")

t0 = time.time()
model = AutoModel(model_dir=MODEL_DIR, fp16=FP16)
model.add_zero_shot_spk(INSTR, PROMPT_WAV, "me")  # instruct2 走存档时，prompt_text 就是这句指令
SR = model.sample_rate
LOCK = threading.Lock()


def speak(text):
    for out in model.inference_instruct2(text, INSTR, PROMPT_WAV, zero_shot_spk_id="me", stream=True):
        x = out["tts_speech"].squeeze(0).float().clamp(-1, 1).cpu().numpy()
        yield (x * 32767).astype("<i2").tobytes()


with LOCK:  # 预热一句，第一条真请求不吃冷启动
    for _ in speak("侬好。"):
        pass
log.info("ready: %s fp16=%s sr=%d prompt=%s load=%.1fs", MODEL_DIR, FP16, SR, PROMPT_WAV, time.time() - t0)


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
            self._json(200, {"ok": True, "model": os.path.basename(MODEL_DIR), "fp16": FP16, "sample_rate": SR,
                             "prompt": os.path.basename(PROMPT_WAV), "busy": LOCK.locked()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/tts":
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            text = str(req.get("text", "")).strip()
        except Exception:
            return self._json(400, {"error": "bad json"})
        if not text:
            return self._json(400, {"error": "empty text"})
        with LOCK:
            t1 = time.time()
            first = None
            n = 0
            self.send_response(200)
            self.send_header("Content-Type", "audio/L16")
            self.send_header("X-Sample-Rate", str(SR))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for pcm in speak(text):
                    if first is None:
                        first = time.time() - t1
                    n += len(pcm)
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(pcm), pcm))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                log.warning("client gone: %r", text[:30])
                return
            log.info("tts %.2fs first %.2fs audio %.1fs: %s", time.time() - t1, first or -1, n / 2 / SR, text[:40])


ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
