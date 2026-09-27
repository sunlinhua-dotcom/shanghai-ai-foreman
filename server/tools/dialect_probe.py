"""豆包上海话「串成普通话」探针：同一批句子换几种写法让豆包念，再用只会普通话的 sherpa 听写打分。

判据：sherpa（中英双语 zipformer）只懂普通话。念成普通话它几乎全对，错字率很低；
真念成上海话它大面积听错，错字率很高。所以「普通话听写错字率」越低，越像普通话。
GX10 的 Qwen3-ASR 会方言，转写只做参考，看豆包实际念了什么。

在 EX10 上跑（要 .env 里的豆包 key）：
  cd /srv/xiaozhi-server && .venv/bin/python tools/dialect_probe.py --variants v0,none --reps 1
每段音频存 --out 目录（wav），结果另存 results.json。每句都真扣豆包字数，别开太大。
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import sys
import time
import uuid
import wave
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.asr.sherpa_onnx_asr import SherpaOnnxASR  # noqa: E402
from app.tts.volc import VolcTTS  # noqa: E402
from app.wu_text import to_wu  # noqa: E402

SR = 16000

# (编号, 原文, 上海话改写)。改写为 None = 原文本来就是上海话写法。
SENTENCES = [
    # 播报：ClaudeVoice 发来的书面普通话，夹英文、数字（9/25 日志原句）
    ("a1", "老孙，目录 ESP 盒子 的任务完成了。", "老孙，ESP 盒子个任务做好了。"),
    ("a2", "具体是，strix 重启完成，虚拟内存设置已核对。", "具体讲，strix 重新开好了，虚拟内存个设置也对过了。"),
    ("a3", "请授权修改Cloudflare DNS。", "麻烦侬授权一下，改 Cloudflare 个 DNS。"),
    ("a4", "有2条提醒。", "有两条提醒。"),
    ("a5", "海外访问速度大幅提升，国内访问速度略有下降。", "海外访问快交关了，国内稍微慢了一眼眼。"),
    ("a6", "需要用户确认是否启用此功能。", "要侬确认一下，这个功能开伐？"),
    ("a7", "具体是，播报完成后直接回待机，不再进入聆听。", "具体讲，播报念好就直接回待机，覅再进聆听了。"),
    ("a8", "请确认播报内容是否完整。", "侬听听看，播报有没有念全。"),
    ("a9", "建议购买首尔的轻量服务器。", "我看还是买首尔个轻量服务器好。"),
    ("a10", "购买后，在网页终端运行指定命令。", "买好以后，到网页终端里去运行指定个命令。"),
    # 模型回话：已经按上海话写（9/25 日志原句）
    ("r1", "侬是爬山到现在还是已经到屋里了？", None),
    ("r2", "玉龙雪山海拔是四千六百米左右，勿是一千二。", None),
    ("r3", "看侬兴奋个样子，是到海边栈道了，平路好走了。", None),
    # 固定句（claude.py::DIALECT_LINES）
    ("f1", "我帮侬看一看。", None),
    ("f2", "网络有点卡，没连上，侬再讲一遍。", None),
    ("f3", "好了。", None),
    # 模型偶尔写回普通话的样子
    ("m1", "今天上海多云，最高二十八度，出门不用带伞。", "今朝上海多云，最高二十八度，出去覅带伞。"),
    ("m2", "我不太清楚，你可以再说一遍吗？", "我不大清楚，侬再讲一遍好伐？"),
]

# 变体：语音指令 + 用原文还是改写
VARIANTS = {
    "none": {"instruction": "", "rewrite": False},               # 对照组：不给指令 = 普通话
    "v0": {"instruction": "用上海话说", "rewrite": False},        # 线上现状
    "v1": {"instruction": "用地道的上海话讲，每个字都用上海话的读音，不要讲普通话。", "rewrite": False},
    "v2": {"instruction": "侬用上海闲话讲，一句普通话也覅讲。", "rewrite": False},
    "rw": {"instruction": "用上海话说", "rewrite": True},         # 线上指令 + 上海话改写
    "rw1": {"instruction": "用地道的上海话讲，每个字都用上海话的读音，不要讲普通话。", "rewrite": True},
    "rule1": {"instruction": "用地道的上海话讲，每个字都用上海话的读音，不要讲普通话。", "rewrite": "rule"},
}

# 不给指令时（纯普通话）9/25 实测 0.00–0.26：英文词、侬/个这类字拉高了一点
MANDARIN_CER = 0.3

_KEEP = re.compile(r"[\u4e00-\u9fff0-9]")  # 只比汉字和数字：英文词两种念法一样


def _norm(s: str) -> str:
    return "".join(_KEEP.findall(s.lower()))


def cer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(re.sub(r"\bSIL\b", "", hyp))  # sherpa 把停顿写成 SIL
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def _env(path: Path) -> dict:
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class _SectionTTS(VolcTTS):
    """带 section_id 的豆包：同一会话的几句共享上下文。"""

    def __init__(self, *a, section_id: str = "", **kw):
        super().__init__(*a, **kw)
        self.section_id = section_id

    def _body(self, text: str) -> dict:
        body = super()._body(text)
        if self.section_id:
            add = json.loads(body["req_params"]["additions"])
            add["section_id"] = self.section_id
            body["req_params"]["additions"] = json.dumps(add, ensure_ascii=False)
        return body


def _wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()


async def qwen_asr(sess: aiohttp.ClientSession, url: str, pcm: bytes) -> str:
    form = aiohttp.FormData()
    form.add_field("file", _wav(pcm), filename="u.wav", content_type="audio/wav")
    form.add_field("model", "qwen3-asr")
    try:
        async with sess.post(url, data=form) as r:
            r.raise_for_status()
            return ((await r.json(content_type=None)).get("text") or "").strip()
    except Exception as e:  # noqa: BLE001
        return f"<失败 {type(e).__name__}>"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="none,v0")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--ids", default="", help="只跑这些编号，逗号分隔；空 = 全部")
    ap.add_argument("--texts", default="", help="不用内置句子，改念这几句（竖线分隔，编号 t1、t2…）")
    ap.add_argument("--section", action="store_true", help="每个变体每轮共用一个 section_id（按句子顺序念）")
    ap.add_argument("--voice", default="")
    ap.add_argument("--out", default="/tmp/dialect_probe")
    ap.add_argument("--no-qwen", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    env = _env(root / ".env")
    key = env.get("XZ_VOLC_API_KEY", "")
    if not key:
        sys.exit(".env 里没有 XZ_VOLC_API_KEY")
    voice = args.voice or env.get("XZ_TTS_VOICE", "")
    resource = env.get("XZ_VOLC_RESOURCE_ID", "seed-tts-2.0")
    asr = SherpaOnnxASR(env.get("XZ_ASR_MODEL_DIR", ""), num_threads=4)
    await asr.warmup()
    qurl = env.get("XZ_ASR_HTTP_URL", "http://127.0.0.1:8091/v1/audio/transcriptions")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ids = {x for x in args.ids.split(",") if x}
    sents = [s for s in SENTENCES if not ids or s[0] in ids]
    if args.texts:
        sents = [(f"t{i}", t, None) for i, t in enumerate(args.texts.split("|"), 1)]
    results = []
    chars = 0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as qs:
        for vname in args.variants.split(","):
            v = VARIANTS[vname]
            for rep in range(args.reps):
                tts = _SectionTTS(key, voice=voice, resource_id=resource, instruction=v["instruction"],
                                  sample_rate=SR, section_id=str(uuid.uuid4()) if args.section else "")

                async def one(sid: str, orig: str, rw: str | None):
                    if v["rewrite"] == "rule":
                        text = to_wu(orig)
                    else:
                        text = (rw or orig) if v["rewrite"] else orig
                    t0 = time.monotonic()
                    pcm = b"".join([c async for c in tts.synth_stream(text)])
                    synth_s = time.monotonic() - t0
                    (out / f"{vname}_{rep}_{sid}.wav").write_bytes(_wav(pcm))
                    mand = await asr.transcribe(pcm, SR)
                    q = "" if args.no_qwen else await qwen_asr(qs, qurl, pcm)
                    return {"variant": vname, "rep": rep, "id": sid, "text": text,
                            "audio_s": round(len(pcm) / 2 / SR, 2), "synth_s": round(synth_s, 2),
                            "sherpa": mand, "cer_mandarin": round(cer(text, mand), 3),
                            "qwen": q, "cer_qwen": round(cer(text, q), 3) if q else None}

                if args.section:  # 共享上下文就得按顺序念
                    rows = [await one(*s) for s in sents]
                else:
                    rows = await asyncio.gather(*(one(*s) for s in sents))
                await tts.close()
                for r in rows:
                    chars += len(r["text"])
                    results.append(r)
                    print(f"{r['variant']:>4} {r['rep']} {r['id']:<3} 普通话听写错字率 {r['cer_mandarin']:.2f}"
                          f"  sherpa={r['sherpa']!r}  qwen={r['qwen']!r}", flush=True)

    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n共合成 {len(results)} 段，约 {chars} 字；音频在 {out}")
    by = {}
    for r in results:
        by.setdefault((r["variant"], r["id"][0]), []).append(r["cer_mandarin"])
    print(f"\n变体 × 句型 平均普通话听写错字率（越低越像普通话；低于 {MANDARIN_CER} 记作念成普通话）：")
    for (vn, kind), xs in sorted(by.items()):
        n_m = sum(x < MANDARIN_CER for x in xs)
        print(f"  {vn:>4} {kind}  {sum(xs) / len(xs):.2f}  念成普通话 {n_m}/{len(xs)}")


if __name__ == "__main__":
    asyncio.run(main())
