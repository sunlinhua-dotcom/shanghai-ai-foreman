"""入口：OTA(HTTP) + WebSocket 同一个端口（只开一个 Tunnel 就够）。"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import logging.handlers
import sys
from pathlib import Path

from aiohttp import web

from .announce import AnnounceHub, rewrite_prompt
from .asr import create_asr
from .config import SERVER_DIR, ConfigError, load_config
from .egress_ip import EgressIPWatcher
from .feishu import FeishuBridge
from .llm import create_llm
from .ota import setup_ota_routes
from .persona import build_system_prompt, dialect_flavor, dialect_words
from .session import DeviceSession
from .tools import close_http, tavily_probe_loop
from .turn import load_smart_turn
from .voiceprint import VoicePrint
from .tts import create_tts

log = logging.getLogger("xz")


class Services:
    def __init__(self, cfg, asr, llm, tts, turn=None):
        self.cfg = cfg
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.turn = turn   # 断句模型（app/turn.py），None = 只看静音
        self.announce: AnnounceHub | None = None
        self.voiceprint: VoicePrint | None = None

    async def close(self):
        for obj in (self.asr, self.tts):
            close = getattr(obj, "close", None)
            if close:
                try:
                    await close()
                except Exception:
                    pass


def setup_logging(level: str = "INFO", to_file: bool = True) -> None:
    fmt = "%(asctime)s %(levelname)-5s %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if to_file:
        logs_dir = SERVER_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                logs_dir / "server.log", maxBytes=8 << 20, backupCount=5, encoding="utf-8"
            )
        )
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers, force=True)
    for noisy in ("aiohttp.access", "httpx", "httpcore", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------- WebSocket ----------

def _check_auth(request: web.Request, cfg) -> str | None:
    """返回 None = 通过；否则返回拒绝原因。token 为空则不校验。"""
    if not cfg.token:
        return None
    auth = request.headers.get("Authorization", "")
    expected = cfg.token if " " in cfg.token else f"Bearer {cfg.token}"
    if auth == expected or auth == cfg.token:
        return None
    return f"Authorization 不匹配（收到 {auth[:24]!r}）"


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    cfg = request.app["cfg"]
    services: Services = request.app["services"]

    device_id = request.headers.get("Device-Id", "")
    client_id = request.headers.get("Client-Id", "")
    proto_ver = request.headers.get("Protocol-Version", "")

    if not device_id:
        log.warning("WS 握手缺 Device-Id，来自 %s", request.remote)
        return web.Response(status=400, text="missing Device-Id")
    if not client_id:
        log.warning("WS 握手缺 Client-Id（device_id=%s），仍然放行", device_id)
    if cfg.allowed_device_ids and device_id.lower() not in cfg.allowed_device_ids:
        log.warning("Device-Id %s 不在 XZ_ALLOWED_DEVICE_IDS 白名单里，拒绝", device_id)
        return web.Response(status=403, text="device not allowed")

    reason = _check_auth(request, cfg)
    if reason:
        log.warning("WS 鉴权失败 device_id=%s: %s", device_id, reason)
        return web.Response(status=401, text="unauthorized")

    ws = web.WebSocketResponse(heartbeat=cfg.ws_heartbeat, max_msg_size=4 << 20)
    await ws.prepare(request)
    log.info(
        "WS 已连接 device_id=%s client_id=%s Protocol-Version=%s from=%s",
        device_id, client_id, proto_ver or "(未带)", request.remote,
    )

    session = DeviceSession(ws, device_id, client_id, services)
    request.app["sessions"].add(session)
    hub: AnnounceHub = request.app["announce"]
    await hub.session_opened(device_id)
    try:
        await session.run()
    finally:
        request.app["sessions"].discard(session)
        await hub.session_closed(device_id)
    return ws


# ---------- Claude Code 任务播报 ----------

def _client_ip(request: web.Request) -> str:
    return request.headers.get("CF-Connecting-IP") or request.remote or "?"


async def announce_handler(request: web.Request) -> web.Response:
    """POST /xiaozhi/announce  {"text": "...", "kind": "done", "project": "ESP32"}"""
    cfg = request.app["cfg"]
    hub: AnnounceHub = request.app["announce"]
    if not hub.enabled:
        return web.Response(status=404, text="not found")
    got = request.headers.get("Authorization", "").encode()
    if not hmac.compare_digest(got, f"Bearer {cfg.announce_token}".encode()):
        log.warning("播报鉴权失败，来自 %s", _client_ip(request))
        return web.Response(status=401, text="unauthorized")
    if (request.content_length or 0) > 8192:
        return web.Response(status=413, text="too large")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "bad json"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"ok": False, "error": "bad json"}, status=400)
    status, res = await hub.add(
        str(body.get("text") or ""),
        kind=str(body.get("kind") or "")[:20],
        project=str(body.get("project") or "")[:40],
        device=str(body.get("device") or "") or None,
    )
    return web.json_response(res, status=status)


async def inbox_handler(request: web.Request) -> web.Response:
    """GET /xiaozhi/inbox?wait=50  盒子待机时的长轮询：200 = 有播报，醒来连上来念。"""
    cfg = request.app["cfg"]
    hub: AnnounceHub = request.app["announce"]
    if not hub.enabled:
        # 没开播报：让盒子一小时来问一次就行
        return web.Response(status=204, headers={"X-Poll-After": "3600"})
    device_id = request.headers.get("Device-Id", "").strip().lower()
    if not device_id:
        return web.Response(status=400, text="missing Device-Id")
    if cfg.allowed_device_ids and device_id not in cfg.allowed_device_ids:
        log.warning("收件箱：Device-Id %s 不在白名单，拒绝（%s）", device_id, _client_ip(request))
        return web.Response(status=403, text="device not allowed", headers={"X-Poll-After": "3600"})
    reason = _check_auth(request, cfg)
    if reason:
        log.warning("收件箱鉴权失败 device_id=%s: %s", device_id, reason)
        return web.Response(status=401, text="unauthorized", headers={"X-Poll-After": "600"})
    try:
        wait = float(request.query.get("wait", "50"))
    except ValueError:
        wait = 50.0
    wait = max(0.0, min(wait, 55.0))   # Cloudflare 100 s 就掐，留足余量
    def alive() -> bool:
        return request.transport is not None and not request.transport.is_closing()

    status, after = await hub.poll(device_id, wait, alive)
    headers = {"X-Poll-After": str(after), "Cache-Control": "no-store"}
    if status == 200:
        return web.json_response({"pending": True}, headers=headers)
    return web.Response(status=status, headers=headers)


async def health_handler(request: web.Request) -> web.Response:
    cfg = request.app["cfg"]
    return web.json_response(
        {
            "ok": True,
            "service": "xiaozhi-selfhost",
            "sessions": len(request.app["sessions"]),
            "protocol_version": cfg.protocol_version,
            "downlink_sample_rate": cfg.downlink_sample_rate,
            "llm_model": cfg.llm_model,
            "asr": cfg.asr_provider,
            "asr_streaming": bool(getattr(request.app["services"].asr, "streaming", False)),
            "tts": cfg.tts_provider,
            "lan_ws": cfg.lan_ws_url or None,
        }
    )


async def root_handler(request: web.Request) -> web.Response:
    cfg = request.app["cfg"]
    return web.Response(
        text=(
            "小智自建服务端\n"
            f"OTA:  POST /xiaozhi/ota/\n"
            f"WS:   /xiaozhi/v1/\n"
            f"健康: /health\n"
            f"帧协议版本 v{cfg.protocol_version}，下行 {cfg.downlink_sample_rate} Hz\n"
        ),
        content_type="text/plain",
    )


async def on_startup(app: web.Application) -> None:
    cfg = app["cfg"]
    if cfg.lan_enabled:
        watcher = EgressIPWatcher(cfg.lan_public_ips)
        app["egress"] = watcher
        await watcher.start()
        log.info(
            "局域网直连已开：出口 IP 命中就回 %s（其余照旧回公网）；本机出口 %s",
            cfg.lan_ws_url, ", ".join(sorted(watcher.ips)) or "(未知)",
        )
    asr = app["services"].asr
    if getattr(asr, "warmup", None):
        asyncio.create_task(_warmup(asr))
    # Tavily 几个 key 这个月谁还有额度（/usage 不扣额度），每 6 小时复查
    app["tavily_probe"] = asyncio.create_task(tavily_probe_loop(cfg))
    if cfg.feishu_enabled:
        if not (cfg.feishu_app_id and cfg.feishu_app_secret):
            log.warning("XZ_FEISHU_ENABLED=1 但没配 app_id / app_secret，飞书入口不启动")
        else:
            bridge = FeishuBridge(cfg, asyncio.get_running_loop())
            app["feishu"] = bridge
            bridge.start()
            log.info(
                "飞书入口已开：在飞书里找这个机器人打字（或用输入框的语音转文字）就能聊；"
                "白名单 %s",
                ", ".join(cfg.feishu_allowed_open_ids) or "(未设，谁都能聊)",
            )
    try:
        tts = app["services"].tts
        health = await tts.health()
        log.info("TTS[%s] 可用：%s", getattr(tts, "name", "?"), health)
    except Exception as e:
        log.warning(
            "TTS[%s] 不可达：%s —— 语音输出会失败，先修 TTS（XZ_TTS_PROVIDER 可切回 voxtts）",
            cfg.tts_provider, e,
        )


async def _warmup(asr) -> None:
    try:
        await asr.warmup()
    except Exception as e:
        log.warning("ASR 预热失败: %s", e)


async def on_cleanup(app: web.Application) -> None:
    probe = app.get("tavily_probe")
    if probe is not None:
        probe.cancel()
    bridge = app.get("feishu")
    if bridge is not None:
        await bridge.stop()
    watcher = app.get("egress")
    if watcher is not None:
        await watcher.stop()
    await app["services"].close()
    await close_http()


def build_app(cfg) -> web.Application:
    system_prompt = build_system_prompt(cfg)
    log.info("system prompt（%d 字）:\n%s", len(system_prompt), system_prompt)
    services = Services(
        cfg=cfg,
        asr=create_asr(cfg),
        llm=create_llm(cfg, system_prompt),
        tts=create_tts(cfg),
        turn=load_smart_turn(cfg),
    )
    app = web.Application()
    app["cfg"] = cfg
    app["services"] = services
    app["sessions"] = set()
    app["announce"] = services.announce = AnnounceHub(cfg)
    if cfg.voice_dialect and cfg.announce_rewrite and hasattr(services.llm, "complete"):
        # 播报原文是书面普通话，豆包照着念常常串成普通话：先让大模型改写成方言口语
        prompt = rewrite_prompt(cfg.voice_dialect, dialect_words(cfg.voice_dialect), dialect_flavor(cfg.voice_dialect))
        services.announce.rewrite = lambda text: services.llm.complete(prompt, text, max_tokens=400)
        log.info("播报先改写成%s口语再念", cfg.voice_dialect)
    if cfg.voiceprint_model:
        try:
            services.voiceprint = VoicePrint(cfg.voiceprint_model, cfg.voiceprint_dir,
                                             cfg.voiceprint_mode, cfg.voiceprint_threshold)
            log.info("声纹：%s 模式，门槛 %.2f，%s", cfg.voiceprint_mode, cfg.voiceprint_threshold,
                     "已有声纹" if services.voiceprint.print_ is not None else "还没建声纹")
        except Exception as e:  # noqa: BLE001  声纹坏了不能拖垮对话
            log.warning("声纹没开起来：%s", e)
    setup_ota_routes(app)
    app.router.add_get("/xiaozhi/v1/", ws_handler)
    app.router.add_get("/xiaozhi/v1", ws_handler)
    app.router.add_post("/xiaozhi/announce", announce_handler)
    app.router.add_get("/xiaozhi/inbox", inbox_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/", root_handler)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="小智协议自建服务端")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--no-file-log", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(args.log_level, to_file=not args.no_file_log)
    try:
        cfg = load_config()
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2

    app = build_app(cfg)
    log.info(
        "监听 http://%s:%d  (OTA=/xiaozhi/ota/  WS=/xiaozhi/v1/)  token=%s",
        cfg.host, cfg.port, "已启用" if cfg.token else "未启用",
    )
    web.run_app(app, host=cfg.host, port=cfg.port, print=None, access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
