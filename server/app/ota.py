"""OTA 端点：POST /xiaozhi/ota/

按 research §1.2 只回最小合法响应：
  - websocket.url / token / version
  - server_time.timestamp(ms) / timezone_offset(min)
**firmware 段只在 app/firmware.py 的清单点名这台、而且版本更新时才回** → 盒子自己升级
**不回 activation 段** → 永不进激活流程（ota.cc:124-144, application.cc:444-469）
"""
from __future__ import annotations

import logging
import time
from urllib.parse import urlsplit, urlunsplit

from aiohttp import web

import ipaddress

from .egress_ip import is_private
from . import firmware

log = logging.getLogger("xz.ota")

TZ_OFFSET_MINUTES = 480  # 东八区

# 固件 2.2.4.2 起开机上报 reset_reason（esp_reset_reason_t）
_RESET = {0: "未知", 1: "上电", 2: "复位脚", 3: "软件重启", 4: "崩溃", 5: "中断看门狗", 6: "任务看门狗",
          7: "看门狗", 8: "深睡唤醒", 9: "欠压", 10: "SDIO", 11: "USB 复位", 12: "JTAG 复位", 13: "eFuse",
          14: "电源毛刺", 15: "CPU 锁死"}
_RESET_BAD = {4, 5, 6, 7, 9, 14, 15}


def _client_ip(request: web.Request) -> str:
    """设备真实来源 IP。经 Cloudflare 时以 CF-Connecting-IP 为准（cloudflared 自己加的）。"""
    cf = (request.headers.get("CF-Connecting-IP") or "").strip()
    if cf:
        return cf
    xff = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    return xff or (request.remote or "")


def _same_egress(ip: str, egress_ips: set[str], v6_prefix: int) -> str:
    """来源 IP 和本机出口算不算「同一个家」？算就返回命中的那个出口地址。

    IPv4 是整个家共用一个 NAT 出口，逐字相等即可；IPv6 每台机器有自己的全局地址，
    只有前缀（默认 /64）是一家人共用的，所以按前缀比。
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ""
    for raw in egress_ips:
        try:
            mine = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if addr.version != mine.version:
            continue
        if addr.version == 4:
            if addr == mine:
                return raw
        elif ipaddress.ip_network(f"{mine}/{v6_prefix}", strict=False).supernet_of(
            ipaddress.ip_network(f"{addr}/{v6_prefix}", strict=False)
        ):
            return raw
    return ""


def lan_decision(request: web.Request, cfg, egress_ips: set[str] | None) -> tuple[bool, str]:
    """要不要回局域网 ws？返回 (是否局域网, 判断依据)。

    两种算「在家」：
      1. 经 Cloudflare 进来，CF-Connecting-IP 和本机公网出口是同一家
         （IPv4 逐字相等 = 同一个 NAT 出口；IPv6 比 /XZ_LAN_IPV6_PREFIX 前缀）；
      2. 直连进来、源地址就是私网（盒子已经在用局域网地址了）。
    """
    if not cfg.lan_enabled:
        return False, "关闭"
    ip = _client_ip(request)
    if not ip:
        return False, "拿不到来源 IP"
    # 先比出口：写进 XZ_LAN_PUBLIC_IPS 的地址不管落在哪个段都算数
    hit = _same_egress(ip, egress_ips or set(), cfg.lan_ipv6_prefix)
    if hit:
        return True, f"来源 {ip} 与本机出口 {hit} 同一出口"
    if is_private(ip):
        if cfg.lan_trust_private:
            return True, f"来源 {ip} 是私网地址"
        return False, f"来源 {ip} 是私网但 XZ_LAN_TRUST_PRIVATE=0"
    return False, f"来源 {ip} 不是本机出口{'' if egress_ips else '（出口 IP 未知）'}"


def _derive_ws_url(request: web.Request, cfg) -> str:
    if cfg.public_ws_url:
        return cfg.public_ws_url
    # 经阿里云上海中转（frp）转进来的请求没有 CF-Connecting-IP：语音也回中转地址。
    # 盒子 OTA 走哪条路来，语音就走哪条路，中转挂了退回 Cloudflare 时两边一致
    if cfg.relay_ws_url and not request.headers.get("CF-Connecting-IP"):
        return cfg.relay_ws_url
    host = request.headers.get("X-Forwarded-Host") or request.headers.get("Host")
    if not host:
        host = f"{cfg.host if cfg.host != '0.0.0.0' else '127.0.0.1'}:{cfg.port}"
    proto = (request.headers.get("X-Forwarded-Proto") or request.scheme or "http").lower()
    scheme = "wss" if proto in ("https", "wss") else "ws"
    return urlunsplit((scheme, host, "/xiaozhi/v1/", "", ""))


def build_ota_response(request: web.Request, cfg, egress_ips: set[str] | None = None) -> dict:
    lan, why = lan_decision(request, cfg, egress_ips)
    ws_url = cfg.lan_ws_url if lan else _derive_ws_url(request, cfg)
    websocket: dict = {"url": ws_url, "version": cfg.protocol_version}
    if cfg.token:
        websocket["token"] = cfg.token
    else:
        # 设备把 token 原样填进 Authorization 头；给个空串也无妨，但更干净的是不带。
        websocket["token"] = ""
    return {
        "server_time": {
            "timestamp": int(time.time() * 1000),
            "timezone_offset": TZ_OFFSET_MINUTES,
        },
        "websocket": websocket,
    }


async def handle_ota(request: web.Request) -> web.Response:
    cfg = request.app["cfg"]
    device_id = request.headers.get("Device-Id", "")
    client_id = request.headers.get("Client-Id", "")
    ua = request.headers.get("User-Agent", "")

    body_summary = ""
    board: dict = {}
    app_info: dict = {}
    if request.method == "POST":
        try:
            body = await request.json()
            board = body.get("board") or {}
            app_info = body.get("application") or {}
            body_summary = (
                f"app={app_info.get('name')}/{app_info.get('version')} "
                f"board={board.get('type')}/{board.get('name')} "
                f"mac={body.get('mac_address')}"
            )
            rr = body.get("reset_reason")
            if isinstance(rr, int):
                body_summary += f" 上次重启={_RESET.get(rr, rr)}"
                if rr in _RESET_BAD:
                    log.warning("%s 上次是异常重启：%s（%d）", device_id, _RESET.get(rr, rr), rr)
        except Exception:
            body_summary = "(body 不是 JSON)"

    watcher = request.app.get("egress")
    egress_ips = watcher.ips if watcher is not None else None
    resp = build_ota_response(request, cfg, egress_ips)
    lan, why = lan_decision(request, cfg, egress_ips)
    if request.method == "POST" and device_id:
        fw = firmware.offer(cfg, device_id, str(board.get("type") or ""), str(app_info.get("version") or ""))
        if fw:
            # 清单里可以直接给下载地址（局域网直链）：经 Cloudflare 盒子只有 ~16 KB/s，
            # 一次 7 分钟，9/23 两次都在 80% 左右断掉；局域网几秒下完
            resp["firmware"] = {"version": fw["version"], "url": fw.get("url") or _fw_url(request, cfg, fw["file"])}
            log.warning("OTA 给 %s 推固件 %s → %s（盒子会自己下载、写入、重启）",
                        device_id, app_info.get("version"), fw["version"])
    log.info(
        "OTA %s device_id=%s client_id=%s ua=%r %s → ws=%s version=%s%s",
        request.method,
        device_id,
        client_id,
        ua,
        body_summary,
        resp["websocket"]["url"],
        resp["websocket"]["version"],
        f"（局域网直连：{why}）" if cfg.lan_enabled else "",
    )
    return web.json_response(resp)


def _fw_url(request: web.Request, cfg, name: str) -> str:
    """固件下载地址：跟盒子查 OTA 走同一个入口、同一种协议。4G 模块的 TLS 跟 Cloudflare 握不上手，
    走 4G 时盒子用明文 http 来查（9/23 起），下载也得给 http；明文进 Cloudflare 不会被 301（9/23 实测）。"""
    host = request.headers.get("X-Forwarded-Host") or request.host or ""
    scheme = (request.headers.get("X-Forwarded-Proto") or request.scheme or "http").lower()
    return f"{scheme}://{host}/xiaozhi/fw/{name}"


async def handle_fw(request: web.Request) -> web.StreamResponse:
    """只放清单里那一个文件；文件名带内容哈希，不列目录。"""
    cfg = request.app["cfg"]
    name = request.match_info.get("name", "")
    m = firmware.load_manifest(cfg)
    if not m or name != m["file"]:
        raise web.HTTPNotFound()
    log.warning("固件下载 %s ← %s", name, _client_ip(request))
    from pathlib import Path
    return web.FileResponse(Path(cfg.fw_dir) / name, headers={
        "Content-Type": "application/octet-stream", "Cache-Control": "no-store"})


def setup_ota_routes(app: web.Application) -> None:
    app.router.add_get("/xiaozhi/fw/{name}", handle_fw)
    for path in ("/xiaozhi/ota/", "/xiaozhi/ota"):
        app.router.add_route("POST", path, handle_ota)
        # GET 只是给人 curl 自检用；设备永远是 POST（ota.cc:93-94）
        app.router.add_route("GET", path, handle_ota)
