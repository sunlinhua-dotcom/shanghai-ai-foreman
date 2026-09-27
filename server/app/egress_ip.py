"""探测「本机的公网出口 IP」，给 OTA 的局域网直连判断用。

盒子在家连 Wi-Fi 时，OTA 请求经 Cloudflare 进来，`CF-Connecting-IP` 是家里宽带的
公网出口 IP —— 和服务端自己出网时用的是同一个 NAT 出口。两者相等 ⇒ 盒子和服务端
在同一个局域网里，可以直接回局域网 ws，省掉隧道那 ~0.6 s 往返。

出口 IP 会变（宽带重播），所以后台每 `refresh_seconds` 重新探一次；
`XZ_LAN_PUBLIC_IPS` 填了就完全以配置为准，不再探测。
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket

import aiohttp

log = logging.getLogger("xz.egress")

# 都是「把你的出口 IP 原样回给你」的端点，不带任何身份信息。
# v4 / v6 各探一遍：家里宽带同时有 IPv4 NAT 出口和 IPv6 前缀，盒子走哪条不一定。
_PROBES = (
    ("https://www.cloudflare.com/cdn-cgi/trace", "trace"),
    ("https://api.ipify.org", "plain"),
)


def _parse(kind: str, body: str) -> str:
    if kind == "trace":
        for line in body.splitlines():
            if line.startswith("ip="):
                return line[3:].strip()
        return ""
    return body.strip()


def is_private(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.is_private or addr.is_loopback or addr.is_link_local


class EgressIPWatcher:
    """后台维护一份「本机公网出口 IP」集合。"""

    def __init__(self, static_ips: tuple[str, ...] = (), refresh_seconds: float = 600.0):
        self.static_ips = {ip for ip in static_ips if ip}
        self.refresh_seconds = refresh_seconds
        self.ips: set[str] = set(self.static_ips)
        self._task: asyncio.Task | None = None

    async def probe_once(self) -> set[str]:
        if self.static_ips:  # 配置写死了就不出网探测
            return set(self.static_ips)
        found: set[str] = set()
        timeout = aiohttp.ClientTimeout(total=8)
        for family in (socket.AF_INET, socket.AF_INET6):
            # trust_env=False：这台机器可能 export 了代理，走代理探到的是代理的出口
            conn = aiohttp.TCPConnector(family=family, ssl=True)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=False, connector=conn
            ) as sess:
                for url, kind in _PROBES:
                    try:
                        async with sess.get(url) as resp:
                            if resp.status != 200:
                                continue
                            ip = _parse(kind, await resp.text())
                    except Exception as e:
                        log.debug("出口 IP 探测失败 %s (%s): %s", url, family, e)
                        continue
                    try:
                        ipaddress.ip_address(ip)
                    except ValueError:
                        continue
                    found.add(ip)
                    break  # 这个协议族探到了就够
        return found

    async def refresh(self) -> set[str]:
        found = await self.probe_once()
        if found:
            if found != self.ips:
                log.info("本机公网出口 IP：%s", ", ".join(sorted(found)))
            self.ips = found
        elif not self.ips:
            log.warning(
                "探不到本机公网出口 IP，局域网直连判断会一直不成立"
                "（可以用 XZ_LAN_PUBLIC_IPS 手工写死）"
            )
        return self.ips

    async def start(self) -> None:
        await self.refresh()
        if self.static_ips:
            return
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.refresh_seconds)
                await self.refresh()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # 探测出错不能把服务带下去
                log.warning("出口 IP 刷新出错: %s", e)

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
