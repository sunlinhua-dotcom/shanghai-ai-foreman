"""无线升级：换固件不用再插线、不用按 BOOT+复位。

盒子每次开机都会 POST /xiaozhi/ota/ 问一次版本；回包里带 firmware 段、而且版本比它新，
它就自己下载、写进另一个 ota 分区、重启（ota.cc CheckVersion → application.cc UpgradeFirmware）。
新固件起不来的话，因为它要联网查完版本才 mark valid，bootloader 会自动退回上一版
（sdkconfig 里 CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE=y）。

只在清单点名的设备、板型对得上、版本比它新时才发。清单放在部署目录外面
（deploy.sh 的 rsync --delete 碰不到）：

  <XZ_FW_DIR>/manifest.json
    {"version": "2.2.4.1", "file": "xiaozhi-2.2.4.1-ab12cd34.bin", "sha256": "…",
     "board": "xingzhi-cube-1.54tft-ml307", "devices": ["fc:01:2c:c5:bf:c0"]}

盒子只在开机时查版本，所以放了新固件以后，它下一次连上来（喊「你好小智」）时让它
重启一次去装；每个版本只催一次，装失败也不会来回重启（记在 reboot_sent.json）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("xz.fw")

# device_id → 它开机时报的固件版本 / 板型（只在内存里，服务重启就忘）
LAST_SEEN: dict[str, tuple[str, str]] = {}


def _parts(v: str) -> list[int]:
    """照抄固件 Ota::ParseVersion：按点切开逐段转数字。"""
    out = []
    for p in (v or "").split("."):
        try:
            out.append(int(p))
        except ValueError:
            out.append(0)
    return out


def is_newer(new: str, cur: str) -> bool:
    """照抄固件 Ota::IsNewVersionAvailable：逐段比，前面都相等时段数多的算新。"""
    a, b = _parts(cur), _parts(new)
    for x, y in zip(a, b):
        if y != x:
            return y > x
    return len(b) > len(a)


def load_manifest(cfg) -> dict | None:
    d = Path(getattr(cfg, "fw_dir", "") or "")
    if not str(d):
        return None
    try:
        m = json.loads((d / "manifest.json").read_text())
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001 - 清单坏了就当没有，绝不能拖垮 OTA
        log.warning("固件清单读不了：%s", e)
        return None
    if not all(m.get(k) for k in ("version", "file", "board", "devices")):
        log.warning("固件清单缺字段（version/file/board/devices），不推")
        return None
    if "/" in m["file"] or not (d / m["file"]).is_file():
        log.warning("固件清单里的文件不在 %s：%s", d, m["file"])
        return None
    return m


def _listed(m: dict, device_id: str) -> bool:
    return device_id.lower() in {x.lower() for x in m["devices"]}


def offer(cfg, device_id: str, board: str, version: str) -> dict | None:
    """开机查版本时：这台该不该升级。该的话返回清单。"""
    LAST_SEEN[device_id.lower()] = (version or "", board or "")
    m = load_manifest(cfg)
    if not m or not _listed(m, device_id):
        return None
    if board != m["board"]:
        log.warning("%s 报的板型 %r 和固件清单 %r 对不上，不推", device_id, board, m["board"])
        return None
    return m if is_newer(m["version"], version) else None


def pending_for(cfg, device_id: str) -> dict | None:
    """连上来时：有没有新固件等着它装（服务重启后不知道它的版本，按「可能没装」算）。"""
    m = load_manifest(cfg)
    if not m or not _listed(m, device_id):
        return None
    seen = LAST_SEEN.get(device_id.lower())
    if seen and not is_newer(m["version"], seen[0]):
        return None
    return m


def claim_reboot(cfg, device_id: str, version: str) -> bool:
    """每台、每个版本只催一次重启。返回 True = 这次该催。"""
    path = Path(cfg.fw_dir) / "reboot_sent.json"
    try:
        sent = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        sent = {}
    key = device_id.lower()
    if sent.get(key) == version:
        return False
    sent[key] = version
    try:
        path.write_text(json.dumps(sent, ensure_ascii=False, indent=1))
    except OSError as e:
        log.warning("记不下「已催重启」（%s），这次不催，免得来回重启", e)
        return False
    return True
