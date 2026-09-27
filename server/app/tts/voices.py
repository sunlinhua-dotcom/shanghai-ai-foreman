"""盒子用哪种声音：吴语配音服务预载了几种，盒子上音量加减一起按或讲「换只声音」轮换。

选中的记在 ~/xz-data/voice.json，重启不丢。id 要跟 GX10 上 voices.json 的一致。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("xz.voices")

# 轮换顺序和盒子念出来的名字。id 要跟配音服务的声音表（deploy/wu_voices.example.json）一致；
# 排第一的是默认（没选过 / 选的声音没了就用它）。
ORDER = ["me", "v2", "v3"]
NAMES = {"me": "侬自家个声音", "v2": "第二种声音", "v3": "第三种声音"}
_FILE = Path(os.path.expanduser(os.environ.get("XZ_VOICE_FILE", "~/xz-data/voice.json")))
_current: str | None = None


def current() -> str:
    global _current
    if _current is None:
        try:
            v = json.loads(_FILE.read_text(encoding="utf-8")).get("voice")
        except (OSError, ValueError):
            v = None
        _current = v if v in ORDER else ORDER[0]
    return _current


def cycle() -> str:
    """换下一种声音并记下来，返回新的 id。"""
    global _current
    _current = ORDER[(ORDER.index(current()) + 1) % len(ORDER)]
    try:
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        _FILE.write_text(json.dumps({"voice": _current}), encoding="utf-8")
    except OSError as e:
        log.warning("声音选择没存下来：%s", e)
    return _current


def name(v: str | None = None) -> str:
    return NAMES.get(v or current(), v or current())
