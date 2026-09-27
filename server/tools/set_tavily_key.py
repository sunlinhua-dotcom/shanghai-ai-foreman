"""tavily-key 在 EX10 上的那一半：stdin 一行一个 key → 逐个试查 → 能用的写进 .env（600）并切到 Tavily。

key 只走 stdin，不进命令行参数（ps 看不到）、不进日志；屏幕上只显示末 4 位。
写成 TAVILY_API_KEYS=k1,k2,…，服务端轮着用，某个额度满了自动换下一个，全不行再退回 SearXNG。
"""
import json
import os
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request

keys = list(dict.fromkeys(ln.strip() for ln in sys.stdin if ln.strip()))
if not keys:
    print("    没收到 key，原来的不动。")
    sys.exit(1)


def check(key: str) -> int:
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps({"query": "今天天气", "max_results": 1}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001
        return 0


keep = []
for i, key in enumerate(keys, 1):
    code = check(key)
    tag = f"第 {i} 个（…{key[-4:]}）"
    if code == 200:
        print(f"    {tag}：能用。")
        keep.append(key)
    elif code in (429, 432, 433):
        print(f"    {tag}：key 是对的，但这个月额度已满（HTTP {code}），先存着，重置后自动用上。")
        keep.append(key)
    elif code in (401, 403):
        print(f"    {tag}：key 不对（HTTP {code}），多半是复制漏了字符，这个不存。")
    else:
        print(f"    {tag}：没验成（{'连不上 Tavily' if code == 0 else f'HTTP {code}'}），这个不存，稍后再试。")
if not keep:
    print("    一个能用的都没有，原来的不动。")
    sys.exit(1)

env = pathlib.Path(".env")
shutil.copy2(env, env.with_name(f".env.bak.{time.strftime('%Y%m%d-%H%M%S')}"))
lines = [ln for ln in env.read_text().splitlines()
         if not ln.startswith(("TAVILY_API_KEY=", "TAVILY_API_KEYS=", "XZ_SEARCH_PROVIDER="))
         and not ln.startswith("# Tavily（tavily-key 写入")]
lines += [f"# Tavily（tavily-key 写入，{time.strftime('%F %T')}）；轮着用，全不行退回 SearXNG",
          "XZ_SEARCH_PROVIDER=tavily", "TAVILY_API_KEYS=" + ",".join(keep)]
tmp = env.with_name(".env.new")
fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    f.write("\n".join(lines) + "\n")
os.replace(tmp, env)
os.chmod(env, 0o600)
print(f"    存了 {len(keep)} 个 key 到 {env.resolve()}，搜索切到 Tavily。")
