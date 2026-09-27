"""doubao-key 在 EX10 上的那一半：stdin 一行 key → 试合成「你好」→ 能用就写进 .env（600）。

key 只走 stdin，不进命令行参数（ps 看不到）、不进日志；屏幕上只显示末 4 位。
只存 key，不切换 TTS：挑好音色后再改 XZ_TTS_PROVIDER=volc。
"""
import json
import os
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid

URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
VOICE = "zh_female_xiaohe_uranus_bigtts"  # 小何 2.0

key = "".join(sys.stdin.read().split())
if not key:
    print("    没收到 key，原来的不动。")
    sys.exit(1)


def check(key: str) -> tuple[bool, str]:
    """→ (能用, 说明)。说明里只有接口回的原因，没有 key。"""
    body = {
        "user": {"uid": "xiaozhi"},
        "req_params": {
            "text": "你好",
            "speaker": VOICE,
            "audio_params": {"format": "pcm", "sample_rate": 16000},
        },
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Api-Key": key,
            "X-Api-Resource-Id": "seed-tts-2.0",
            "X-Api-Request-Id": str(uuid.uuid4()),
        },
    )
    # 国内接口直连，不走代理环境变量
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=20) as r:
            audio = 0
            for line in r:
                line = line.strip()
                if not line:
                    continue
                msg = json.loads(line)
                code = msg.get("code", 0)
                if code == 20000000:
                    break
                if code != 0:
                    return False, f"{code} {str(msg.get('message', ''))[:200]}"
                audio += len(msg.get("data") or "")
            return (True, "") if audio else (False, "接口没返回音频")
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.read()[:200].decode('utf-8', 'replace')}"
    except Exception as e:  # noqa: BLE001
        return False, f"连不上豆包：{e}"


ok, why = check(key)
why = why.replace(key, f"…{key[-4:]}")  # 万一接口把 key 原样写进报错里
tag = f"key（…{key[-4:]}）"
if not ok:
    print(f"    {tag} 没验过：{why}")
    low = why.lower()
    if "permission" in low or "resource" in low or "45000000" in why:
        print("    多半是「豆包语音合成模型 2.0」还没开通：控制台概览页点「一键开通」或到「开通管理」开通，再敲一次 doubao-key。")
    elif "401" in why or "403" in why or "auth" in low or "key" in low:
        print("    多半是 key 复制漏了字符，重新复制整串再敲一次 doubao-key。")
    elif "quota" in low:
        print("    额度或并发满了，稍后再试。")
    print("    原来的 .env 没动。")
    sys.exit(1)
print(f"    {tag}：能用，试合成「你好」成功（扣了 2 个字）。")

env = pathlib.Path(".env")
bak = pathlib.Path("tmp") / f"env.bak-{time.strftime('%Y%m%d-%H%M%S')}"
bak.parent.mkdir(exist_ok=True)
shutil.copy2(env, bak)  # tmp/ 不在 deploy.sh 的 rsync --delete 范围里
lines = [ln for ln in env.read_text().splitlines()
         if not ln.startswith("XZ_VOLC_API_KEY=") and not ln.startswith("# 豆包语音（doubao-key 写入")]
lines += [f"# 豆包语音（doubao-key 写入，{time.strftime('%F %T')}）；XZ_TTS_PROVIDER=volc 时才用",
          "XZ_VOLC_API_KEY=" + key]
tmp = env.with_name(".env.new")
fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    f.write("\n".join(lines) + "\n")
os.replace(tmp, env)
os.chmod(env, 0o600)
print(f"    存进了 {env.resolve()}（原来的备份在 {bak}）。盒子的声音先不变。")
