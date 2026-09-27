#!/usr/bin/env bash
# 小智自建服务端进程管理
#   ./run.sh start | stop | restart | status | fg | test | logs
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PY="$DIR/.venv/bin/python"
PID_FILE="$DIR/xiaozhi-server.pid"
LOG_DIR="$DIR/logs"
OUT_LOG="$LOG_DIR/stdout.log"

mkdir -p "$LOG_DIR" "$DIR/tmp"

if [ ! -x "$PY" ]; then
  echo "找不到 $PY —— 先建虚拟环境："
  echo "    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

port() { grep -E '^XZ_PORT=' .env 2>/dev/null | tail -1 | cut -d= -f2 | tr -d ' ' || true; }

running_pid() {
  [ -f "$PID_FILE" ] || return 1
  local pid; pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

case "${1:-status}" in
  start)
    if pid="$(running_pid)"; then echo "已经在跑了，pid=$pid"; exit 0; fi
    nohup "$PY" -u -m app.main >> "$OUT_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    # 冷启动要导入 numpy/anthropic/aiohttp，磁盘冷时能到 30 s，这里等健康检查而不是死等固定秒数
    P="$(port)"; P="${P:-8095}"
    for _ in $(seq 1 180); do
      if ! running_pid >/dev/null; then
        echo "启动失败，最后 30 行："; tail -n 30 "$OUT_LOG"; rm -f "$PID_FILE"; exit 1
      fi
      if curl -s --noproxy '*' -m 2 "http://127.0.0.1:$P/health" >/dev/null 2>&1; then break; fi
      sleep 1
    done
    if pid="$(running_pid)"; then
      echo "已启动 pid=$pid  端口=$P"
      echo "日志：$LOG_DIR/server.log  (stdout: $OUT_LOG)"
      curl -s --noproxy '*' -m 3 "http://127.0.0.1:$P/health" || echo "(健康检查还没起来，看日志)"
      echo
    else
      echo "启动失败，最后 30 行："; tail -n 30 "$OUT_LOG"; rm -f "$PID_FILE"; exit 1
    fi
    ;;
  stop)
    if pid="$(running_pid)"; then
      kill "$pid"
      for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.3; done
      kill -9 "$pid" 2>/dev/null || true
      rm -f "$PID_FILE"
      echo "已停止 pid=$pid"
    else
      rm -f "$PID_FILE"; echo "没在跑"
    fi
    ;;
  restart) "$0" stop; "$0" start ;;
  status)
    if pid="$(running_pid)"; then
      echo "运行中 pid=$pid 端口=$(port)"
      curl -s --noproxy '*' -m 3 "http://127.0.0.1:$(port)/health" || echo "(健康检查无响应)"
      echo
    else
      echo "未运行"; exit 1
    fi
    ;;
  fg)   exec "$PY" -m app.main --log-level "${2:-INFO}" ;;
  test) shift; exec "$PY" tools/fake_device.py "$@" ;;
  logs) tail -f "$LOG_DIR/server.log" ;;
  *) sed -n '2,4p' "$0"; exit 1 ;;
esac
