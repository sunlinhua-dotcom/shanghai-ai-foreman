#!/usr/bin/env bash
# 把上游 xiaozhi-esp32 拉到固定提交，再把 overlay/ 里改过的文件盖上去。
# 用法：./setup.sh [目标目录]，默认 ./xiaozhi-esp32
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="https://github.com/78/xiaozhi-esp32.git"
COMMIT="$(tr -d '[:space:]' < "$HERE/UPSTREAM_COMMIT")"
DEST="${1:-$HERE/xiaozhi-esp32}"

if [ ! -d "$DEST/.git" ]; then
  git clone "$UPSTREAM" "$DEST"
elif [ -n "$(git -C "$DEST" status --porcelain --untracked-files=no)" ]; then
  echo "$DEST 里有没提交的改动。先处理掉（或者换个目标目录）再跑。" >&2
  exit 1
fi

git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$COMMIT"
cp -R "$HERE/overlay/." "$DEST/"

echo "好了：$DEST 已切到上游 ${COMMIT:0:7}，overlay 已盖上。下一步："
echo "  cd \"$DEST\""
echo "  idf.py set-target esp32s3 && idf.py menuconfig    # 填 OTA 地址和 Self-host"
echo "  python scripts/release.py xingzhi-cube-1.54tft-ml307"
