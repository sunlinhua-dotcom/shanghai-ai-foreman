#!/bin/bash
# Builds the desktop eye simulator from the firmware's own eye engine.
#   ./build.sh            -> out/eyesim
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
board="$here/../../main/boards/xingzhi-cube-1.54tft-ml307"
mkdir -p "$here/out"
clang++ -std=c++17 -O2 -Wall -Wextra -Wshadow -I"$board" \
    "$here/eyesim.cc" "$board/eye_engine.cc" -o "$here/out/eyesim"
echo "$here/out/eyesim"
