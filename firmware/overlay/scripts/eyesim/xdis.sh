#!/bin/bash
# eye_engine.cc compiled alone for the ESP32-S3, with the firmware build's own
# flags, and disassembled. Needs a firmware build (build/compile_commands.json).
#   scripts/eyesim/xdis.sh [OUT_DIR]           -> OUT_DIR/eye_engine.dis
#   scripts/eyesim/loops.py OUT_DIR/eye_engine.dis 'LitRow<false, true>'
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${1:-$HERE/out}
mkdir -p "$OUT"
source ~/esp/esp-idf-v5.5.2/export.sh > /dev/null 2>&1
python3 - "$ROOT" "$OUT" <<'PY'
import json, shlex, subprocess, sys
root, out = sys.argv[1], sys.argv[2]
for e in json.load(open(f"{root}/build/compile_commands.json")):
    if e["file"].endswith("/eye_engine.cc"):
        args, keep = shlex.split(e["command"]), []
        skip = 0
        for a in args:
            if skip: skip -= 1; continue
            if a in ("-MT", "-MF"): skip = 1; continue   # leave the build's dependency files alone
            if a == "-MD": continue
            keep.append(a)
        keep[keep.index("-o") + 1] = f"{out}/eye_engine.obj"
        subprocess.run(keep, cwd=e["directory"], check=True)
        break
else:
    sys.exit("eye_engine.cc is not in build/compile_commands.json: build the firmware first")
PY
xtensa-esp32s3-elf-objdump -d -C "$OUT/eye_engine.obj" > "$OUT/eye_engine.dis"
echo "$OUT/eye_engine.dis"
