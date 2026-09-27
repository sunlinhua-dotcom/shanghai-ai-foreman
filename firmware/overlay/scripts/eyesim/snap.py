#!/usr/bin/env python3
"""Pulls the frames a `snap` / `snap lv` USB command printed out of a serial log,
checks them, and writes each one as raw RGB565 (.565) and PNG.

    snap.py LOG OUT_PREFIX [--sim]

With --sim, each canvas frame is also rendered by `eyesim still idle idle` at the
engine time and strip the box reported, and the two are compared pixel by pixel.
That only means something for a frame taken in `eye det` mode (plain eyes, no
randomness). Differences are in 8-bit units per channel.
"""
import base64
import os
import re
import struct
import subprocess
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = re.compile(r"#SNAP (canvas|screen) (\d+) (\d+) t=(\d+)\.(\d+) strip=(\d+),(\d+) crc=([0-9a-f]{8})")
LINE = re.compile(r"#S(\d+) ([A-Za-z0-9+/=]+)\s*$")
END = re.compile(r"#SNAP END (\d+)")


def parse(path):
    frames, cur = [], None
    with open(path, errors="replace") as f:
        for text in f:
            m = HEAD.search(text)
            if m:
                cur = {"kind": m[1], "w": int(m[2]), "h": int(m[3]),
                       "t": int(m[4]) + int(m[5]) / 1e6, "strip": (int(m[6]), int(m[7])),
                       "crc": int(m[8], 16), "lines": {}}
                continue
            if cur is None:
                continue
            m = LINE.search(text)
            if m:
                cur["lines"][int(m[1])] = m[2]
                continue
            m = END.search(text)
            if m:
                cur["count"] = int(m[1])
                frames.append(cur)
                cur = None
    return frames


def to_rgb888(raw, w, h):
    px = struct.unpack("<%dH" % (w * h), raw)
    out = bytearray(w * h * 3)
    for i, v in enumerate(px):
        r, g, b = v >> 11, (v >> 5) & 63, v & 31
        out[3 * i] = (r << 3) | (r >> 2)
        out[3 * i + 1] = (g << 2) | (g >> 4)
        out[3 * i + 2] = (b << 3) | (b >> 2)
    return bytes(out)


def write_png(path, rgb, w, h):
    rows = b"".join(b"\x00" + rgb[y * w * 3:(y + 1) * w * 3] for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def compare(a, b, w, h):
    diff = [abs(x - y) for x, y in zip(a, b)]
    lit = sum(1 for i in range(w * h) if max(diff[3 * i:3 * i + 3]) > 0)
    worst = max(diff) if diff else 0
    at = diff.index(worst) // 3 if worst else -1
    over2 = sum(1 for i in range(w * h) if max(diff[3 * i:3 * i + 3]) > 2)
    return lit, over2, worst, (at % w, at // w) if at >= 0 else None


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    log, prefix, sim = sys.argv[1], sys.argv[2], "--sim" in sys.argv[3:]
    frames = parse(log)
    if not frames:
        print("no complete #SNAP frame in", log)
        return 1
    rc = 0
    for k, fr in enumerate(frames):
        w, h = fr["w"], fr["h"]
        missing = [i for i in range(fr["count"]) if i not in fr["lines"]]
        raw = b"".join(base64.b64decode(fr["lines"][i]) for i in sorted(fr["lines"]))
        crc = zlib.crc32(raw)
        ok = not missing and len(raw) == w * h * 2 and fr["crc"] in (crc, crc ^ 0xFFFFFFFF)
        name = "%s-%d-%s" % (prefix, k, fr["kind"])
        print("%s: %dx%d t=%.6f strip=%d..%d lines=%d missing=%d bytes=%d crc %s" % (
            name, w, h, fr["t"], fr["strip"][0], fr["strip"][1], fr["count"], len(missing), len(raw),
            "ok" if ok else "MISMATCH"))
        if not ok:
            rc = 1
            continue
        with open(name + ".565", "wb") as f:
            f.write(raw)
        rgb = to_rgb888(raw, w, h)
        write_png(name + ".png", rgb, w, h)
        if sim and fr["kind"] == "canvas":
            sim_path = name + "-sim.565"
            subprocess.run([os.path.join(HERE, "out", "eyesim"), "still", "idle", "idle", str(fr["strip"][1]),
                            "%.6f" % fr["t"], sim_path, "101214"], check=True)
            with open(sim_path, "rb") as f:
                sim_rgb = to_rgb888(f.read(), w, h)
            write_png(name + "-sim.png", sim_rgb, w, h)
            lit, over2, worst, at = compare(rgb, sim_rgb, w, h)
            print("  vs eyesim still: %d px differ, %d by more than 2, worst %d at %s" % (lit, over2, worst, at))
    return rc


if __name__ == "__main__":
    sys.exit(main())
