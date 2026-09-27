// Desktop harness for the eye engine. It compiles the firmware's
// main/boards/xingzhi-cube-1.54tft-ml307/eye_engine.cc unchanged.
//
//   eyesim seq OUT.rgb [exact]    the mock-up's expression sequence, deterministic,
//                                 RGB888; compare with `node ref.mjs`
//   eyesim cmp A.rgb B.rgb        per-segment differences between two seq renders
//   eyesim live OUT.rgb [SEED]    device-like demo: RGB565 with dither, random
//                                 glances and blinks, the ring, layout changes
//   eyesim bench [SEED]           the live demo again, timing every drawn frame
//   eyesim still SHAPE PALETTE Y1 SEC OUT.565 [BG_RGB888]
//                                 the frame at SEC, deterministic, strip 26..Y1,
//                                 raw native RGB565 (to hold against a snap;
//                                 the device's ground is 101214)
//
// Frames are raw 240x240 RGB888, back to back, 60 fps.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "eye_engine.h"

namespace {

using eyes::Engine;
using eyes::Palette;
using eyes::Ring;
using eyes::Shape;

constexpr int kW = 240, kH = 240;
constexpr double kFps = 60;
constexpr int kTop = 26;   // the device's top bar

struct Seg {
    const char* name;
    Shape shape;
    Palette palette;
    double sec;
};
// Same order and timing as ref.mjs.
const Seg kSeq[] = {
    {"idle", Shape::kIdle, Palette::kIdle, 3.4},    {"happy", Shape::kHappy, Palette::kHappy, 2.2},
    {"love", Shape::kLove, Palette::kLove, 2.4},    {"star", Shape::kStar, Palette::kStar, 2.2},
    {"think", Shape::kThink, Palette::kThink, 2.4}, {"done", Shape::kDone, Palette::kDone, 2.4},
    {"fail", Shape::kFail, Palette::kFail, 2.2},    {"wide", Shape::kWide, Palette::kWide, 1.6},
    {"idle", Shape::kIdle, Palette::kIdle, 2.0},
};
constexpr int kSeqN = sizeof(kSeq) / sizeof(kSeq[0]);

// Segment shown in each frame; frame i (from 1) is at i/60 s, as in ref.mjs.
std::vector<int> SeqSegments() {
    double total = 0;
    for (const Seg& s : kSeq) total += s.sec;
    const int frames = static_cast<int>(std::lround(total * kFps));
    std::vector<int> out;
    int seg = 0;
    double next_ms = kSeq[0].sec * 1000;
    for (int i = 1; i <= frames; i++) {
        const double t_ms = i * 1000.0 / kFps;
        if (seg + 1 < kSeqN && t_ms >= next_ms - 1e-6) {
            seg++;
            next_ms += kSeq[seg].sec * 1000;
        }
        out.push_back(seg);
    }
    return out;
}

bool WriteAll(FILE* f, const void* p, size_t n) { return fwrite(p, 1, n, f) == n; }

int RunSeq(const char* path, bool exact) {
    std::vector<uint8_t> canvas(kW * kH * 3, 0);
    std::vector<float> work(Engine::WorkFloats());
    Engine e;
    e.Init(canvas.data(), eyes::PixelFormat::kRgb888, kW, kH, kW, work.data(), work.size(), 1);
    e.SetDeterministic(true);
    e.SetReference(true);
    e.SetExact(exact);
    FILE* f = fopen(path, "wb");
    if (f == nullptr) {
        perror(path);
        return 1;
    }
    const std::vector<int> segs = SeqSegments();
    int cur = 0;
    eyes::Rect rects[Engine::kMaxRects];
    uint64_t evals = 0;
    uint32_t evals_max = 0;
    for (int seg : segs) {
        if (seg != cur) {
            cur = seg;
            e.SetShape(kSeq[cur].shape);
            e.SetPalette(kSeq[cur].palette);
        }
        e.Advance(1 / kFps);
        e.Draw(rects, Engine::kMaxRects);
        evals += e.sdf_evals();
        evals_max = std::max(evals_max, e.sdf_evals());
        if (!WriteAll(f, canvas.data(), canvas.size())) {
            perror(path);
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    fprintf(stderr, "%zu frames -> %s; sdf evals per frame avg %.0f, max %u\n", segs.size(), path,
            static_cast<double>(evals) / segs.size(), evals_max);
    return 0;
}

std::vector<uint8_t> ReadFile(const char* path) {
    std::vector<uint8_t> out;
    FILE* f = fopen(path, "rb");
    if (f == nullptr) {
        perror(path);
        return out;
    }
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.insert(out.end(), buf, buf + n);
    fclose(f);
    return out;
}

// Differences per segment over lit pixels (either side above 2 in any channel).
int RunCmp(const char* pa, const char* pb) {
    const std::vector<uint8_t> a = ReadFile(pa), b = ReadFile(pb);
    const size_t fs = kW * kH * 3;
    if (a.empty() || b.empty()) return 1;
    if (a.size() != b.size()) fprintf(stderr, "sizes differ: %zu vs %zu bytes\n", a.size(), b.size());
    const size_t frames = std::min(a.size(), b.size()) / fs;
    const std::vector<int> segs = SeqSegments();
    struct Acc {
        uint64_t hist[256] = {};
        double sum = 0;
        uint64_t n = 0;
        int max = 0, max_frame = -1, frames = 0;
    };
    std::vector<Acc> acc(kSeqN + 1);
    for (size_t fi = 0; fi < frames; fi++) {
        const int s = fi < segs.size() ? segs[fi] : 0;
        Acc& A = acc[s];
        Acc& T = acc[kSeqN];
        A.frames++;
        T.frames++;
        const uint8_t* x = &a[fi * fs];
        const uint8_t* y = &b[fi * fs];
        for (size_t p = 0; p < fs; p += 3) {
            const bool lit = x[p] > 2 || x[p + 1] > 2 || x[p + 2] > 2 || y[p] > 2 || y[p + 1] > 2 || y[p + 2] > 2;
            if (!lit) continue;
            for (int c = 0; c < 3; c++) {
                const int d = std::abs(static_cast<int>(x[p + c]) - static_cast<int>(y[p + c]));
                for (Acc* q : {&A, &T}) {
                    q->hist[d]++;
                    q->sum += d;
                    q->n++;
                    if (d > q->max) {
                        q->max = d;
                        q->max_frame = static_cast<int>(fi) + 1;
                    }
                }
            }
        }
    }
    auto pct = [](const Acc& q, double p) {
        const uint64_t want = static_cast<uint64_t>(std::ceil(q.n * p));
        uint64_t run = 0;
        for (int i = 0; i < 256; i++) {
            run += q.hist[i];
            if (run >= want) return i;
        }
        return 255;
    };
    printf("%-6s %6s %8s %5s %5s %5s %7s %s\n", "seg", "frames", "mae", "p99", "p999", "max", ">8 %", "worst frame");
    for (int s = 0; s <= kSeqN; s++) {
        const Acc& q = acc[s];
        if (q.frames == 0 || q.n == 0) continue;
        uint64_t over8 = 0;
        for (int i = 9; i < 256; i++) over8 += q.hist[i];
        printf("%-6s %6d %8.4f %5d %5d %5d %7.4f %d (%.3f s)\n", s == kSeqN ? "ALL" : kSeq[s].name, q.frames,
               q.sum / q.n, pct(q, .99), pct(q, .999), q.max, 100.0 * over8 / q.n, q.max_frame,
               q.max_frame / kFps);
    }
    return 0;
}

// ---- the live demo ------------------------------------------------------------------------

struct Cue {
    double t = 0;
    int shape = -1, palette = -1, ring = -1, voice = -1, y1 = -1;
    bool set_lids = false, set_gaze = false, wander = true;
    eyes::LidPose lids;
    float gx = 0, gy = 0;
    Cue& S(Shape s) { shape = static_cast<int>(s); return *this; }
    Cue& P(Palette p) { palette = static_cast<int>(p); return *this; }
    Cue& R(Ring r) { ring = static_cast<int>(r); return *this; }
    Cue& V(bool on) { voice = on ? 1 : 0; return *this; }
    Cue& Strip(int bottom) { y1 = bottom; return *this; }
    Cue& Lids(float o, float i, float b) { set_lids = true; lids.outer = o; lids.inner = i; lids.bottom = b; return *this; }
    Cue& Look(float x, float y) { set_gaze = true; wander = false; gx = x; gy = y; return *this; }
    Cue& Wander() { set_gaze = true; wander = true; gx = gy = 0; return *this; }
};
Cue At(double t) {
    Cue c;
    c.t = t;
    return c;
}

constexpr double kLiveSec = 34;
const std::vector<Cue>& LiveCues() {
    static const std::vector<Cue> cues = {
        At(0).S(Shape::kIdle).P(Palette::kIdle).Wander().Strip(kH),
        // listening: ticks swell while the voice is on
        At(4.5).P(Palette::kListen).R(Ring::kListen).Look(0, -4),
        At(5.2).V(true), At(6.0).V(false), At(6.4).V(true), At(7.4).V(false),
        At(8.0).S(Shape::kThink).P(Palette::kThink).R(Ring::kOff),
        // speaking with subtitles: the strip shrinks, the ring has no room
        At(10.4).S(Shape::kHappy).P(Palette::kHappy).R(Ring::kSpeak).Look(0, 2).Strip(150),
        At(13.0).S(Shape::kLove).P(Palette::kLove),
        At(15.4).S(Shape::kStar).P(Palette::kStar).R(Ring::kOff).Strip(kH),
        At(17.6).S(Shape::kDone).P(Palette::kDone).R(Ring::kCelebrate),
        At(20.2).S(Shape::kFail).P(Palette::kFail).R(Ring::kAlert),
        At(22.6).S(Shape::kWide).P(Palette::kWide).R(Ring::kOff),
        At(24.4).S(Shape::kIdle).P(Palette::kSad).Lids(18, 0, 0).Look(0, 8),
        At(26.6).P(Palette::kAngry).Lids(0, 20, 0).Look(0, 0),
        At(28.6).P(Palette::kSleepy).Lids(30, 30, 0),
        At(30.6).P(Palette::kIdle).Lids(0, 0, 0).Wander(),
    };
    return cues;
}

void Apply(Engine& e, const Cue& c) {
    if (c.y1 >= 0) e.SetStrip(kTop, c.y1);
    if (c.shape >= 0) e.SetShape(static_cast<Shape>(c.shape));
    if (c.palette >= 0) e.SetPalette(static_cast<Palette>(c.palette));
    if (c.ring >= 0) e.SetRing(static_cast<Ring>(c.ring));
    if (c.voice >= 0) e.SetVoice(c.voice != 0);
    if (c.set_lids) e.SetLids(c.lids);
    if (c.set_gaze) e.SetGaze(c.wander, c.gx, c.gy);
}

struct Stats {
    int frames = 0, draws = 0;
    double ms_sum = 0, ms_max = 0;
    uint64_t evals_sum = 0, px_sum = 0;
    uint32_t evals_max = 0, px_max = 0;
};

// Runs the demo in RGB565 like the device. out: RGB888 frames, or null.
int RunLive(const char* path, uint32_t seed, Stats* st) {
    std::vector<uint16_t> canvas(kW * kH, 0);
    std::vector<uint8_t> rgb(kW * kH * 3);
    std::vector<float> work(Engine::WorkFloats());
    Engine e;
    e.Init(canvas.data(), eyes::PixelFormat::kRgb565, kW, kH, kW, work.data(), work.size(), seed);
    // EYESIM_EXACT=1: every pixel evaluated exactly, the yardstick for the grid.
    if (getenv("EYESIM_EXACT") != nullptr) e.SetExact(true);
    FILE* f = nullptr;
    if (path != nullptr) {
        f = fopen(path, "wb");
        if (f == nullptr) {
            perror(path);
            return 1;
        }
    }
    const std::vector<Cue>& cues = LiveCues();
    size_t next = 0;
    const int frames = static_cast<int>(kLiveSec * kFps);
    eyes::Rect rects[Engine::kMaxRects];
    for (int i = 1; i <= frames; i++) {
        const double t = i / kFps;
        while (next < cues.size() && cues[next].t <= t + 1e-9) Apply(e, cues[next++]);
        const bool need = e.Advance(1 / kFps);
        st->frames++;
        if (need) {
            const auto t0 = std::chrono::steady_clock::now();
            e.Draw(rects, Engine::kMaxRects);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            st->draws++;
            st->ms_sum += ms;
            st->ms_max = std::max(st->ms_max, ms);
            st->evals_sum += e.sdf_evals();
            st->evals_max = std::max(st->evals_max, e.sdf_evals());
            st->px_sum += e.dirty_pixels();
            st->px_max = std::max(st->px_max, e.dirty_pixels());
        }
        if (f != nullptr) {
            for (int p = 0; p < kW * kH; p++) {
                const uint16_t v = canvas[p];
                const int r = v >> 11, g = (v >> 5) & 63, b = v & 31;
                rgb[p * 3] = static_cast<uint8_t>((r << 3) | (r >> 2));
                rgb[p * 3 + 1] = static_cast<uint8_t>((g << 2) | (g >> 4));
                rgb[p * 3 + 2] = static_cast<uint8_t>((b << 3) | (b >> 2));
            }
            if (!WriteAll(f, rgb.data(), rgb.size())) {
                perror(path);
                fclose(f);
                return 1;
            }
        }
    }
    if (f != nullptr) fclose(f);
    return 0;
}

void PrintStats(const Stats& s) {
    const double avg_px = s.draws ? static_cast<double>(s.px_sum) / s.draws : 0;
    printf("frames %d, drawn %d (%.0f%%)\n", s.frames, s.draws, 100.0 * s.draws / std::max(1, s.frames));
    printf("host draw ms: avg %.3f, max %.3f\n", s.draws ? s.ms_sum / s.draws : 0, s.ms_max);
    printf("sdf evals per draw: avg %.0f, max %u\n", s.draws ? static_cast<double>(s.evals_sum) / s.draws : 0,
           s.evals_max);
    printf("dirty px per draw: avg %.0f, max %u  (SPI at 40 MHz: avg %.1f ms, max %.1f ms)\n", avg_px, s.px_max,
           avg_px * 16 / 40e3, s.px_max * 16 / 40e3);
}

bool ParseShape(const char* s, Shape* out) {
    static const char* names[] = {"idle", "happy", "love", "star", "think", "done", "fail", "wide"};
    for (int i = 0; i < static_cast<int>(Shape::kCount); i++) {
        if (strcmp(s, names[i]) == 0) {
            *out = static_cast<Shape>(i);
            return true;
        }
    }
    return false;
}

bool ParsePalette(const char* s, Palette* out) {
    static const char* names[] = {"idle", "listen", "think", "speak", "happy", "love", "star",
                                  "done", "fail",   "sad",   "angry", "wide",  "sleepy"};
    for (int i = 0; i < static_cast<int>(Palette::kCount); i++) {
        if (strcmp(s, names[i]) == 0) {
            *out = static_cast<Palette>(i);
            return true;
        }
    }
    return false;
}

int RunStill(const char* shape, const char* palette, int y1, double sec, const char* path, uint32_t bg) {
    Shape s;
    Palette p;
    if (!ParseShape(shape, &s) || !ParsePalette(palette, &p)) {
        fprintf(stderr, "unknown shape or palette\n");
        return 2;
    }
    std::vector<uint16_t> canvas(kW * kH, 0);
    std::vector<float> work(Engine::WorkFloats());
    Engine e;
    e.Init(canvas.data(), eyes::PixelFormat::kRgb565, kW, kH, kW, work.data(), work.size(), 1);
    e.SetDeterministic(true);
    e.SetBackground(bg);
    e.SetStrip(kTop, y1);
    e.SetShape(s);
    e.SetPalette(p);
    if (getenv("EYESIM_EXACT") != nullptr) e.SetExact(true);
    eyes::Rect rects[Engine::kMaxRects];
    // Land on `sec` exactly: the idle eyes breathe, so a device snapshot only
    // matches at the time it reports.
    const int frames = static_cast<int>(sec * kFps);
    for (int i = 0; i < frames; i++) {
        e.Advance(1 / kFps);
        e.Draw(rects, Engine::kMaxRects);
    }
    const double rest = sec - frames / kFps;
    if (rest > 1e-9) {
        e.Advance(rest);
        e.Draw(rects, Engine::kMaxRects);
    }
    FILE* f = fopen(path, "wb");
    if (f == nullptr || !WriteAll(f, canvas.data(), canvas.size() * 2)) {
        perror(path);
        if (f != nullptr) fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

int Usage() {
    fprintf(stderr,
            "usage: eyesim seq OUT.rgb [exact] | cmp A.rgb B.rgb | live OUT.rgb [SEED] | bench [SEED]\n"
            "       | still SHAPE PALETTE Y1 SEC OUT.565 [BG_RGB888]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return Usage();
    const char* mode = argv[1];
    if (strcmp(mode, "seq") == 0 && argc >= 3) {
        return RunSeq(argv[2], argc >= 4 && strcmp(argv[3], "exact") == 0);
    }
    if (strcmp(mode, "cmp") == 0 && argc >= 4) return RunCmp(argv[2], argv[3]);
    if (strcmp(mode, "live") == 0 && argc >= 3) {
        Stats st;
        const int rc = RunLive(argv[2], argc >= 4 ? static_cast<uint32_t>(strtoul(argv[3], nullptr, 0)) : 7, &st);
        if (rc == 0) PrintStats(st);
        return rc;
    }
    if (strcmp(mode, "bench") == 0) {
        Stats st;
        const int rc = RunLive(nullptr, argc >= 3 ? static_cast<uint32_t>(strtoul(argv[2], nullptr, 0)) : 7, &st);
        if (rc == 0) PrintStats(st);
        return rc;
    }
    if (strcmp(mode, "still") == 0 && argc >= 7) {
        const uint32_t bg = argc >= 8 ? static_cast<uint32_t>(strtoul(argv[7], nullptr, 16)) : 0;
        return RunStill(argv[2], argv[3], atoi(argv[4]), atof(argv[5]), argv[6], bg);
    }
    return Usage();
}
