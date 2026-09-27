#include "eye_engine.h"

#include <math.h>
#include <string.h>

#include <algorithm>
#include <new>

#if defined(__XTENSA__)
// The firmware is built for size; this file is its one hot loop.
#pragma GCC optimize("O2")
#endif

// Everything a pixel touches is forced inline. On the Xtensa ABI a call costs
// far more than it looks: floats travel in integer registers and every live
// float register is spilled around it, and -Os left even Clamp() out of line.
#if defined(__GNUC__)
#define EYE_INLINE inline __attribute__((always_inline))
#define EYE_LAMBDA __attribute__((always_inline))
#define EYE_NOINLINE __attribute__((noinline))
#else
#define EYE_INLINE inline
#define EYE_LAMBDA
#define EYE_NOINLINE
#endif

namespace eyes {
namespace {

constexpr float kPi = 3.14159265f;
constexpr float kTau = 6.28318531f;
constexpr double kTauD = 6.283185307179586;
constexpr uint8_t kDropKind = 0xFF;
constexpr float kUnset = 1e30f;

// Face layout in design pixels: scale 1 is the approved mock-up.
constexpr float kEyeDx = 52.f;     // eye centre to the middle of the face
constexpr float kHighlightParallax = .35f;  // highlight shift per unit of gaze, against it
constexpr float kBev = 9.f;        // depth of the rounded bevel inside the edge
constexpr float kGlw = 18.f;       // how far the glow reaches outside the edge
constexpr float kGlwFade = 14.f;   // ...fading to nothing from here on
constexpr int kBlock = 8;
// RenderEye's grid. A lit pixel may need its exact distance, and its exact gradient.
enum : uint8_t { kExactD = 1, kExactG = 2 };
// A true distance field has |gradient| 1; well short of that the grid is
// straddling a ridge or a corner and its gradient cannot be trusted.
constexpr float kMinGrad2 = .97f * .97f;
// Second differences (per 2 px node step) below this count as smooth.
constexpr float kSmooth = .6f;
constexpr float kRough = 1e3f;

enum : uint8_t { kTrDrop, kTrFast, kTrSlow };

struct PaletteDef {
    uint32_t top, bottom;
};
constexpr PaletteDef kPalettes[] = {
    {0x9BEBFF, 0x2F7BFF},  // idle
    {0xC2FFD2, 0x16C47F},  // listen
    {0xF0C8FF, 0x8B4DFF},  // think
    {0xFFE39A, 0xFF8A3D},  // speak
    {0xFFE870, 0xFF5FA2},  // happy
    {0xFFA3C8, 0xFF2D6F},  // love
    {0xFFF6B0, 0xFFAA1E},  // star
    {0xB8FFC8, 0x1FC85A},  // done
    {0xFFB4A6, 0xFF3B30},  // fail
    {0xB4CCF5, 0x4262D6},  // sad
    {0xFFC3A0, 0xFF5A28},  // angry
    {0xFFFFFF, 0x9FD8FF},  // wide
    {0xCFC8FF, 0x5A4FCF},  // sleepy
};
static_assert(sizeof(kPalettes) / sizeof(kPalettes[0]) == static_cast<size_t>(Palette::kCount),
              "one palette per Palette value");

// Half extents of each shape around its eye centre, design px.
constexpr float kExtent[][2] = {
    {35, 42},  // idle, before the blink widens it
    {31, 19},  // happy
    {31, 28},  // love
    {35, 35},  // star
    {28, 28},  // think
    {29, 23},  // done, stroke popped
    {25, 25},  // fail, stroke popped
    {34, 34},  // wide
};
static_assert(sizeof(kExtent) / sizeof(kExtent[0]) == static_cast<size_t>(Shape::kCount),
              "one extent per Shape value");

EYE_INLINE float Clamp(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
// The S3 has no square-root or divide instruction: sqrtf() and `/` are library
// calls, costly inside per-pixel loops. The inverse square root by the bit trick
// plus two Newton steps is good to ~5e-6 relative, and inline. (ESP-IDF builds
// with -fno-builtin-memcpy, so a plain memcpy here would be a real call.)
EYE_INLINE float FastRsqrt(float x) {
    uint32_t i;
    float y;
#if defined(__XTENSA__)
    // Register to register: GCC routes a memcpy between the float and the
    // integer registers through the stack, a store and a stalled load each way.
    __asm__("rfr %0, %1" : "=a"(i) : "f"(x));
    i = 0x5f375a86u - (i >> 1);
    __asm__("wfr %0, %1" : "=f"(y) : "a"(i));
#else
    __builtin_memcpy(&i, &x, 4);
    i = 0x5f375a86u - (i >> 1);
    __builtin_memcpy(&y, &i, 4);
#endif
    const float hx = .5f * x;
    y *= 1.5f - hx * y * y;
    y *= 1.5f - hx * y * y;
    return y;
}
EYE_INLINE float FastSqrt(float x) { return x * FastRsqrt(x); }   // x >= 0
EYE_INLINE float FastRecip(float x) {                            // x > 0
    const float r = FastRsqrt(x);
    return r * r;
}
EYE_INLINE float Len(float x, float y) { return FastSqrt(x * x + y * y); }
EYE_INLINE float Sign(float v) { return v > 0 ? 1.f : (v < 0 ? -1.f : 0.f); }
inline float EaseInOut(float u) {
    if (u < .5f) return 4 * u * u * u;
    const float v = 2 - 2 * u;
    return 1 - v * v * v * .5f;
}
inline float EaseOut(float u) {
    const float v = 1 - u;
    return 1 - v * v * v;
}
// Semi-implicit Euler, the same integrator the mock-up used.
inline void Spring(float& p, float& v, float to, float w, float z, float dt) {
    v += (-2 * z * w * v - w * w * (p - to)) * dt;
    p += v * dt;
}
// Angle reduced in double first, so float trig stays exact over long uptimes.
inline float Wrap(double a) { return static_cast<float>(a - kTauD * floor(a / kTauD)); }
inline float WrapF(float a) { return a - kTau * floorf(a / kTau); }
// floorf() is a library call on the S3; this is exact for |v| < 2^23.
EYE_INLINE float FloorF(float v) {
    const float t = static_cast<float>(static_cast<int>(v));
    return t > v ? t - 1 : t;
}

// ---- Oklab ------------------------------------------------------------------------------

inline float SrgbToLinear(float c) {
    c /= 255.f;
    return c <= .04045f ? c / 12.92f : powf((c + .055f) / 1.055f, 2.4f);
}
inline float LinearToSrgb(float c) {
    c = c <= .0031308f ? 12.92f * c : 1.055f * powf(std::max(c, 0.f), 1 / 2.4f) - .055f;
    return Clamp(c * 255.f, 0, 255);
}
void ToOklab(uint32_t rgb, float o[3]) {
    const float r = SrgbToLinear((rgb >> 16) & 0xFF), g = SrgbToLinear((rgb >> 8) & 0xFF),
                b = SrgbToLinear(rgb & 0xFF);
    const float l = cbrtf(.4122214708f * r + .5363325363f * g + .0514459929f * b);
    const float m = cbrtf(.2119034982f * r + .6806995451f * g + .1073969566f * b);
    const float s = cbrtf(.0883024619f * r + .2817188376f * g + .6299787005f * b);
    o[0] = .2104542553f * l + .793617785f * m - .0040720468f * s;
    o[1] = 1.9779984951f * l - 2.428592205f * m + .4505937099f * s;
    o[2] = .0259040371f * l + .7827717662f * m - .808675766f * s;
}
void FromOklab(const float o[3], float rgb[3]) {
    float l = o[0] + .3963377774f * o[1] + .2158037573f * o[2];
    float m = o[0] - .1055613458f * o[1] - .0638541728f * o[2];
    float s = o[0] - .0894841775f * o[1] - 1.291485548f * o[2];
    l = l * l * l;
    m = m * m * m;
    s = s * s * s;
    rgb[0] = LinearToSrgb(4.0767416621f * l - 3.3077115913f * m + .2309699292f * s);
    rgb[1] = LinearToSrgb(-1.2684380046f * l + 2.6097574011f * m - .3413193965f * s);
    rgb[2] = LinearToSrgb(-.0041960863f * l - .7034186147f * m + 1.707614701f * s);
}

// ---- distance functions (after Inigo Quilez) ------------------------------------------

EYE_INLINE float Seg(float px, float py, float ax, float ay, float bx, float by) {
    const float pax = px - ax, pay = py - ay, bax = bx - ax, bay = by - ay;
    const float bb = bax * bax + bay * bay;
    const float h = bb > 1e-9f ? Clamp((pax * bax + pay * bay) * FastRecip(bb), 0, 1) : 0;
    return Len(pax - bax * h, pay - bay * h);
}
EYE_INLINE float Box(float px, float py, float hx, float hy, float r) {
    const float qx = fabsf(px) - hx + r, qy = fabsf(py) - hy + r;
    return Len(std::max(qx, 0.f), std::max(qy, 0.f)) + std::min(std::max(qx, qy), 0.f) - r;
}
// Arc of radius ra and thickness rb, aperture given by (s, c) = sin/cos, open downwards.
EYE_INLINE float ArcD(float px, float py, float s, float c, float ra, float rb) {
    px = fabsf(px);
    return (c * px > s * py ? Len(px - s * ra, py - c * ra) : fabsf(Len(px, py) - ra)) - rb;
}
EYE_INLINE float Heart(float x, float y) {
    x = fabsf(x);
    if (y + x > 1) return Len(x - .25f, y - .75f) - .35355339f;
    const float m = .5f * std::max(x + y, 0.f);
    const float a = x * x + (y - 1) * (y - 1), b = (x - m) * (x - m) + (y - m) * (y - m);
    return FastSqrt(std::min(a, b)) * Sign(x - y);
}
EYE_INLINE float Star5(float px, float py, float r, float rf) {
    const float k1x = .809016994375f, k1y = -.587785252292f;
    px = fabsf(px);
    float d = 2 * std::max(k1x * px + k1y * py, 0.f);
    px -= d * k1x;
    py -= d * k1y;
    d = 2 * std::max(-k1x * px + k1y * py, 0.f);
    px += d * k1x;
    py -= d * k1y;
    px = fabsf(px);
    py -= r;
    const float bx = -rf * k1y, by = rf * k1x - 1;
    const float h = Clamp((px * bx + py * by) * (1.f / (bx * bx + by * by)), 0, r);
    return Len(px - bx * h, py - by * h) * Sign(py * bx - px * by);
}

// The idle capsule, bent by the lids and squeezed by the blink. Parameters are
// passed in so a batch of points keeps them in registers.
struct IdleP {
    float k;                     // blink
    float inner, outer, bottom;  // lids
    float zone;                  // zoning-out lid
    float inv_wd;                // 1 / blink widening
    bool left;                   // eye 0: the outer corner is at -x
};
EYE_INLINE float IdleD(float x, float y, const IdleP& p) {
    const float hw = 35, hh = 42, cz = hh * .44f;
    const float xo = p.left ? -x : x;   // + towards the outer corner
    const float u = Clamp((xo + hw) * (.5f / hw), 0, 1);
    const float drop = p.inner + (p.outer - p.inner) * u;
    const float y0 = -hh - 1 + p.zone * 2 * hh + drop;
    const float ya = y0 + (cz - 1.5f - y0) * p.k;
    const float xr = x * (1.f / hw);
    const float yb = hh + 1 - p.bottom * (1 - .5f * xr * xr);
    const float yl = yb + (cz + 1.5f - yb) * p.k;
    return std::max(std::max(Box(x * p.inv_wd, y, hw, hh, 17), ya - y), y - yl);
}
// Batches of the above for Engine::DistN, each loop in a function of its own so
// its constants stay in registers. BoxN is the capsule alone, which is all of
// it while the eye is open (see Engine::OpenIdle).
EYE_NOINLINE void BoxN(int n, const float* __restrict lx, const float* __restrict ly, float* __restrict out,
                       float isx, float iw, float isy, float sm) {
    for (int k = 0; k < n; k++) out[k] = Box(lx[k] * isx * iw, ly[k] * isy, 35, 42, 17) * sm;
}
EYE_NOINLINE void IdleN(int n, const float* __restrict lx, const float* __restrict ly, float* __restrict out,
                        float isx, float isy, float sm, const IdleP& p) {
    const IdleP P = p;
    for (int k = 0; k < n; k++) out[k] = IdleD(lx[k] * isx, ly[k] * isy, P) * sm;
}
// The capsule along a row, at points lx[0 .. n): BoxN's values, only with no
// root taken where the edge is straight, and the corners two points at a time
// so that their roots overlap. (lx steps by 1 px.)
EYE_NOINLINE void BoxRow(int n, const float* __restrict lx, float ly, float* __restrict out, float ax, float ay,
                         float sm) {
    const float qy = fabsf(ly * ay) - (42 - 17);
    if (qy <= 0) {
        for (int k = 0; k < n; k++) out[k] = (std::max(fabsf(lx[k] * ax) - (35 - 17), qy) - 17) * sm;
        return;
    }
    // Above or below the sides: flat across the middle (|lx| * ax <= 18).
    const float hm = (35 - 17) / ax;
    const int ka = n > 0 ? std::min(std::max(-static_cast<int>(FloorF(hm + lx[0])), 0), n) : 0;
    const int kb = n > 0 ? std::min(std::max(static_cast<int>(FloorF(hm - lx[0])) + 1, ka), n) : 0;
    const float mid = (qy - 17) * sm, qy2 = qy * qy;
    auto corner = [&](int k0, int k1) EYE_LAMBDA {
        int k = k0;
        for (; k + 1 < k1; k += 2) {
            const float qa = fabsf(lx[k] * ax) - (35 - 17), qb = fabsf(lx[k + 1] * ax) - (35 - 17);
            const float ra = FastSqrt(qa * qa + qy2), rb = FastSqrt(qb * qb + qy2);
            out[k] = (ra - 17) * sm;
            out[k + 1] = (rb - 17) * sm;
        }
        if (k < k1) {
            const float qa = fabsf(lx[k] * ax) - (35 - 17);
            out[k] = (FastSqrt(qa * qa + qy2) - 17) * sm;
        }
    };
    corner(0, ka);
    for (int k = ka; k < kb; k++) out[k] = mid;
    corner(kb, n);
}

// Columns k with |k - kc| <= h, as [lo, hi) clipped to [0, n]; empty if h < 0.
struct Span {
    int lo, hi;
};
EYE_INLINE Span Around(float kc, float h, int n) {
    int lo, hi;
    if (h >= 0) {
        lo = -static_cast<int>(FloorF(h - kc));
        hi = static_cast<int>(FloorF(kc + h)) + 1;
    } else {
        lo = hi = static_cast<int>(FloorF(kc)) + 1;
    }
    lo = std::min(std::max(lo, 0), n);
    return {lo, std::min(std::max(hi, lo), n)};
}
// `s` squeezed into `in`, so that spans nest.
EYE_INLINE Span Inside(Span s, Span in) {
    s.lo = std::min(std::max(s.lo, in.lo), in.hi);
    s.hi = std::min(std::max(s.hi, s.lo), in.hi);
    return s;
}


constexpr float kHappyS = .91276394f, kHappyC = .40848744f;   // sin, cos of 1.15
constexpr float kCheck1 = 19.79898987f;   // hypot(14, 14)
constexpr float kCheck2 = 40.36087214f;   // hypot(27, 30)

// Check mark drawn from its left end; p = 0..1 of the stroke.
EYE_INLINE float CheckD(float x, float y, float p, float w) {
    const float len = p * (kCheck1 + kCheck2);
    if (len <= kCheck1) {
        const float u = len * (1.f / kCheck1);
        return Seg(x, y, -20, 1, -20 + 14 * u, 1 + 14 * u) - w;
    }
    const float u = (len - kCheck1) * (1.f / kCheck2);
    return std::min(Seg(x, y, -20, 1, -6, 15), Seg(x, y, -6, 15, -6 + 27 * u, 15 - 30 * u)) - w;
}
// Cross: first stroke over the first half, second stroke over the second.
EYE_INLINE float CrossD(float x, float y, float p, float w) {
    const float u1 = Clamp(p * 2, 0, 1);
    float d = Seg(x, y, -17, -17, -17 + 34 * u1, -17 + 34 * u1) - w;
    if (p > .5f) {
        const float u2 = Clamp(p * 2 - 1, 0, 1);
        d = std::min(d, Seg(x, y, 17, -17, 17 - 34 * u2, -17 + 34 * u2) - w * Clamp((p - .5f) * 8, 0, 1));
    }
    return d;
}

// Minimax atan2, error ~1e-3 rad: plenty for a 1 px wide stroke 100 px out.
EYE_INLINE float FastAtan2(float y, float x) {
    const float ax = fabsf(x), ay = fabsf(y);
    const float a = std::min(ax, ay) * FastRecip(std::max(ax, ay) + 1e-12f);
    const float s = a * a;
    float r = ((-.0464964749f * s + .15931422f) * s - .327622764f) * s * a + a;
    if (ay > ax) r = 1.57079637f - r;
    if (x < 0) r = kPi - r;
    return y < 0 ? -r : r;
}

// ---- lighting and lookup tables -------------------------------------------------------

constexpr int kGlowRes = 16;   // entries per px
constexpr int kGlowN = static_cast<int>(kGlw) * kGlowRes + 2;
float g_glow[kGlowN];      // fades out towards kGlw
float g_glow_raw[kGlowN];  // the mock-up's glow, cut off hard (reference mode)
float g_spec[258];         // 230 nh^60 + 36 nh^8
float g_glint[258];        // exp(-q^2/30), q in [-16, 16]
float g_light[6];          // key light L, then the half vector H
bool g_tables = false;
const uint8_t kBayer[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};
// Half a 5-bit step times the Bayer threshold, (b + .5) / 16 * 8: added to red
// and blue, and halved for green's 6 bits.
float g_dither[4][4];
// The same thresholds for Swar565(): 2 b + 1 in each of its three fields.
uint32_t g_dswar[4][4];
// Direction of each listening-ring tick, from the top clockwise: cos, sin.
constexpr int kTicks = 60;
float g_tick[kTicks][2];

// Glow over the plain background per 1/16 px of distance: its amount, and the
// RGB565 colour ready for Swar565(). Both eyes share them; they are rebuilt
// when what they depend on changes.
uint32_t g_glow565[kGlowN];
float g_glowv[kGlowN];
float g_glow_key[8];
bool g_glow_ok = false;

// A colour packed for dithering by integer adds: blue in bits 0-9 in quarter
// steps, green in 10-20 in eighths, red in 21-30 in quarters, each clamped so
// a g_dswar threshold never carries out of its field. The top bits of each
// field after the add are what WritePixel() would write.
EYE_INLINE uint16_t Swar565(uint32_t v) {
    return static_cast<uint16_t>(((v >> 15) & 0xF800u) | ((v >> 10) & 0x07E0u) | ((v >> 5) & 0x001Fu));
}
// Ordered dither: gradients on RGB565 would band without it. Nothing drawn is
// negative, so only the top needs clamping. `th` is the pixel's g_dither entry.
EYE_INLINE uint16_t Pack565(float r, float g, float b, float th) {
    const int ri = std::min(static_cast<int>(r + th), 255), gi = std::min(static_cast<int>(g + th * .5f), 255),
              bi = std::min(static_cast<int>(b + th), 255);
    return static_cast<uint16_t>(((ri & 0xF8) << 8) | ((gi & 0xFC) << 3) | (bi >> 3));
}
// The glow table's bin for a distance in table steps.
EYE_INLINE int GlowBin(float fb) {
    const int bin = static_cast<int>(fb);
    return bin < 0 ? 0 : (bin > kGlowN - 1 ? kGlowN - 1 : bin);
}

// What PaintRow needs of a row.
struct RowPaint {
    uint16_t* o;       // the row's pixels, from the box's left edge
    const float* d;    // its distances
    const float* ch;   // the column highlight, from the box's left edge
    float th[4];       // Pack565's dither by column & 3
    uint32_t dl[4];    // the glow's, likewise
    float fb[3], rh;   // the row's flat colour and highlight
    int xo;            // glow left of this goes onto eye 0's: queued with the lit
};

// One row of the open eye, fast path: the flat run, the glow runs, and pixel by
// pixel the rest. Out here so that nothing it writes can alias what it reads.
// Queues lit pixels in `lit` and returns how many; adds up flat and glow ones.
EYE_NOINLINE int PaintRow(const RowPaint& p, Span f, const Span* gs, const Span* cs, uint16_t* __restrict lit,
                          int* __restrict counts) {
    uint16_t* __restrict o = p.o;
    const float* __restrict d = p.d;
    const float* __restrict ch = p.ch;
    float th[4];
    uint32_t dl[4];
    for (int i = 0; i < 4; i++) {
        th[i] = p.th[i];
        dl[i] = p.dl[i];
    }
    const float r = p.fb[0], g = p.fb[1], b = p.fb[2], rh = p.rh;
    const int xo = p.xo;
    int nf = f.hi - f.lo, ng = 0, nlit = 0;
    for (int k = f.lo; k < f.hi; k++) {
        const float ad = rh * ch[k];
        o[k] = Pack565(r + ad, g + ad, b + ad, th[k & 3]);
    }
    for (int s = 0; s < 2; s++) {
        const int lo = gs[s].lo, hi = gs[s].hi, mid = std::min(std::max(xo, lo), hi);
        for (int k = lo; k < mid; k++) lit[nlit++] = static_cast<uint16_t>(k);
        for (int k = mid; k < hi; k++) o[k] = Swar565(g_glow565[GlowBin(d[k] * kGlowRes)] + dl[k & 3]);
        ng += hi - mid;
    }
    for (int s = 0; s < 4; s++) {
        for (int k = cs[s].lo; k < cs[s].hi; k++) {
            const float di = d[k];
            if (di > kGlw) continue;
            if (di <= -kBev) {
                const float ad = rh * ch[k];
                o[k] = Pack565(r + ad, g + ad, b + ad, th[k & 3]);
                nf++;
            } else if (di >= .5f && k >= xo) {
                o[k] = Swar565(g_glow565[GlowBin(di * kGlowRes)] + dl[k & 3]);
                ng++;
            } else {
                lit[nlit++] = static_cast<uint16_t>(k);
            }
        }
    }
    counts[0] += nf;
    counts[1] += ng;
    return nlit;
}

void BuildTables() {
    if (g_tables) return;
    for (int i = 0; i < kGlowN; i++) {
        const float d = static_cast<float>(i) / kGlowRes;
        const float g = expf(-d / 3.2f) * .34f + expf(-d / 9.f) * .2f;
        float f = Clamp((kGlw - d) / (kGlw - kGlwFade), 0, 1);
        f = f * f * (3 - 2 * f);
        g_glow_raw[i] = g;
        g_glow[i] = g * f;
    }
    for (int i = 0; i < 258; i++) {
        const float nh = std::min(1.f, i / 256.f);
        g_spec[i] = 230.f * powf(nh, 60.f) + 36.f * powf(nh, 8.f);
        const float q = i / 8.f - 16.f;
        g_glint[i] = expf(-q * q / 30.f);
    }
    const double l[3] = {-.5, -.6, .62};
    const double ll = sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
    const double h[3] = {l[0] / ll, l[1] / ll, l[2] / ll + 1};
    const double hl = sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
    for (int i = 0; i < 3; i++) {
        g_light[i] = static_cast<float>(l[i] / ll);
        g_light[3 + i] = static_cast<float>(h[i] / hl);
    }
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            g_dither[y][x] = (kBayer[y][x] + .5f) * .5f;
            const uint32_t t = 2u * kBayer[y][x] + 1;
            g_dswar[y][x] = t | t << 10 | t << 21;
        }
    }
    for (int i = 0; i < kTicks; i++) {
        const double a = i * kTauD / kTicks - kTauD / 4;
        g_tick[i][0] = static_cast<float>(cos(a));
        g_tick[i][1] = static_cast<float>(sin(a));
    }
    g_tables = true;
}
EYE_INLINE float GlowAt(const float* lut, float d) {
    const float f = d * kGlowRes;
    const int i = static_cast<int>(f);
    if (i >= kGlowN - 1) return 0;
    return lut[i] + (lut[i + 1] - lut[i]) * (f - i);
}
EYE_INLINE float SpecAt(float nh) {
    const float f = nh * 256.f;
    const int i = static_cast<int>(f);
    return g_spec[i] + (g_spec[i + 1] - g_spec[i]) * (f - i);
}
EYE_INLINE float GlintAt(float q) {
    if (q <= -16.f || q >= 16.f) return 0;
    const float f = (q + 16.f) * 8.f;
    const int i = static_cast<int>(f);
    return g_glint[i] + (g_glint[i + 1] - g_glint[i]) * (f - i);
}

// A lit pixel's lighting (Engine::Light) from the light and half vectors.
struct LightV {
    float lx, ly, lz, hx, hy, hz;
};
EYE_INLINE void LightOf(const LightV& v, float d, float gx, float gy, float& br, float& sp, float& rm) {
    const float k = d < 0 ? 1 + d * (1.f / kBev) : 1.f;
    // The highlight is nh^60: the normal needs the full-precision root.
    const float sc = k * FastRsqrt(std::max(gx * gx + gy * gy, 1e-18f));
    const float nx = gx * sc, ny = gy * sc;
    const float nz = FastSqrt(std::max(0.f, (1 - k) * (1 + k)));
    const float nl = std::max(0.f, nx * v.lx + ny * v.ly + nz * v.lz);
    const float nh = std::max(0.f, nx * v.hx + ny * v.hy + nz * v.hz);
    const float k2 = k * k;
    br = .34f + .72f * nl;
    sp = SpecAt(nh);
    rm = k > .5f ? k2 * k2 * .35f : 0.f;
}

// Fills g_glow565 / g_glowv for this glow strength, colour and background.
void PrepareGlow(bool ref, float en, const float cb[3], const float bgc[3]) {
    const float key[8] = {ref ? 1.f : 0.f, en, cb[0], cb[1], cb[2], bgc[0], bgc[1], bgc[2]};
    bool same = g_glow_ok;
    for (int i = 0; i < 8 && same; i++) same = key[i] == g_glow_key[i];
    if (same) return;
    std::copy(key, key + 8, g_glow_key);
    g_glow_ok = true;
    const float* lut = ref ? g_glow_raw : g_glow;
    const float cmax = std::max(cb[0], std::max(cb[1], cb[2]));
    auto fix = [](float v, float unit, int top) {
        const int q = static_cast<int>(v * unit + .5f);
        return static_cast<uint32_t>(q < 0 ? 0 : (q > top ? top : q));
    };
    for (int i = 0; i < kGlowN; i++) {
        const float d = (i + .5f) / kGlowRes;
        float gl = d > kGlw ? 0.f : GlowAt(lut, d) * en;
        // Too faint to show: the per-pixel path leaves such pixels alone.
        if (gl * cmax < .25f) gl = 0;
        g_glowv[i] = gl;
        // 248 and 252 already dither to full scale, and leave the headroom.
        g_glow565[i] = fix(bgc[0] + cb[0] * gl, 4, 992) << 21 | fix(bgc[1] + cb[1] * gl, 8, 2016) << 10 |
                       fix(bgc[2] + cb[2] * gl, 4, 992);
    }
}

// out[i] = amp exp(-(q0 + i dq)^2 / s2) for i < n (dq > 0), by ratios from the
// sample nearest the peak outwards: a multiply per sample instead of an expf(),
// and a tail that underflows stays at zero.
void Gauss(float* out, int n, float q0, float dq, float s2, float amp) {
    if (n <= 0) return;
    int ip = static_cast<int>(-q0 / dq + .5f);
    ip = ip < 0 ? 0 : (ip > n - 1 ? n - 1 : ip);
    const float inv = 1 / s2, qp = q0 + ip * dq;
    const float peak = amp * expf(-qp * qp * inv), c = expf(-2 * dq * dq * inv);
    out[ip] = peak;
    float v = peak, r = expf(-(2 * qp * dq + dq * dq) * inv);
    for (int i = ip + 1; i < n; i++) {
        v *= r;
        r *= c;
        out[i] = v;
    }
    v = peak;
    r = expf(-(dq * dq - 2 * qp * dq) * inv);
    for (int i = ip - 1; i >= 0; i--) {
        v *= r;
        r *= c;
        out[i] = v;
    }
}


float RingBand(Ring r) {
    switch (r) {
        case Ring::kListen:
        case Ring::kSpeak: return 9.5f;
        case Ring::kCelebrate: return 3.5f;
        default: return 4.f;
    }
}

}  // namespace

// ---- setup ------------------------------------------------------------------------------

void Engine::Init(void* canvas, PixelFormat fmt, int width, int height, int stride_px,
                  float* work, size_t work_floats, uint32_t seed) {
    BuildTables();
    canvas_ = canvas;
    fmt_ = fmt;
    w_ = std::min(width, kMaxDim);
    h_ = std::min(height, kMaxDim);
    stride_ = stride_px;
    grid_ = (work != nullptr && work_floats >= kWorkFloats) ? work : nullptr;
    rng_ = seed != 0 ? seed : 0x9E3779B9u;
    strip_y0_ = clear_y0_ = 0;
    strip_y1_ = clear_y1_ = h_;
    mid_ = h_ * .5f;
    for (Morph& m : morph_) {
        m = Morph();
        m.to = static_cast<uint8_t>(Shape::kIdle);
        m.drop = DropFor(Shape::kIdle);
    }
    colors_init_ = false;
    UpdateColors(0);
    SetBackground(0);
    ComputeXforms();
}

void Engine::SetStrip(int y0, int y1) {
    y0 = std::max(0, std::min(y0, h_));
    y1 = std::max(y0, std::min(y1, h_));
    if (y0 == strip_y0_ && y1 == strip_y1_) return;
    // Rows the face leaves behind get cleared too.
    clear_y0_ = std::min(clear_y0_, strip_y0_);
    clear_y1_ = std::max(clear_y1_, strip_y1_);
    strip_y0_ = y0;
    strip_y1_ = y1;
    full_clear_ = true;
}

void Engine::SetBackground(uint32_t rgb) {
    bg_[0] = static_cast<float>((rgb >> 16) & 0xFF);
    bg_[1] = static_cast<float>((rgb >> 8) & 0xFF);
    bg_[2] = static_cast<float>(rgb & 0xFF);
    bg565_ = static_cast<uint16_t>(((rgb >> 8) & 0xF800) | ((rgb >> 5) & 0x07E0) | ((rgb >> 3) & 0x001F));
    full_clear_ = true;
}

void Engine::SetShape(Shape s) {
    if (s == shape_) {
        pending_shape_ = -1;
        return;
    }
    pending_shape_ = static_cast<int>(s);
}

void Engine::SetPalette(Palette p) { palette_ = p; }
void Engine::SetLids(const LidPose& lids) { lids_target_ = lids; }

void Engine::SetGaze(bool wander, float x, float y) {
    wander_ = wander;
    hold_x_ = x;
    hold_y_ = y;
}

void Engine::SetRing(Ring r) {
    ring_ = r;
    if (r != Ring::kOff) ring_shown_ = r;
}

void Engine::SetVoice(bool active) { voice_ = active; }

float Engine::Rand() {
    uint32_t x = rng_;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_ = x;
    return (x >> 8) * (1.f / 16777216.f);
}

// ---- shapes -----------------------------------------------------------------------------

Engine::Term Engine::DropFor(Shape s) {
    switch (s) {
        case Shape::kDone: return {kDropKind, 1, -20, 1, 6.5f};
        case Shape::kFail: return {kDropKind, 1, -17, -17, 6.5f};
        default: return {kDropKind, 1, 0, 0, 22};
    }
}

float Engine::EvalShape(uint8_t kind, float x, float y, int eye) const {
    switch (static_cast<Shape>(kind)) {
        case Shape::kIdle:
            return IdleD(x, y, IdleP{blink_, lids_.inner, lids_.outer, lids_.bottom, zone_lid_, inv_wd_, eye == 0});
        case Shape::kHappy:
            return ArcD(x, 14 - y, kHappyS, kHappyC, 26, 7);
        case Shape::kLove: {
            constexpr float kS = 50;
            return Heart(x * (1.f / kS), -y * (1.f / kS) + .55f) * kS;
        }
        case Shape::kStar:
            return Star5(x * sc_ - y * ss_, -(x * ss_ + y * sc_), 32, .48f) - 3;
        case Shape::kThink: {
            const float sn = eye == 0 ? -rs_ : rs_;
            return ArcD(x * rc_ - y * sn, x * sn + y * rc_, ta_, tb_, 22, 5.5f);
        }
        case Shape::kDone:
            return CheckD(x, y, prog_[0][eye], 6.5f * (1 + 1.2f * pop_[0][eye]));
        case Shape::kFail:
            return CrossD(x, y, prog_[1][eye], 6.5f * (1 + 1.2f * pop_[1][eye]));
        case Shape::kWide:
            return Len(x, y) - 34;
        default:
            return 1e3f;
    }
}

EYE_INLINE float Engine::EvalTerm(const Term& t, float x, float y, int eye) const {
    if (t.kind == kDropKind) return Len(x - t.a, y - t.b) - t.c;
    return EvalShape(t.kind, x, y, eye);
}

float Engine::EvalMorph(const Morph& m, float x, float y, int eye) const {
    if (m.phase == 1) {
        float a = 0;
        for (int i = 0; i < m.nfrom; i++) a += m.from[i].w * EvalTerm(m.from[i], x, y, eye);
        return a + (EvalTerm(m.drop, x, y, eye) - a) * m.e;
    }
    const float b = EvalShape(m.to, x, y, eye);
    if (m.pure || m.draw) return b;
    float a = 0;
    if (m.via) {
        a = EvalTerm(m.drop, x, y, eye);
    } else {
        for (int i = 0; i < m.nfrom; i++) a += m.from[i].w * EvalTerm(m.from[i], x, y, eye);
    }
    return a + (b - a) * m.q;
}

EYE_INLINE float Engine::Dist(int eye, float lx, float ly) const {
    const Xform& X = xf_[eye];
    const Morph& m = morph_[eye];
    ++sdf_evals_;
    const float x = lx * X.inv_sx, y = ly * X.inv_sy;
    // A settled (or stroke-drawing) shape is just itself: one call, no blend.
    if (m.phase != 1 && (m.pure || m.draw)) return EvalShape(m.to, x, y, eye) * X.sm;
    return EvalMorph(m, x, y, eye) * X.sm;
}

// Open and not blinking, the lids stay half a pixel or more clear of the
// idle capsule (whose field is never below |y| - 42): the capsule alone gives
// the same values, for a third of the work.
bool Engine::OpenIdle() const {
    return blink_ == 0 && std::max(lids_.inner, lids_.outer) + 84 * zone_lid_ <= .5f && lids_.bottom >= -1e-3f &&
           lids_.bottom <= .5f;
}

// Dist() for n points at once. The settled idle shape, most of what is ever on
// screen, runs as one tight loop; everything else still pays a call per point.
void Engine::DistN(int eye, int n, const float* lx, const float* ly, float* out) const {
    const Xform& X = xf_[eye];
    const Morph& m = morph_[eye];
    sdf_evals_ += n;
    const float isx = X.inv_sx, isy = X.inv_sy, sm = X.sm;
    if (m.phase == 1 || !(m.pure || m.draw)) {
        for (int k = 0; k < n; k++) out[k] = EvalMorph(m, lx[k] * isx, ly[k] * isy, eye) * sm;
        return;
    }
    if (m.to == static_cast<uint8_t>(Shape::kIdle)) {
        if (OpenIdle()) {
            BoxN(n, lx, ly, out, isx, inv_wd_, isy, sm);
        } else {
            IdleN(n, lx, ly, out, isx, isy, sm, IdleP{blink_, lids_.inner, lids_.outer, lids_.bottom, zone_lid_,
                                                      inv_wd_, eye == 0});
        }
        return;
    }
    for (int k = 0; k < n; k++) out[k] = EvalShape(m.to, lx[k] * isx, ly[k] * isy, eye) * sm;
}

void Engine::TermExtent(const Term& t, float& ex, float& ey) const {
    if (t.kind == kDropKind) {
        ex = fabsf(t.a) + t.c;
        ey = fabsf(t.b) + t.c;
        return;
    }
    ex = kExtent[t.kind][0];
    ey = kExtent[t.kind][1];
    if (t.kind == static_cast<uint8_t>(Shape::kIdle)) ex *= wd_;
}

// How far the eye's current shape reaches, and how steep its distance can get:
// blends of distance fields are not distance fields, and a spring overshoot
// (q > 1) pushes the blend past both shapes.
void Engine::MorphBounds(const Morph& m, float& ex, float& ey, float& lip) const {
    auto lip_of = [](const Term& t) { return t.kind == static_cast<uint8_t>(Shape::kIdle) ? 1.05f : 1.f; };
    const Term to = {m.to, 1, 0, 0, 0};
    TermExtent(to, ex, ey);
    auto add = [&](const Term& t) {
        float a, b;
        TermExtent(t, a, b);
        ex = std::max(ex, a);
        ey = std::max(ey, b);
    };
    float from_lip = 0;
    for (int i = 0; i < m.nfrom; i++) from_lip += fabsf(m.from[i].w) * lip_of(m.from[i]);
    float over = 0;
    if (m.phase == 1) {
        for (int i = 0; i < m.nfrom; i++) add(m.from[i]);
        add(m.drop);
        lip = (1 - m.e) * from_lip + m.e;
    } else if (m.pure || m.draw) {
        lip = lip_of(to);
    } else {
        if (m.via) {
            add(m.drop);
        } else {
            for (int i = 0; i < m.nfrom; i++) add(m.from[i]);
        }
        lip = fabsf(1 - m.q) * (m.via ? 1.f : from_lip) + fabsf(m.q) * lip_of(to);
        over = std::max(0.f, m.q - 1);
    }
    const float margin = 2 + over * 40;
    ex += margin;
    ey += margin;
}

// Freezes whatever an eye is showing right now into at most kMaxTerms weighted
// distance fields, so a new morph can start from it mid-flight.
int Engine::Snapshot(const Morph& m, Term* out) const {
    Term tmp[kMaxTerms + 2];
    int n = 0;
    auto add = [&](const Term& t, float w) {
        if (fabsf(w) < 1e-3f) return;
        for (int i = 0; i < n; i++) {
            const bool same = tmp[i].kind == t.kind &&
                              (t.kind != kDropKind || (tmp[i].a == t.a && tmp[i].b == t.b && tmp[i].c == t.c));
            if (same) {
                tmp[i].w += w;
                return;
            }
        }
        tmp[n] = t;
        tmp[n].w = w;
        n++;
    };
    const Term to = {m.to, 1, 0, 0, 0};
    if (m.phase == 1) {
        for (int i = 0; i < m.nfrom; i++) add(m.from[i], m.from[i].w * (1 - m.e));
        add(m.drop, m.e);
    } else if (m.pure || m.draw) {
        add(to, 1);
    } else if (m.via) {
        add(m.drop, 1 - m.q);
        add(to, m.q);
    } else {
        for (int i = 0; i < m.nfrom; i++) add(m.from[i], m.from[i].w * (1 - m.q));
        add(to, m.q);
    }
    // Heaviest first. Insertion sort: at most five terms, stable like the JS
    // mock-up's sort (and std::sort trips GCC's -Warray-bounds here).
    for (int i = 1; i < n; i++) {
        const Term t = tmp[i];
        int j = i;
        for (; j > 0 && fabsf(tmp[j - 1].w) < fabsf(t.w); j--) tmp[j] = tmp[j - 1];
        tmp[j] = t;
    }
    while (n > kMaxTerms) {
        tmp[0].w += tmp[n - 1].w;
        n--;
    }
    if (n == 0) {
        out[0] = to;
        return 1;
    }
    std::copy(tmp, tmp + n, out);
    return n;
}

void Engine::StartMorph(Shape s) {
    const uint8_t style = s == Shape::kWide ? kTrFast : kTrDrop;
    for (int e = 0; e < 2; e++) {
        Morph& m = morph_[e];
        Term snap[kMaxTerms];
        const int n = Snapshot(m, snap);
        std::copy(snap, snap + n, m.from);
        m.nfrom = n;
        m.to = static_cast<uint8_t>(s);
        m.drop = DropFor(s);
        // The right eye trails the left by a beat, so the pair reads as alive.
        m.t0 = t_ + ((e == 1 && style != kTrFast) ? .055 : 0);
        m.e = 0;
        m.q = 0;
        m.qv = 0;
        m.pure = false;
        m.style = style;
        m.via = style == kTrDrop;
        m.draw = (s == Shape::kDone || s == Shape::kFail) && style == kTrDrop;
        m.phase = style == kTrDrop ? 1 : 2;
        m.glint = -1;
    }
    if (s == Shape::kDone) prog_[0][0] = prog_[0][1] = 0;
    if (s == Shape::kFail) prog_[1][0] = prog_[1][1] = 0;
    if (s == Shape::kWide) {
        pk_ = .78f;
        pk_v_ = 0;
    }
    shape_ = s;
    enter_ = t_;
}

void Engine::StepMorphs(float dt) {
    const float h = dt / 2;
    for (int e = 0; e < 2; e++) {
        Morph& m = morph_[e];
        if (m.phase == 1) {
            const float u = static_cast<float>((t_ - m.t0) / .160);
            m.e = EaseInOut(Clamp(u, 0, 1));
            if (u >= 1) {
                m.phase = 2;
                m.q = 0;
                m.qv = 0;
            }
        } else if (t_ >= m.t0 && !m.pure) {
            const float w = m.style == kTrFast ? 26.f : (m.style == kTrSlow ? 6.f : 15.f);
            const float z = m.style == kTrFast ? .36f : (m.style == kTrSlow ? .95f : .58f);
            Spring(m.q, m.qv, 1, w, z, h);
            Spring(m.q, m.qv, 1, w, z, h);
            m.pure = fabsf(m.q - 1) < .002f && fabsf(m.qv) < .01f;
            if (m.pure) {
                m.q = 1;
                m.qv = 0;
            }
            if (m.via && m.glint < 0 && m.q > .92f) m.glint = t_;
        }
        if (m.draw) {
            const int k = m.to == static_cast<uint8_t>(Shape::kDone) ? 0 : 1;
            prog_[k][e] = m.phase == 2 ? Clamp(m.q, 0, 1) : 0;
            pop_[k][e] = std::max(0.f, m.q - 1);
        }
    }
}

// ---- behaviour --------------------------------------------------------------------------

float Engine::BlinkAmount() {
    if (!bl_on_) {
        if (t_ < next_blink_) return 0;
        bl_on_ = true;
        bl_start_ = t_;
        bl_dbl_ = Rand() < .18f;
        bl_half_ = Rand() < .1f;
    }
    // Fast close, a short hold, slow open: 70 / 30 / 160 ms.
    const float e = static_cast<float>((t_ - bl_start_) * 1000);
    const float mx = bl_half_ ? .55f : 1.f;
    if (e < 70) {
        const float q = e / 70;
        return q * q * q * mx;
    }
    if (e < 100) return mx;
    if (e < 260) return (1 - EaseOut((e - 100) / 160)) * mx;
    if (bl_dbl_) {
        bl_start_ = t_;
        bl_dbl_ = false;
        bl_half_ = false;
        return 0;
    }
    bl_on_ = false;
    next_blink_ = t_ + Rand(2, 5);
    return 0;
}

// Where to look: wandering glances and the odd zoned-out stare when idle,
// otherwise the held point with a slow sway.
void Engine::Direct() {
    if (shape_ == Shape::kIdle && wander_) {
        if (deterministic_) {
            tx_ = hold_x_;
            ty_ = hold_y_;
            zoning_ = false;
            return;
        }
        if (t_ >= zone_at_) {
            zoning_ = !zoning_;
            zone_at_ = t_ + (zoning_ ? Rand(1.3f, 1.9f) : Rand(6, 9));
        }
        if (!zoning_ && t_ >= pick_at_) {
            if (Rand() < .3f) {
                tx_ = 0;
                ty_ = 0;
            } else {
                tx_ = Rand(-26, 26);
                ty_ = Rand(-16, 16);
            }
            pick_at_ = t_ + Rand(1.2f, 3.2f);
            picked_ = t_;
        }
    } else {
        zoning_ = false;
        tx_ = hold_x_ + sinf(Wrap(t_ * .7)) * 3;
        ty_ = hold_y_ + cosf(Wrap(t_ * .9)) * 2;
    }
}

void Engine::StepGaze(float dt) {
    if (!deterministic_ && picked_ != last_pick_seen_) {
        last_pick_seen_ = picked_;
        // A big glance often comes with a blink.
        if (shape_ == Shape::kIdle && !bl_on_ && Len(tx_ - gx_, ty_ - gy_) > 18 && Rand() < .35f) {
            next_blink_ = t_;
        }
    }
    float ax = tx_, ay = ty_;
    // Land a little short, then settle onto the target.
    if (shape_ == Shape::kIdle && t_ - picked_ < .12) {
        ax *= .9f;
        ay *= .9f;
    }
    if (deterministic_) {
        mx_ = my_ = 0;
    } else if (t_ >= next_micro_) {
        mx_ = Rand(-1, 1);
        my_ = Rand(-.7f, .7f);
        next_micro_ = t_ + Rand(.35f, 1.1f);
    }
    const float h = dt / 2;
    const float zone_target = (zoning_ && shape_ == Shape::kIdle) ? .26f : 0.f;
    for (int i = 0; i < 2; i++) {
        // Eyes snap, the face follows slowly.
        Spring(ex_, evx_, ax + mx_, 34, .78f, h);
        Spring(ey_, evy_, ay + my_, 34, .78f, h);
        Spring(fx_, fvx_, ax, 5.5f, .95f, h);
        Spring(fy_, fvy_, ay, 5.5f, .95f, h);
        Spring(zone_lid_, zone_lid_v_, zone_target, 16, .85f, h);
    }
    gx_ = .62f * ex_ + .38f * fx_;
    gy_ = .62f * ey_ + .38f * fy_;
}

void Engine::StepLids(float dt) {
    const float h = dt / 2;
    for (int i = 0; i < 2; i++) {
        Spring(lids_.outer, lids_v_.outer, lids_target_.outer, 14, .8f, h);
        Spring(lids_.inner, lids_v_.inner, lids_target_.inner, 14, .8f, h);
        Spring(lids_.bottom, lids_v_.bottom, lids_target_.bottom, 14, .8f, h);
    }
}

void Engine::StepShapeAnim(float dt) {
    const double t = t_;
    const float bp = static_cast<float>(t * 1.25 - floor(t * 1.25));
    beat_ = expf(-(bp - .1f) * (bp - .1f) / .002f) + .6f * expf(-(bp - .32f) * (bp - .32f) / .002f);

    // Star turns a fifth of a turn at a time.
    const double nn = floor(t / 1.6);
    const float f = Clamp(static_cast<float>((t - nn * 1.6) / .55), 0, 1);
    const double sa = (fmod(nn, 5.0) + EaseInOut(f)) * kTauD / 5;
    sc_ = cosf(Wrap(sa));
    ss_ = sinf(Wrap(sa));

    // Loading arcs: uneven speed, breathing aperture.
    const float s24 = sinf(Wrap(t * 2.4)), c24 = cosf(Wrap(t * 2.4));
    const float th = Wrap(t * 4.2 + .9 * s24);
    const float ap = .35f + 1.9f * (.5f - .5f * c24);
    rc_ = cosf(th);
    rs_ = sinf(th);
    ta_ = sinf(ap);
    tb_ = cosf(ap);

    float kx = 1, ky = 1, yo = 0, en = 1, shx = 0;
    const float age_ms = static_cast<float>((t - enter_) * 1000);
    switch (shape_) {
        case Shape::kIdle: {
            const float b = sinf(Wrap(t * 1.5));
            kx += .012f * b;
            ky += .012f * b;
            en += .06f * b;
            break;
        }
        case Shape::kHappy: {
            const float s = fabsf(sinf(Wrap(t * 3.6)));
            const float c = (1 - s) * (1 - s) * (1 - s);
            yo = -5 * powf(s, .8f);
            kx += .05f * c;
            ky -= .06f * c;
            break;
        }
        case Shape::kLove:
            kx += .05f * beat_;
            ky += .05f * beat_;
            en += .2f * beat_;
            break;
        case Shape::kStar: {
            const float s = std::max(0.f, sinf(Wrap(t * 2.2)));
            const float s2 = s * s, tw = .05f * s2 * s2 * s2;
            kx += tw;
            ky += tw;
            en += .2f;
            break;
        }
        case Shape::kWide: {
            Spring(pk_, pk_v_, 1, 24, .32f, dt);
            kx *= pk_;
            ky *= pk_;
            const float a = (age_ms - 70) / 110;
            yo = -6 * expf(-a * a);
            shx = sinf(Wrap(t * 38)) * 3.2f * expf(-age_ms / 320);
            en += .25f;
            break;
        }
        default:
            break;
    }
    // Fast glances stretch the eyes along the motion a touch.
    const float vx = std::min(1.f, fabsf(evx_) / 700), vy = std::min(1.f, fabsf(evy_) / 700);
    kx *= 1 + .06f * vx - .04f * vy;
    ky *= 1 + .06f * vy - .04f * vx;
    kx_ = kx;
    ky_ = ky;
    yo_ = yo;
    en_ = en;
    shx_ = shx;
}

float Engine::RingRadius() const {
    return std::min(104.f, (strip_y1_ - strip_y0_) * .5f - 8);
}

// The face fits the strip it is given, and shrinks to make room for the ring.
void Engine::StepLayout(float dt) {
    const float strip_h = static_cast<float>(strip_y1_ - strip_y0_);
    const float mid_target = (strip_y0_ + strip_y1_) * .5f;
    const float fit = Clamp((strip_h - 16) / 100, .4f, 1);
    const float rr = RingRadius();
    const bool ring_on = ring_ != Ring::kOff && rr >= 70;
    const float target = ring_on ? std::min(fit, (rr - 22) / 90) : fit;
    if (!layout_init_) {
        scale_ = target;
        mid_ = mid_target;
        layout_init_ = true;
    }
    const float h = dt / 2;
    for (int i = 0; i < 2; i++) {
        Spring(scale_, scale_v_, target, 10, .9f, h);
        Spring(mid_, mid_v_, mid_target, 10, .9f, h);
    }
    ring_alpha_ += ((ring_on ? 1.f : 0.f) - ring_alpha_) * (1 - expf(-dt * 8));
    if (!ring_on && ring_alpha_ < .005f) ring_alpha_ = 0;
    voice_lv_ += ((voice_ ? 1.f : .3f) - voice_lv_) * (1 - expf(-dt * 10));
}

// Little twinkles thrown off by the hearts on each beat.
void Engine::StepSparks() {
    int n = 0;
    for (int i = 0; i < nsparks_; i++) {
        if (t_ - sparks_[i].t0 < .65) sparks_[n++] = sparks_[i];
    }
    nsparks_ = n;
    if (deterministic_ || shape_ != Shape::kLove) return;
    const Morph& m = morph_[1];
    if (m.phase != 2 || m.q <= .7f) return;
    if (beat_ > .9f && !beat_on_) {
        beat_on_ = true;
        for (int e = 0; e < 2; e++) {
            for (int j = 0; j < 2 && nsparks_ < 8; j++) {
                const float a = Rand(-2.7f, -.45f), rr = Rand(36, 46) * scale_;
                sparks_[nsparks_++] = {xf_[e].cx + cosf(a) * rr, xf_[e].cy + sinf(a) * rr, scale_, t_};
            }
        }
    } else if (beat_ < .4f) {
        beat_on_ = false;
    }
}

void Engine::UpdateColors(float dt) {
    float ta[3], tb[3];
    const PaletteDef& p = kPalettes[static_cast<int>(palette_)];
    ToOklab(p.top, ta);
    ToOklab(p.bottom, tb);
    if (!colors_init_) {
        std::copy(ta, ta + 3, oa_);
        std::copy(tb, tb + 3, ob_);
        colors_init_ = true;
        color_moving_ = true;
    } else {
        const float kc = 1 - expf(-dt * 6);
        float left = 0;
        for (int c = 0; c < 3; c++) {
            oa_[c] += (ta[c] - oa_[c]) * kc;
            ob_[c] += (tb[c] - ob_[c]) * kc;
            left = std::max(left, std::max(fabsf(ta[c] - oa_[c]), fabsf(tb[c] - ob_[c])));
        }
        const bool moving = left >= 2e-4f;
        if (!moving && !color_moving_) return;
        color_moving_ = moving;
    }
    FromOklab(ob_, cb_);
    for (int i = 0; i < 64; i++) {
        const float v = i / 63.f, s = v * v * (3 - 2 * v);
        float o[3];
        for (int c = 0; c < 3; c++) o[c] = oa_[c] + (ob_[c] - oa_[c]) * s;
        FromOklab(o, lut_[i]);
    }
}

void Engine::ComputeXforms() {
    const float gxn = Clamp(gx_ / 30, -1, 1);
    const float s = scale_;
    for (int e = 0; e < 2; e++) {
        const float side = e == 0 ? -1.f : 1.f;
        Xform& X = xf_[e];
        X.cx = w_ * .5f + (side * kEyeDx + gx_ + shx_) * s;
        X.cy = mid_ + (gy_ + yo_) * s;
        // Perspective: the eye on the side we look towards grows a little.
        const float ps = 1 + .06f * gxn * side;
        X.sx = kx_ * ps * s;
        X.sy = ky_ * ps * s;
        X.sm = std::min(kx_, ky_) * ps * s;
        X.inv_sx = 1 / X.sx;
        X.inv_sy = 1 / X.sy;
        float ex, ey;
        MorphBounds(morph_[e], ex, ey, X.lip);
        // Past the shape's box the distance grows at least this fast.
        const float hx = ex * X.sx + kGlw * X.sx / X.sm + 1.5f;
        const float hy = ey * X.sy + kGlw * X.sy / X.sm + 1.5f;
        X.x0 = std::max(0, static_cast<int>(floorf(X.cx - hx)));
        X.x1 = std::min(w_, static_cast<int>(ceilf(X.cx + hx)));
        X.y0 = std::max(strip_y0_, static_cast<int>(floorf(X.cy - hy)));
        X.y1 = std::min(strip_y1_, static_cast<int>(ceilf(X.cy + hy)));
    }
}

// Everything that moves the picture, in roughly pixels: a frame is drawn only
// once something has moved by more than a fraction of a pixel.
void Engine::Signature(float* sig) const {
    int n = 0;
    const float s = scale_;
    for (int e = 0; e < 2; e++) {
        const Xform& X = xf_[e];
        const Morph& m = morph_[e];
        sig[n++] = X.cx;
        sig[n++] = X.cy;
        sig[n++] = X.sx * 50;
        sig[n++] = X.sy * 50;
        sig[n++] = (m.phase == 1 ? m.e : 1 + m.q) * 30 * s;
        sig[n++] = prog_[0][e] * 60 * s;
        sig[n++] = prog_[1][e] * 60 * s;
        sig[n++] = pop_[0][e] * 8 * s;
        sig[n++] = pop_[1][e] * 8 * s;
    }
    sig[n++] = blink_ * 42 * s;
    sig[n++] = zone_lid_ * 84 * s;
    sig[n++] = wd_ * 35 * s;
    sig[n++] = lids_.outer * s;
    sig[n++] = lids_.inner * s;
    sig[n++] = lids_.bottom * s;
    sig[n++] = en_ * 20;
    const bool star = shape_ == Shape::kStar, think = shape_ == Shape::kThink;
    sig[n++] = star ? sc_ * 35 * s : 0;
    sig[n++] = star ? ss_ * 35 * s : 0;
    sig[n++] = think ? rc_ * 22 * s : 0;
    sig[n++] = think ? rs_ * 22 * s : 0;
    sig[n++] = think ? ta_ * 22 * s : 0;
    sig[n++] = think ? tb_ * 22 * s : 0;
    for (int c = 0; c < 3; c++) {
        sig[n++] = oa_[c] * 250;
        sig[n++] = ob_[c] * 250;
    }
    sig[n++] = ring_alpha_ * 30;
    sig[n++] = s * 100;
    sig[n++] = mid_;
}

bool Engine::Tick(uint32_t now_ms) {
    if (!started_) {
        started_ = true;
        last_ms_ = now_ms;
    }
    uint32_t ms = now_ms - last_ms_;
    last_ms_ = now_ms;
    // Slow callers (power save ticks at 5 fps) still get real time, in steps the
    // springs can take; after a long stall, skip ahead rather than replay it.
    ms = std::min<uint32_t>(ms, 1000);
    bool need = false;
    while (ms > 50) {
        need |= Advance(.05);
        ms -= 50;
    }
    return Advance(ms / 1000.0) || need;
}

bool Engine::Advance(double dt_in) {
    const double dtd = dt_in < 0 ? 0 : (dt_in > .05 ? .05 : dt_in);
    const float dt = static_cast<float>(dtd);
    t_ += dtd;

    if (pending_shape_ >= 0) {
        StartMorph(static_cast<Shape>(pending_shape_));
        pending_shape_ = -1;
    }
    Direct();
    StepMorphs(dt);
    UpdateColors(dt);
    blink_ = (shape_ == Shape::kIdle && blinking_ && !deterministic_) ? BlinkAmount() : 0;
    StepGaze(dt);
    wd_ = 1 + .12f * blink_;
    inv_wd_ = 1 / wd_;
    StepLids(dt);
    StepShapeAnim(dt);
    StepLayout(dt);
    StepSparks();
    ComputeXforms();

    Signature(cur_sig_);
    bool need = full_clear_ || !sig_valid_ || t_ - last_draw_t_ > 1.0;
    for (int i = 0; i < kSig && !need; i++) need = fabsf(cur_sig_[i] - sig_[i]) > .12f;
    for (const Morph& m : morph_) {
        if (m.glint >= 0 && t_ - m.glint < .55) need = true;
    }
    if (nsparks_ > 0) need = true;
    const bool ring_vis = ring_alpha_ >= .02f && ring_shown_ != Ring::kOff;
    if (ring_vis != ring_was_visible_) need = true;
    // The ring keeps moving; 30 fps is enough for it.
    if (ring_vis && t_ - last_draw_t_ >= .03) need = true;
    return need;
}

// ---- pixels -----------------------------------------------------------------------------

template <bool k888>
EYE_INLINE void Engine::ReadPixel(int x, int y, float& r, float& g, float& b) const {
    if (k888) {
        const uint8_t* p = static_cast<const uint8_t*>(canvas_) + (y * stride_ + x) * 3;
        r = p[0];
        g = p[1];
        b = p[2];
    } else {
        // The inverse of the dithered quantiser below, so reading a pixel back
        // and writing it again leaves it as it was.
        const uint16_t v = static_cast<const uint16_t*>(canvas_)[y * stride_ + x];
        r = static_cast<float>((v >> 11) << 3);
        g = static_cast<float>(((v >> 5) & 63) << 2);
        b = static_cast<float>((v & 31) << 3);
    }
}

template <bool k888>
EYE_INLINE void Engine::WritePixel(int x, int y, float r, float g, float b) {
    if (k888) {
        uint8_t* p = static_cast<uint8_t*>(canvas_) + (y * stride_ + x) * 3;
        p[0] = static_cast<uint8_t>(Clamp(r + .5f, 0, 255));
        p[1] = static_cast<uint8_t>(Clamp(g + .5f, 0, 255));
        p[2] = static_cast<uint8_t>(Clamp(b + .5f, 0, 255));
    } else {
        static_cast<uint16_t*>(canvas_)[y * stride_ + x] = Pack565(r, g, b, g_dither[y & 3][x & 3]);
    }
}

template <bool k888>
void Engine::ClearRect(const Rect& rc) {
    for (int y = rc.y0; y < rc.y1; y++) {
        if (k888) {
            uint8_t* p = static_cast<uint8_t*>(canvas_) + (y * stride_ + rc.x0) * 3;
            const uint8_t r = static_cast<uint8_t>(bg_[0]), g = static_cast<uint8_t>(bg_[1]),
                          b = static_cast<uint8_t>(bg_[2]);
            for (int x = rc.x0; x < rc.x1; x++, p += 3) {
                p[0] = r;
                p[1] = g;
                p[2] = b;
            }
        } else {
            uint16_t* p = static_cast<uint16_t*>(canvas_) + y * stride_;
            std::fill(p + rc.x0, p + rc.x1, bg565_);
        }
    }
}

// ---- one eye ----------------------------------------------------------------------------

// The box was cleared this frame, so under the glow and the rim is the
// background -- except where eye 1 overlaps eye 0's box and its glow.
template <bool k888>
EYE_INLINE void Engine::Under(const PixCtx& c, int px, int py, float& r, float& g, float& b) const {
    if (c.overlap && px < c.ox1 && py >= c.oy0 && py < c.oy1) {
        ReadPixel<k888>(px, py, r, g, b);
    } else {
        r = c.bg[0];
        g = c.bg[1];
        b = c.bg[2];
    }
}

EYE_INLINE float Engine::Glint(const PixCtx& c, int px, int py) const {
    const float q = ((px + .5f - c.cx) + (py + .5f - c.cy)) * c.inv_s * .7f - c.gp;
    return 140.f * c.ga * GlintAt(q);
}

// The flat middle: one colour per row, plus the highlight per pixel.
EYE_INLINE void Engine::FlatBase(const PixCtx& c, int py, float* fb) const {
    const float* col = lut_[row_vi_[py]];
    fb[0] = col[0] * c.flat_br + c.flat_ad;
    fb[1] = col[1] * c.flat_br + c.flat_ad;
    fb[2] = col[2] * c.flat_br + c.flat_ad;
}

// Outside the edge: only the glow, added onto what is there.
template <bool k888>
EYE_INLINE void Engine::PutGlow(const PixCtx& c, int px, int py, float d) {
    const float gl = GlowAt(c.glow, d) * c.en;
    if (gl * c.cmax < .25f) return;
    float pr, pg, pb;
    Under<k888>(c, px, py, pr, pg, pb);
    WritePixel<k888>(px, py, pr + c.cb[0] * gl, pg + c.cb[1] * gl, pb + c.cb[2] * gl);
}

// The same where nothing but the background is under it, from the table;
// `fb` is the distance in table steps (1/16 px).
template <bool k888>
EYE_INLINE void Engine::PutGlowBin(const PixCtx& c, int px, int py, float fb) {
    if constexpr (k888) {
        // The desktop reference path, unbinned.
        const float gl = GlowAt(c.glow, fb * (1.f / kGlowRes)) * c.en;
        if (gl * c.cmax >= .25f && fb < kGlw * kGlowRes) {
            WritePixel<true>(px, py, c.bg[0] + c.cb[0] * gl, c.bg[1] + c.cb[1] * gl, c.bg[2] + c.cb[2] * gl);
        }
    } else {
        static_cast<uint16_t*>(canvas_)[py * stride_ + px] =
            Swar565(g_glow565[GlowBin(fb)] + g_dswar[py & 3][px & 3]);
    }
}

template <bool k888>
EYE_INLINE void Engine::PutFlat(const PixCtx& c, int px, int py, const float* fb, float rh) {
    float ad = rh * col_hl_[px];
    if (c.ga > 0) ad += Glint(c, px, py);
    WritePixel<k888>(px, py, fb[0] + ad, fb[1] + ad, fb[2] + ad);
}

// A lit pixel: inside the edge (d < .5) and within the bevel (d > -kBev).
// (gx, gy) is the distance gradient, any length.
EYE_INLINE Engine::Lit Engine::Light(const PixCtx& c, float d, float gx, float gy) const {
    Lit L;
    LightOf(LightV{c.lx, c.ly, c.lz, c.hx, c.hy, c.hz}, d, gx, gy, L.br, L.sp, L.rm);
    return L;
}

template <bool k888>
EYE_INLINE void Engine::PutLit(const PixCtx& c, int px, int py, float d, float gx, float gy) {
    const Lit L = Light(c, d, gx, gy);
    float ad = L.sp + row_hl_[py] * col_hl_[px];
    if (c.ga > 0) ad += Glint(c, px, py);
    const float* col = lut_[row_vi_[py]];
    float r = col[0] * L.br + ad, g = col[1] * L.br + ad, b = col[2] * L.br + ad;
    // The rim: nothing added where there is none.
    r += c.cb[0] * L.rm;
    g += c.cb[1] * L.rm;
    b += c.cb[2] * L.rm;
    if (d > -.5f) {
        // The anti-aliased edge: the rest of the pixel is glow over what is behind.
        const float cov = .5f - d, ic = 1 - cov;
        const float gl = GlowAt(c.glow, std::max(d, 0.f)) * c.en;
        float pr, pg, pb;
        Under<k888>(c, px, py, pr, pg, pb);
        r = r * cov + (pr + c.cb[0] * gl) * ic;
        g = g * cov + (pg + c.cb[1] * gl) * ic;
        b = b * cov + (pb + c.cb[2] * gl) * ic;
    }
    WritePixel<k888>(px, py, r, g, b);
}

void Engine::Phase(int k) {
    if (prof_now_ == nullptr) return;
    const uint32_t t = prof_now_();
    prof_phase_[ph_cur_] += t - ph_t_;
    ph_t_ = t;
    ph_cur_ = k;
}

// Evaluates the unset grid nodes in columns ia..ib, rows ja..jb.
EYE_NOINLINE void Engine::FillNodes(const PixCtx& c, int ia, int ib, int ja, int jb) {
    ia = std::max(ia, 0);
    ib = std::min(ib, c.nw - 1);
    ja = std::max(ja, 0);
    jb = std::min(jb, c.nh - 1);
    float* __restrict G = grid_;
    float* __restrict qx = scratch_.qx;
    float* __restrict qy = scratch_.qy;
    int* __restrict qi = scratch_.qi;
    const int nw = c.nw, x0 = c.x0, y0 = c.y0;
    const float cx = c.cx, cy = c.cy;
    int n = 0;
    for (int j = ja; j <= jb; j++) {
        const float* row = G + j * nw;
        const float y = y0 + 2 * j + .5f - cy;
        for (int i = ia; i <= ib; i++) {
            if (row[i] != kUnset) continue;
            qx[n] = x0 + 2 * i + .5f - cx;
            qy[n] = y;
            qi[n] = j * nw + i;
            n++;
        }
    }
    if (n == 0) return;
    const float* __restrict qd = scratch_.qd;
    DistN(c.eye, n, qx, qy, scratch_.qd);
    for (int k = 0; k < n; k++) G[qi[k]] = qd[k];
}

template <bool k888>
EYE_NOINLINE void Engine::FlatBlock(const PixCtx& c, const Blk& b) {
    const int bx = b.bx, bw = b.bw;
    for (int py = b.by; py < b.by + b.bh; py++) {
        float fb[3];
        FlatBase(c, py, fb);
        const float rh = row_hl_[py];
        if (k888 || c.ga > 0) {
            for (int px = bx; px < bx + bw; px++) PutFlat<k888>(c, px, py, fb, rh);
            continue;
        }
        if constexpr (!k888) {
            uint16_t* __restrict o = static_cast<uint16_t*>(canvas_) + py * stride_ + bx;
            const float* __restrict ch = col_hl_ + bx;
            const float* __restrict th = scratch_.dth[py & 3];
            const float r = fb[0], g = fb[1], bl = fb[2];
            for (int lx = 0; lx < bw; lx++) {
                const float ad = rh * ch[lx];
                o[lx] = Pack565(r + ad, g + ad, bl + ad, th[lx]);
            }
        }
    }
    px_flat_ += bw * b.bh;
}

// All glow. Blocks start on even pixels, so even columns and rows sit on
// nodes and odd ones halfway between.
template <bool k888>
EYE_NOINLINE void Engine::GlowBlock(const PixCtx& c, const Blk& b) {
    const int nw = c.nw, y0 = c.y0, bx = b.bx, bw = b.bw;
    const float* const G = grid_ + b.i0;
    for (int ly = 0; ly < b.bh; ly++) {
        const int py = b.by + ly;
        const float* __restrict r0 = G + ((py - y0) >> 1) * nw;
        const float* __restrict r1 = r0 + nw;
        const float wy = (ly & 1) ? 1.f : 0.f;
        const float f0 = (ly & 1) ? .5f * kGlowRes : 1.f * kGlowRes;
        const float f1 = .5f * f0;
        float a = r0[0] + r1[0] * wy;
        if constexpr (k888) {
            for (int lx = 0; lx < bw; lx += 2) {
                const int k = lx >> 1;
                const float a2 = r0[k + 1] + r1[k + 1] * wy;
                PutGlowBin<true>(c, bx + lx, py, a * f0);
                if (lx + 1 < bw) PutGlowBin<true>(c, bx + lx + 1, py, (a + a2) * f1);
                a = a2;
            }
        } else {
            uint16_t* __restrict o = static_cast<uint16_t*>(canvas_) + py * stride_ + bx;
            const uint32_t* __restrict dl = scratch_.dsw[py & 3];
            const uint32_t* __restrict tab = g_glow565;
            if (bw == kBlock) {
#pragma GCC unroll 4
                for (int k = 0; k < kBlock / 2; k++) {
                    const float a2 = r0[k + 1] + r1[k + 1] * wy;
                    o[2 * k] = Swar565(tab[GlowBin(a * f0)] + dl[2 * k]);
                    o[2 * k + 1] = Swar565(tab[GlowBin((a + a2) * f1)] + dl[2 * k + 1]);
                    a = a2;
                }
            } else {
                for (int lx = 0; lx < bw; lx += 2) {
                    const int k = lx >> 1;
                    const float a2 = r0[k + 1] + r1[k + 1] * wy;
                    o[lx] = Swar565(tab[GlowBin(a * f0)] + dl[lx]);
                    if (lx + 1 < bw) o[lx + 1] = Swar565(tab[GlowBin((a + a2) * f1)] + dl[lx + 1]);
                    a = a2;
                }
            }
        }
    }
    px_glow_ += bw * b.bh;
}

// Sorts a block's pixels by the grid's distance: flat and glow ones are
// painted on the spot, lit ones queue for the finer work. Returns how many.
// kFast: RGB565 with nothing but the background under the glow, and no glint.
template <bool k888, bool kFast>
EYE_NOINLINE int Engine::SortBlock(const PixCtx& c, const Blk& b) {
    constexpr bool fast = kFast && !k888;
    uint8_t* __restrict lp = scratch_.lp;
    float* __restrict ld = scratch_.dv;
    const float* const G = grid_ + b.i0;
    const int nw = c.nw, y0 = c.y0, bx = b.bx, bw = b.bw;
    const float flat_max = c.flat_max, glow_min = c.glow_min;
    int nlit = 0, nf = 0, ng = 0;
    for (int ly = 0; ly < b.bh; ly++) {
        const int py = b.by + ly;
        const float* __restrict r0 = G + ((py - y0) >> 1) * nw;
        const float* __restrict r1 = r0 + nw;
        const float fy = (ly & 1) ? .5f : 0.f;
        // The row's distances, node by node and halfway between.
        float dr[kBlock];
        float a = r0[0] + (r1[0] - r0[0]) * fy;
        for (int k = 0; 2 * k < bw; k++) {
            const float a2 = r0[k + 1] + (r1[k + 1] - r0[k + 1]) * fy;
            dr[2 * k] = a;
            dr[2 * k + 1] = (a + a2) * .5f;
            a = a2;
        }
        float fb[3];
        FlatBase(c, py, fb);
        const float rh = row_hl_[py];
        if constexpr (fast) {
            uint16_t* __restrict o = static_cast<uint16_t*>(canvas_) + py * stride_ + bx;
            const float* __restrict ch = col_hl_ + bx;
            const float* __restrict th = scratch_.dth[py & 3];
            const uint32_t* __restrict dl = scratch_.dsw[py & 3];
            for (int lx = 0; lx < bw; lx++) {
                const float di = dr[lx];
                if (di > kGlw) continue;
                if (di <= flat_max) {
                    const float ad = rh * ch[lx];
                    o[lx] = Pack565(fb[0] + ad, fb[1] + ad, fb[2] + ad, th[lx]);
                    nf++;
                } else if (di >= glow_min) {
                    o[lx] = Swar565(g_glow565[GlowBin(di * kGlowRes)] + dl[lx]);
                    ng++;
                } else {
                    lp[nlit] = static_cast<uint8_t>(ly << 3 | lx);
                    ld[nlit] = di;
                    nlit++;
                }
            }
        } else {
            for (int lx = 0; lx < bw; lx++) {
                const float di = dr[lx];
                if (di > kGlw) continue;
                const int px = bx + lx;
                if (di <= flat_max) {
                    PutFlat<k888>(c, px, py, fb, rh);
                    nf++;
                } else if (di >= glow_min) {
                    if (b.ovl) {
                        PutGlow<k888>(c, px, py, di);
                    } else {
                        PutGlowBin<k888>(c, px, py, di * kGlowRes);
                    }
                    ng++;
                } else {
                    lp[nlit] = static_cast<uint8_t>(ly << 3 | lx);
                    ld[nlit] = di;
                    nlit++;
                }
            }
        }
    }
    px_flat_ += nf;
    px_glow_ += ng;
    return nlit;
}

// At a block's nodes, from one node further out: the gradient by central
// differences, and second differences.
EYE_NOINLINE void Engine::NodeGrads(const PixCtx& c, const Blk& b) {
    const float* const G = grid_;
    const int nw = c.nw, nh = c.nh;
    float(*__restrict gxn)[kBlock / 2 + 1] = scratch_.gxn;
    float(*__restrict gyn)[kBlock / 2 + 1] = scratch_.gyn;
    float(*__restrict cxn)[kBlock / 2 + 1] = scratch_.cxn;
    float(*__restrict cyn)[kBlock / 2 + 1] = scratch_.cyn;
    float(*__restrict sdn)[kBlock / 2 + 1] = scratch_.sdn;
    for (int j = b.j0; j <= b.j1; j++) {
        const float* row = G + j * nw;
        const float* up = G + std::max(j - 1, 0) * nw;
        const float* dn = G + std::min(j + 1, nh - 1) * nw;
        const bool jin = j > 0 && j < nh - 1;
        const float sy = jin ? .25f : .5f;
        const int cj = j - b.j0;
        for (int i = b.i0; i <= b.i1; i++) {
            const bool iin = i > 0 && i < nw - 1;
            const float c0 = row[i], l = row[std::max(i - 1, 0)], r = row[std::min(i + 1, nw - 1)];
            const float u = up[i], w = dn[i];
            const float dxx = iin ? l + r - 2 * c0 : kRough;
            const float dyy = jin ? u + w - 2 * c0 : kRough;
            const int ci = i - b.i0;
            gxn[cj][ci] = (r - l) * (iin ? .25f : .5f);
            gyn[cj][ci] = (w - u) * sy;
            cxn[cj][ci] = dxx;
            cyn[cj][ci] = dyy;
            sdn[cj][ci] = std::max(fabsf(dxx), fabsf(dyy));
        }
    }
}

// The lit pixels' gradients, interpolated: x half a pixel right, y half a
// pixel down, where the mock-up's forward differences sit. Where the cell is
// rough the rim takes its exact distance, and a gradient that comes out short
// its exact forward difference.
EYE_NOINLINE void Engine::LitInterp(const PixCtx& c, const Blk& b, int nlit) {
    constexpr int kBatch = Scratch::kBatch;
    const uint8_t* __restrict lp = scratch_.lp;
    uint8_t* __restrict lf = scratch_.cls;
    float* __restrict ld = scratch_.dv;
    float* __restrict lgx = scratch_.gxv;
    float* __restrict lgy = scratch_.gyv;
    const float(*__restrict gxn)[kBlock / 2 + 1] = scratch_.gxn;
    const float(*__restrict gyn)[kBlock / 2 + 1] = scratch_.gyn;
    const float(*__restrict cxn)[kBlock / 2 + 1] = scratch_.cxn;
    const float(*__restrict cyn)[kBlock / 2 + 1] = scratch_.cyn;
    const float(*__restrict sdn)[kBlock / 2 + 1] = scratch_.sdn;
    float* __restrict exf = &scratch_.ex[0][0];
    uint8_t(*__restrict need)[kBlock + 1] = scratch_.need;
    float* __restrict qx = scratch_.qx;
    float* __restrict qy = scratch_.qy;
    int* __restrict qi = scratch_.qi;
    const int eye = c.eye, bx = b.bx, by = b.by;
    const float cx = c.cx, cy = c.cy, rim_min = c.rim_min;
    std::fill(&need[0][0], &need[0][0] + (kBlock + 1) * (kBlock + 1), 0);
    int nq = 0;
    auto want = [&](int qxl, int qyl) EYE_LAMBDA {
        if (need[qyl][qxl]) return;
        need[qyl][qxl] = 1;
        qx[nq] = bx + qxl + .5f - cx;
        qy[nq] = by + qyl + .5f - cy;
        qi[nq] = qyl * (kBlock + 1) + qxl;
        nq++;
    };
    auto flush = [&]() EYE_LAMBDA {
        DistN(eye, nq, qx, qy, scratch_.qd);
        const float* qd = scratch_.qd;
        for (int k = 0; k < nq; k++) exf[qi[k]] = qd[k];
        nq = 0;
    };
    for (int k = 0; k < nlit; k++) {
        const int lx = lp[k] & 7, ly = lp[k] >> 3;
        const int ii = lx >> 1, jj = ly >> 1;
        const bool oddx = (lx & 1) != 0, oddy = (ly & 1) != 0;
        const float ux = oddx ? .75f : .25f, fx = oddx ? .5f : 0.f;
        const float vy = oddy ? .75f : .25f, fy = oddy ? .5f : 0.f;
        const float* a = gxn[jj] + ii;
        const float* a2 = gxn[jj + 1] + ii;
        const float gx = (a[0] + (a[1] - a[0]) * ux) * (1 - fy) + (a2[0] + (a2[1] - a2[0]) * ux) * fy;
        const float* e = gyn[jj] + ii;
        const float* e2 = gyn[jj + 1] + ii;
        const float gy = (e[0] + (e[1] - e[0]) * fx) * (1 - vy) + (e2[0] + (e2[1] - e2[0]) * fx) * vy;
        lgx[k] = gx;
        lgy[k] = gy;
        float d = ld[k];
        uint8_t f = 0;
        const float rough = std::max(std::max(sdn[jj][ii], sdn[jj][ii + 1]),
                                     std::max(sdn[jj + 1][ii], sdn[jj + 1][ii + 1]));
        if (rough < kSmooth) {
            // Between nodes, straight-line interpolation overshoots a
            // curving field by an eighth of its second difference.
            if (oddx && oddy) {
                d -= (cxn[jj][ii] + cxn[jj][ii + 1] + cxn[jj + 1][ii] + cxn[jj + 1][ii + 1] +
                      cyn[jj][ii] + cyn[jj][ii + 1] + cyn[jj + 1][ii] + cyn[jj + 1][ii + 1]) *
                     (1.f / 32);
            } else if (oddx) {
                d -= (cxn[jj][ii] + cxn[jj][ii + 1]) * (1.f / 16);
            } else if (oddy) {
                d -= (cyn[jj][ii] + cyn[jj + 1][ii]) * (1.f / 16);
            }
            ld[k] = d;
        } else if (d > rim_min) {
            f = kExactD;
        }
        // A short gradient -- a ridge, or the saddle of a blend
        // mid-morph -- points anywhere: take the exact one.
        if (gx * gx + gy * gy < kMinGrad2) f = kExactD | kExactG;
        if (f != 0) {
            want(lx, ly);
            if (f & kExactG) {
                want(lx + 1, ly);
                want(lx, ly + 1);
            }
            if (nq > kBatch - 3) flush();
        }
        lf[k] = f;
    }
    if (nq > 0) flush();
}

// A settled shape's field is cheap next to the interpolation above, so its
// lit pixels take their exact distance and forward differences, as the
// mock-up does: each needs the point itself and those right of and below it.
EYE_NOINLINE void Engine::LitExact(const PixCtx& c, const Blk& b, int nlit) {
    constexpr int kBatch = Scratch::kBatch;
    const uint8_t* __restrict lp = scratch_.lp;
    uint8_t* __restrict lf = scratch_.cls;
    float* __restrict exf = &scratch_.ex[0][0];
    uint8_t(*__restrict need)[kBlock + 1] = scratch_.need;
    float* __restrict qx = scratch_.qx;
    float* __restrict qy = scratch_.qy;
    int* __restrict qi = scratch_.qi;
    const int eye = c.eye, bx = b.bx, by = b.by;
    const float cx = c.cx, cy = c.cy;
    std::fill(&need[0][0], &need[0][0] + (kBlock + 1) * (kBlock + 1), 0);
    for (int k = 0; k < nlit; k++) {
        const int lx = lp[k] & 7, ly = lp[k] >> 3;
        need[ly][lx] = 1;
        need[ly][lx + 1] = 1;
        need[ly + 1][lx] = 1;
        lf[k] = kExactD | kExactG;
    }
    int nq = 0;
    for (int qyl = 0; qyl <= b.bh; qyl++) {
        const float y = by + qyl + .5f - cy;
        for (int qxl = 0; qxl <= b.bw; qxl++) {
            if (!need[qyl][qxl]) continue;
            qx[nq] = bx + qxl + .5f - cx;
            qy[nq] = y;
            qi[nq] = qyl * (kBlock + 1) + qxl;
            if (++nq == kBatch) {
                DistN(eye, nq, qx, qy, scratch_.qd);
                for (int k = 0; k < nq; k++) exf[qi[k]] = scratch_.qd[k];
                nq = 0;
            }
        }
    }
    if (nq > 0) {
        DistN(eye, nq, qx, qy, scratch_.qd);
        for (int k = 0; k < nq; k++) exf[qi[k]] = scratch_.qd[k];
    }
}

template <bool k888>
EYE_NOINLINE void Engine::LitShade(const PixCtx& c, const Blk& b, int nlit) {
    const uint8_t* __restrict lp = scratch_.lp;
    const uint8_t* __restrict lf = scratch_.cls;
    const float* __restrict ld = scratch_.dv;
    const float* __restrict lgx = scratch_.gxv;
    const float* __restrict lgy = scratch_.gyv;
    const float(*__restrict ex)[kBlock + 1] = scratch_.ex;
    int nf = 0, ng = 0, ne = 0;
    for (int k = 0; k < nlit; k++) {
        const int lx = lp[k] & 7, ly = lp[k] >> 3;
        const int px = b.bx + lx, py = b.by + ly;
        const uint8_t f = lf[k];
        const float d = (f & kExactD) ? ex[ly][lx] : ld[k];
        if (d >= .5f) {
            if (d > kGlw) continue;
            if (b.ovl) {
                PutGlow<k888>(c, px, py, d);
            } else {
                PutGlowBin<k888>(c, px, py, d * kGlowRes);
            }
            ng++;
        } else if (d <= -kBev) {
            float fb[3];
            FlatBase(c, py, fb);
            PutFlat<k888>(c, px, py, fb, row_hl_[py]);
            nf++;
        } else {
            float gx, gy;
            if (f & kExactG) {
                gx = ex[ly][lx + 1] - d;
                gy = ex[ly + 1][lx] - d;
            } else {
                gx = lgx[k];
                gy = lgy[k];
            }
            PutLit<k888>(c, px, py, d, gx, gy);
            ne++;
        }
    }
    px_flat_ += nf;
    px_glow_ += ng;
    px_edge_ += ne;
}

// The open eye, settled, is the capsule alone (see OpenIdle): cheap enough for
// every pixel to take its exact distance, a row at a time, and the row below
// to give the forward differences. No grid, no blocks, no sorting. The field
// only grows away from the eye's middle column, so each row splits into runs
// found from the row alone: flat across the middle, then the lit bevel, the
// glow, and nothing. Runs are trimmed so that anything near a boundary still
// goes pixel by pixel; flat runs need no distances at all.
// kFast: RGB565 and no glint.
template <bool k888, bool kFast>
EYE_NOINLINE void Engine::OpenEye(const PixCtx& c) {
    constexpr bool fast = kFast && !k888;
    const Xform& X = xf_[c.eye];
    const int x0 = c.x0, y0 = c.y0, x1 = X.x1, y1 = X.y1;
    const int w = x1 - x0, n = w + 1;   // one past the box, for the differences
    const float ax = X.inv_sx * inv_wd_, ay = X.inv_sy, sm = X.sm;
    const float inv_ax = 1 / ax, inv_sm = 1 / sm;
    const float kc = c.cx - .5f - x0;   // the middle, in columns
    // Runs of row py: the glow's reach (s), sure glow (go minus gi), sure flat (f).
    struct Runs {
        Span s, go, gi, f;
    };
    auto runs = [&](int py) EYE_LAMBDA {
        const float qy = fabsf((py + .5f - c.cy) * ay) - (42 - 17);
        // Half-width in px of where the field is <= v; negative if nowhere.
        auto half = [&](float v) EYE_LAMBDA {
            const float t = v * inv_sm + 17;
            if (qy > t) return -1.f;
            return ((35 - 17) + (qy > 0 ? FastSqrt((t - qy) * (t + qy)) : t)) * inv_ax;
        };
        const float hs = half(kGlw), hg = half(.5f), hf = half(-kBev);
        Runs r;
        r.s = Around(kc, hs + .02f, w);
        r.go = Inside(Around(kc, hs - .02f, w), r.s);
        r.gi = Inside(Around(kc, hg + .02f, w), r.go);
        // A lit pixel's right-hand neighbour stays out of the flat run.
        r.f = Inside(Around(kc, hf - 1.02f, w), r.gi);
        return r;
    };
    // In the grid's memory: two rows of distances, a row's lit pixels, each
    // column's x, the lighting last worked out in each column, and LitRow's
    // scratch.
    const int span = (n + 3) & ~3;
    float* d0 = grid_;
    float* d1 = grid_ + span;
    uint16_t* const lit = reinterpret_cast<uint16_t*>(grid_ + 2 * span);
    float* const lx = grid_ + 3 * span;
    float* const memo = grid_ + 4 * span;   // 6 per column: d, gx, gy, then Lit
    float* const buf = grid_ + 10 * span;   // 7 per lit pixel
    // Every row takes a column's x from here, so that a straight edge gives
    // every row bit-identical distances.
    for (int k = 0; k < n; k++) lx[k] = x0 + k + .5f - c.cx;
    for (int k = 0; k < w; k++) memo[6 * k] = kUnset;
    // A row's distances where the row, or the one above it, may read them;
    // `hole` is left unset.
    uint32_t evals = 0;
    auto fill = [&](int py, const Runs& r, const Runs& up, float* d, Span& hole) EYE_LAMBDA {
        const int lo = std::min(r.s.lo, up.s.lo), hi = std::min(std::max(r.s.hi, up.s.hi) + 1, n);
        hole = {std::max(r.f.lo, up.f.lo), std::min(r.f.hi, up.f.hi)};
        if (hole.hi <= hole.lo) hole = {lo, lo};
        const float ly = py + .5f - c.cy;
        BoxRow(hole.lo - lo, lx + lo, ly, d + lo, ax, ay, sm);
        BoxRow(hi - hole.hi, lx + hole.hi, ly, d + hole.hi, ax, ay, sm);
        evals += static_cast<uint32_t>(hi - lo - (hole.hi - hole.lo));
    };
    Runs cur = runs(y0), nxt;
    Span hole0, hole1;
    fill(y0, cur, cur, d1, hole1);
    int counts[2] = {0, 0};   // flat, glow
    for (int py = y0; py < y1; py++) {
        std::swap(d0, d1);
        hole0 = hole1;
        Phase(2);
        nxt = runs(py + 1);
        fill(py + 1, nxt, cur, d1, hole1);
        Phase(4);
        const Runs& rs = cur;
        float fb[3];
        FlatBase(c, py, fb);
        const float rh = row_hl_[py];
        // Where eye 1's box overlaps eye 0's, its glow goes onto what is there.
        const int xo = c.overlap && py >= c.oy0 && py < c.oy1 ? std::min(c.ox1, x1) - x0 : 0;
        const Span gs[2] = {{rs.go.lo, rs.gi.lo}, {rs.gi.hi, rs.go.hi}};
        const Span cs[4] = {{rs.s.lo, rs.go.lo}, {rs.gi.lo, rs.f.lo}, {rs.f.hi, rs.gi.hi}, {rs.go.hi, rs.s.hi}};
        int nlit = 0;
        if constexpr (fast) {
            RowPaint p;
            p.o = static_cast<uint16_t*>(canvas_) + py * stride_ + x0;
            p.d = d0;
            p.ch = col_hl_ + x0;
            for (int j = 0; j < 4; j++) {
                p.th[j] = g_dither[py & 3][(x0 + j) & 3];
                p.dl[j] = g_dswar[py & 3][(x0 + j) & 3];
            }
            p.fb[0] = fb[0];
            p.fb[1] = fb[1];
            p.fb[2] = fb[2];
            p.rh = rh;
            p.xo = xo;
            nlit = PaintRow(p, rs.f, gs, cs, lit, counts);
        } else {
            const float* __restrict dr = d0;
            for (int k = rs.f.lo; k < rs.f.hi; k++) PutFlat<k888>(c, x0 + k, py, fb, rh);
            counts[0] += rs.f.hi - rs.f.lo;
            for (const Span& g : gs) {
                for (int k = g.lo; k < g.hi; k++) {
                    if (k < xo) {
                        PutGlow<k888>(c, x0 + k, py, dr[k]);
                    } else {
                        PutGlowBin<k888>(c, x0 + k, py, dr[k] * kGlowRes);
                    }
                }
                counts[1] += g.hi - g.lo;
            }
            for (const Span& cr : cs) {
                for (int k = cr.lo; k < cr.hi; k++) {
                    const float di = dr[k];
                    if (di > kGlw) continue;
                    const int px = x0 + k;
                    if (di <= -kBev) {
                        PutFlat<k888>(c, px, py, fb, rh);
                        counts[0]++;
                    } else if (di >= .5f) {
                        if (k < xo) {
                            PutGlow<k888>(c, px, py, di);
                        } else {
                            PutGlowBin<k888>(c, px, py, di * kGlowRes);
                        }
                        counts[1]++;
                    } else {
                        lit[nlit++] = static_cast<uint16_t>(k);
                    }
                }
            }
        }
        Phase(6);
        if (nlit > 0) {
            const OpenRow orow = {py,       x0,       nlit,     xo,       lit, d0, d1, hole0.lo, hole0.hi, hole1.lo,
                                  hole1.hi, memo,     lx,       ax,       ay,  sm, buf};
            LitRow<k888, fast>(c, orow);
        }
        cur = nxt;
    }
    px_flat_ += static_cast<uint32_t>(counts[0]);
    px_glow_ += static_cast<uint32_t>(counts[1]);
    sdf_evals_ += evals;
}

// OpenEye's lit pixels of one row, in three passes: gradients and what is
// already known, the lighting still to work out, then colour. Down the sides and across the top and bottom the field
// is straight, so a pixel there has exactly the distance and gradient of the
// one above it or the one before it, and takes their lighting as it is.
template <bool k888, bool kFast>
EYE_NOINLINE void Engine::LitRow(const PixCtx& c, const OpenRow& r) {
    const int py = r.py, x0 = r.x0, xo = r.xo, cap = r.nlit;
    const uint16_t* __restrict lit = r.lit;
    const float* __restrict dr = r.dr;
    const float* __restrict dn = r.dn;
    const int h0lo = r.h0lo, h1lo = r.h1lo;
    const unsigned h0w = static_cast<unsigned>(r.h0hi - h0lo), h1w = static_cast<unsigned>(r.h1hi - h1lo);
    // Only near-impossible rounding reaches a distance the rows skipped.
    auto grad = [&](int k, float d, float& gx, float& gy) EYE_LAMBDA {
        float dx = dr[k + 1], dy = dn[k];
        if (static_cast<unsigned>(k + 1 - h0lo) < h0w) BoxRow(1, r.lx + k + 1, py + .5f - c.cy, &dx, r.ax, r.ay, r.sm);
        if (static_cast<unsigned>(k - h1lo) < h1w) BoxRow(1, r.lx + k, py + 1.5f - c.cy, &dy, r.ax, r.ay, r.sm);
        gx = dx - d;
        gy = dy - d;
    };
    if constexpr (!kFast) {
        int ng = 0, ne = 0;
        for (int j = 0; j < cap; j++) {
            const int k = lit[j];
            const float d = dr[k];
            if (d >= .5f) {
                PutGlow<k888>(c, x0 + k, py, d);   // onto eye 0's
                ng++;
                continue;
            }
            float gx, gy;
            grad(k, d, gx, gy);
            PutLit<k888>(c, x0 + k, py, d, gx, gy);
            ne++;
        }
        px_glow_ += static_cast<uint32_t>(ng);
        px_edge_ += static_cast<uint32_t>(ne);
    } else {
        // Pass 1: gradients, and whose lighting is known. A pixel takes the
        // lighting of the one before it (dup), or its column's from an earlier
        // row, or goes on the list to work out (its key goes in the column now).
        struct Px {
            float d;
            int k;
            int dup;
        };
        Px* const list = reinterpret_cast<Px*>(r.buf);
        uint16_t* const miss = reinterpret_cast<uint16_t*>(list + cap);
        // What falls on eye 0's goes last, one by one; its columns wait here.
        uint16_t* const later = miss + cap;
        float* __restrict memo = r.memo;
        int m = 0, nm = 0, nl = 0;
        Phase(5);
        {
            Px* __restrict q = list;
            uint16_t* __restrict mq = miss;
            float pd = kUnset, pgx = 0, pgy = 0;
            // A lit pixel is never in the flat run, so never in either hole,
            // which lie inside it; only the one just left of the run can have
            // a neighbour there, and only if rounding made it lit.
            const int kh = h0w > 0 ? h0lo - 1 : -1;
            // No calls in here: they would cost every float register.
            for (const uint16_t* lp = lit; lp < lit + cap; lp++) {
                const int k = *lp;
                const float d = dr[k];
                if (d >= .5f || (k < xo && d > -.5f) || k == kh) {
                    later[nl++] = static_cast<uint16_t>(k);
                    continue;
                }
                const float gx = dr[k + 1] - d, gy = dn[k] - d;
                const int dup = d == pd && gx == pgx && gy == pgy;
                if (!dup) {
                    float* __restrict mm = memo + 6 * k;
                    if (!(d == mm[0] && gx == mm[1] && gy == mm[2])) {
                        mm[0] = d;
                        mm[1] = gx;
                        mm[2] = gy;
                        *mq++ = static_cast<uint16_t>(k);
                    }
                    pd = d;
                    pgx = gx;
                    pgy = gy;
                }
                q->d = d;
                q->k = k;
                q->dup = dup;
                q++;
            }
            m = static_cast<int>(q - list);
            nm = static_cast<int>(mq - miss);
        }
        // Pass 2: the lighting still unknown.
        Phase(1);
        {
            const LightV lv = {c.lx, c.ly, c.lz, c.hx, c.hy, c.hz};
            for (int i = 0; i < nm; i++) {
                float* __restrict mm = memo + 6 * miss[i];
                LightOf(lv, mm[0], mm[1], mm[2], mm[3], mm[4], mm[5]);
            }
        }
        // Pass 3: colour, PutLit's, with the row's factors at hand.
        Phase(6);
        {
            const float* col = lut_[row_vi_[py]];
            const float c0 = col[0], c1 = col[1], c2 = col[2], rh = row_hl_[py];
            const float cb0 = c.cb[0], cb1 = c.cb[1], cb2 = c.cb[2];
            const float* __restrict ch = col_hl_ + x0;
            uint16_t* __restrict o = static_cast<uint16_t*>(canvas_) + py * stride_ + x0;
            float th[4];
            for (int j = 0; j < 4; j++) th[j] = g_dither[py & 3][(x0 + j) & 3];
            float br = 0, sp = 0, rm = 0;
            for (const Px* p = list; p < list + m; p++) {
                const int k = p->k;
                const float d = p->d;
                if (!p->dup) {
                    const float* mm = memo + 6 * k;
                    br = mm[3];
                    sp = mm[4];
                    rm = mm[5];
                }
                const float ad = sp + rh * ch[k];
                float cr = c0 * br + ad, cg = c1 * br + ad, cbl = c2 * br + ad;
                cr += cb0 * rm;
                cg += cb1 * rm;
                cbl += cb2 * rm;
                if (d > -.5f) {
                    // The anti-aliased edge, over the background.
                    const float cov = .5f - d, ic = 1 - cov;
                    const float gl = GlowAt(c.glow, std::max(d, 0.f)) * c.en;
                    cr = cr * cov + (c.bg[0] + cb0 * gl) * ic;
                    cg = cg * cov + (c.bg[1] + cb1 * gl) * ic;
                    cbl = cbl * cov + (c.bg[2] + cb2 * gl) * ic;
                }
                o[k] = Pack565(cr, cg, cbl, th[k & 3]);
            }
        }
        int ng = 0;
        for (int j = 0; j < nl; j++) {
            const int k = later[j];
            const float d = dr[k];
            if (d >= .5f) {
                PutGlow<k888>(c, x0 + k, py, d);
                ng++;
                continue;
            }
            float gx, gy;
            grad(k, d, gx, gy);
            PutLit<k888>(c, x0 + k, py, d, gx, gy);
        }
        px_glow_ += static_cast<uint32_t>(ng);
        px_edge_ += static_cast<uint32_t>(m + nl - ng);
    }
}

template <bool k888>
void Engine::RenderEye(int eye) {
    const uint32_t prof_t0 = prof_now_ != nullptr ? prof_now_() : 0;
    const Xform& X = xf_[eye];
    const int x0 = X.x0, y0 = X.y0, x1 = X.x1, y1 = X.y1;
    if (x1 <= x0 || y1 <= y0) return;
    PixCtx c;
    c.eye = eye;
    c.x0 = x0;
    c.y0 = y0;
    c.cx = X.cx;
    c.cy = X.cy;
    c.inv_s = 1.f / scale_;
    const float cx = c.cx, cy = c.cy, inv_s = c.inv_s;

    // Row and column factors: the gradient row, and the soft highlight blob
    // up and to the left, which separates into a row and a column term.
    for (int y = y0; y < y1; y++) {
        const int vi = static_cast<int>(((y + .5f - cy) * inv_s + 40) * .7875f);
        row_vi_[y] = static_cast<uint8_t>(vi < 0 ? 0 : (vi > 63 ? 63 : vi));
    }
    // The blob is a window's reflection: it holds still while the eye turns,
    // so inside the eye it slides against the gaze (not in the mock-up).
    const float par = reference_ ? 0.f : kHighlightParallax;
    Gauss(row_hl_ + y0, y1 - y0, (y0 + .5f - cy) * inv_s + 17 + par * gy_, inv_s, 90.f, 38.f);
    Gauss(col_hl_ + x0, x1 - x0, (x0 + .5f - cx) * inv_s + 11 + par * gx_, inv_s, 300.f, 1.f);
    // A band of light sweeps across once a morph lands.
    c.ga = 0;
    c.gp = 0;
    const Morph& m = morph_[eye];
    if (m.glint >= 0) {
        const float u = static_cast<float>((t_ - m.glint) / .52);
        if (u < 1) {
            c.ga = sinf(kPi * u);
            c.gp = -60 + 120 * EaseInOut(u);
        }
    }
    c.glow = reference_ ? g_glow_raw : g_glow;
    c.en = en_;
    c.lx = g_light[0];
    c.ly = g_light[1];
    c.lz = g_light[2];
    c.hx = g_light[3];
    c.hy = g_light[4];
    c.hz = g_light[5];
    c.flat_br = .34f + .72f * c.lz;
    c.flat_ad = SpecAt(c.hz);
    for (int i = 0; i < 3; i++) c.cb[i] = cb_[i];
    c.cmax = std::max(cb_[0], std::max(cb_[1], cb_[2]));
    if (k888) {
        for (int i = 0; i < 3; i++) c.bg[i] = static_cast<float>(static_cast<uint8_t>(bg_[i]));
    } else {
        c.bg[0] = static_cast<float>((bg565_ >> 11) << 3);
        c.bg[1] = static_cast<float>(((bg565_ >> 5) & 63) << 2);
        c.bg[2] = static_cast<float>((bg565_ & 31) << 3);
    }
    PrepareGlow(reference_, c.en, cb_, c.bg);
    const Xform& O = xf_[0];
    c.overlap = eye == 1 && O.x1 > O.x0;
    c.ox1 = O.x1;
    c.oy0 = O.y0;
    c.oy1 = O.y1;

    const int nw = ((x1 - x0 - 1) >> 1) + 2;
    const int nhh = ((y1 - y0 - 1) >> 1) + 2;
    c.nw = nw;
    c.nh = nhh;
    if (exact_ || grid_ == nullptr || static_cast<size_t>(nw) * nhh > kWorkFloats) {
        // Every pixel evaluated exactly, with the mock-up's forward differences.
        for (int py = y0; py < y1; py++) {
            const float ly = py + .5f - cy;
            float fb[3];
            FlatBase(c, py, fb);
            const float rh = row_hl_[py];
            for (int px = x0; px < x1; px++) {
                const float lx = px + .5f - cx;
                const float d = Dist(eye, lx, ly);
                if (d > kGlw) continue;
                if (d >= .5f) {
                    PutGlow<k888>(c, px, py, d);
                } else if (d <= -kBev) {
                    PutFlat<k888>(c, px, py, fb, rh);
                } else {
                    PutLit<k888>(c, px, py, d, Dist(eye, lx + 1, ly) - d, Dist(eye, lx, ly + 1) - d);
                }
            }
        }
        return;
    }

    // A coarse pass on a 2 px grid, filled lazily and in batches, sorts pixels
    // block by 8x8 block. Blocks far outside the glow or deep inside never
    // touch it; blocks all in the glow read their distances off it. In the
    // rest, the flat middle and the glow come off the grid too, and so does
    // everything lit wherever the grid follows the field smoothly: distance
    // interpolated with a curvature correction, gradient interpolated from
    // central differences at the nodes. Only where it does not -- corners,
    // ridges down the middle of strokes -- is the rim evaluated exactly.
    // Each kind of block has its own out-of-line loop (FlatBlock ...).
    // The open eye needs none of it (see OpenEye).
    if (m.phase != 1 && (m.pure || m.draw) && m.to == static_cast<uint8_t>(Shape::kIdle) && OpenIdle() &&
        17 * static_cast<size_t>(x1 - x0 + 4) <= kWorkFloats) {
        if (prof_now_ != nullptr) prof_setup_[eye] += prof_now_() - prof_t0;
        ph_t_ = prof_now_ != nullptr ? prof_now_() : 0;
        ph_cur_ = 7;
        if (!k888 && !(c.ga > 0)) {
            OpenEye<k888, true>(c);
        } else {
            OpenEye<k888, false>(c);
        }
        Phase(7);
        return;
    }
    std::fill(grid_, grid_ + static_cast<size_t>(nw) * nhh, kUnset);
    const float lip = X.lip;
    // Bilinear interpolation is off by at most this much from the true distance.
    const float slack = 1.45f * lip;
    c.glow_min = .5f + slack;
    c.flat_max = -kBev - slack;
    c.rim_min = -.5f - slack;
    // Mid-morph the field is a blend of several shapes, dear to evaluate.
    c.exact_lit = m.phase != 1 && (m.pure || m.draw);
    // Blocks start at x0 + 8k: one row of dither per row phase serves them all.
    for (int r = 0; r < 4; r++) {
        for (int lx = 0; lx < kBlock; lx++) {
            scratch_.dth[r][lx] = g_dither[r][(x0 + lx) & 3];
            scratch_.dsw[r][lx] = g_dswar[r][(x0 + lx) & 3];
        }
    }
    // Per-block scratch lives in the engine: the LVGL task's stack is small.
    static_assert(kBlock == kBlockPx, "scratch is sized for the block");
    float* __restrict qx = scratch_.qx;
    float* __restrict qy = scratch_.qy;
    float* __restrict dcs = scratch_.dcs;
    if (prof_now_ != nullptr) prof_setup_[eye] += prof_now_() - prof_t0;
    // Profiling: the block just finished, and what it held.
    uint32_t pb_t0 = 0, pb_ev = 0, pb_f = 0, pb_g = 0, pb_e = 0;
    ph_t_ = prof_now_ != nullptr ? prof_now_() : 0;
    ph_cur_ = 7;
    auto prof_end = [&](uint8_t kind) EYE_LAMBDA {
        if (prof_now_ == nullptr || prof_n_ >= prof_max_) return;
        BlockProf& p = prof_[prof_n_++];
        p.cycles = prof_now_() - pb_t0;
        p.flat = static_cast<uint16_t>(px_flat_ - pb_f);
        p.glow = static_cast<uint16_t>(px_glow_ - pb_g);
        p.edge = static_cast<uint16_t>(px_edge_ - pb_e);
        p.evals = static_cast<uint16_t>(sdf_evals_ - pb_ev);
        p.kind = kind;
        p.eye = static_cast<uint8_t>(eye);
    };
    for (int by = y0; by < y1; by += kBlock) {
        const int bh = std::min(kBlock, y1 - by);
        // The block centres of this row, in one batch.
        int nb = 0;
        for (int bx = x0; bx < x1; bx += kBlock, nb++) {
            qx[nb] = bx + std::min(kBlock, x1 - bx) * .5f - cx;
            qy[nb] = by + bh * .5f - cy;
        }
        if (prof_now_ != nullptr) pb_t0 = prof_now_();
        Phase(0);
        DistN(eye, nb, qx, qy, dcs);
        Phase(7);
        int bi = 0;
        for (int bx = x0; bx < x1; bx += kBlock, bi++) {
            const int bw = std::min(kBlock, x1 - bx);
            if (prof_now_ != nullptr) {
                if (bi > 0) pb_t0 = prof_now_();
                pb_ev = sdf_evals_ - (bi == 0 ? nb : 0);
                pb_f = px_flat_;
                pb_g = px_glow_;
                pb_e = px_edge_;
            }
            const float dc = dcs[bi];
            const float rad = .5f * FastSqrt(static_cast<float>(bw * bw + bh * bh)) * lip + .75f;
            if (dc - rad > kGlw) continue;
            Blk blk{bx, by, bw, bh, 0, 0, 0, 0, false};
            if (dc + rad < -kBev) {
                Phase(1);
                FlatBlock<k888>(c, blk);
                Phase(7);
                prof_end(1);
                continue;
            }
            // Where eye 1's box overlaps eye 0's, its glow goes onto what is there.
            blk.ovl = c.overlap && bx < O.x1 && by < O.y1 && by + bh > O.y0;
            blk.i0 = (bx - x0) >> 1;
            blk.i1 = ((bx + bw - 1 - x0) >> 1) + 1;
            blk.j0 = (by - y0) >> 1;
            blk.j1 = ((by + bh - 1 - y0) >> 1) + 1;
            Phase(2);
            FillNodes(c, blk.i0, blk.i1, blk.j0, blk.j1);
            if (dc - rad >= .5f && !blk.ovl) {
                Phase(3);
                GlowBlock<k888>(c, blk);
                Phase(7);
                prof_end(3);
                continue;
            }
            Phase(4);
            const int nlit = !k888 && !blk.ovl && !(c.ga > 0) ? SortBlock<k888, true>(c, blk)
                                                              : SortBlock<k888, false>(c, blk);
            if (nlit > 0) {
                if (c.exact_lit) {
                    Phase(5);
                    LitExact(c, blk, nlit);
                } else {
                    Phase(2);
                    FillNodes(c, blk.i0 - 1, blk.i1 + 1, blk.j0 - 1, blk.j1 + 1);
                    Phase(5);
                    NodeGrads(c, blk);
                    LitInterp(c, blk, nlit);
                }
                Phase(6);
                LitShade<k888>(c, blk, nlit);
            }
            Phase(7);
            prof_end(2);
        }
    }
    Phase(7);
}

template <bool k888>
void Engine::RenderSparks() {
    for (int i = 0; i < nsparks_; i++) {
        const Spark& s = sparks_[i];
        const float u = static_cast<float>((t_ - s.t0) / .65);
        if (u >= 1) continue;
        const float a = sinf(kPi * u), z = 5 * a * s.s, inv_2s = 1 / (2.f * s.s);
        const int r = static_cast<int>(ceilf(10 * s.s));
        const int x0 = std::max(0, static_cast<int>(s.x) - r), x1 = std::min(w_, static_cast<int>(s.x) + r + 1);
        const int y0 = std::max(strip_y0_, static_cast<int>(s.y) - r),
                  y1 = std::min(strip_y1_, static_cast<int>(s.y) + r + 1);
        for (int py = y0; py < y1; py++) {
            for (int px = x0; px < x1; px++) {
                const float lx = px + .5f - s.x, ly = py + .5f - s.y;
                const float d = std::min(Seg(lx, ly, -z, 0, z, 0), Seg(lx, ly, 0, -z, 0, z)) - .8f * s.s;
                const float cov = Clamp(.5f - d, 0, 1) * a;
                const float halo = expf(-std::max(d, 0.f) * inv_2s) * .5f * a;
                if (cov < .004f && halo < .01f) continue;
                float pr, pg, pb;
                ReadPixel<k888>(px, py, pr, pg, pb);
                // A pink halo, then the white-ish cross over it.
                pr += (255 - pr) * halo;
                pg += (160 - pg) * halo;
                pb += (200 - pb) * halo;
                pr += (255 - pr) * cov;
                pg += (225 - pg) * cov;
                pb += (238 - pb) * cov;
                WritePixel<k888>(px, py, pr, pg, pb);
            }
        }
    }
}

int Engine::RingRects(Rect* out, int max) const {
    const float r = RingRadius(), band = RingBand(ring_shown_) + 1.5f;
    const float ro = r + band, ri = std::max(0.f, r - band);
    const float cx = w_ * .5f, cy = mid_;
    int n = 0;
    for (int s = 0; s < 12 && n < max; s++) {
        float xa = 1e9f, ya = 1e9f, xb = -1e9f, yb = -1e9f;
        for (int k = 0; k <= 4; k++) {
            const float a = (s + k / 4.f) * kTau / 12;
            const float c = cosf(a), sn = sinf(a);
            for (const float rad : {ri, ro}) {
                xa = std::min(xa, cx + c * rad);
                xb = std::max(xb, cx + c * rad);
                ya = std::min(ya, cy + sn * rad);
                yb = std::max(yb, cy + sn * rad);
            }
        }
        const Rect rc = {std::max(0, static_cast<int>(floorf(xa)) - 1),
                         std::max(strip_y0_, static_cast<int>(floorf(ya)) - 1),
                         std::min(w_, static_cast<int>(ceilf(xb)) + 1),
                         std::min(strip_y1_, static_cast<int>(ceilf(yb)) + 1)};
        if (rc.x1 > rc.x0 && rc.y1 > rc.y0) out[n++] = rc;
    }
    return n;
}

template <bool k888>
void Engine::RenderRing() {
    const Ring mode = ring_shown_;
    const float al = ring_alpha_;
    if (al < .02f || mode == Ring::kOff) return;
    const float R = RingRadius();
    const float cx = w_ * .5f, cy = mid_;
    const double t = t_;
    auto blend = [&](int px, int py, const float* col, float cov) EYE_LAMBDA {
        float pr, pg, pb;
        ReadPixel<k888>(px, py, pr, pg, pb);
        WritePixel<k888>(px, py, pr + (col[0] - pr) * cov, pg + (col[1] - pg) * cov, pb + (col[2] - pb) * cov);
    };

    const float dtk = kTau / kTicks;
    if (mode == Ring::kListen || mode == Ring::kSpeak) {
        const bool speak = mode == Ring::kSpeak;
        float lv = voice_lv_;
        if (speak) {
            const float pulse = std::max(0.f, sinf(Wrap(t * 14))) * (.55f + .45f * sinf(Wrap(t * 2.3)));
            lv = .35f + .65f * pulse;
        }
        // Phases wrapped once in double, then float per tick: the S3 has no double FPU.
        const float p1 = Wrap(t * (speak ? 11 : 7)), p2 = Wrap(t * 3), p3 = Wrap(t);
        // Tick by tick: a radial stroke 3 px wide with round ends. Only the rows
        // it crosses are visited, and in each only the columns it can reach.
        constexpr float kHalfW = 1.5f, kReach = kHalfW + .5f;
        for (int i = 0; i < kTicks; i++) {
            const float wob = fabsf(sinf(WrapF(i * 1.7f + p1)) * sinf(WrapF(i * .63f - p2)));
            const float len = 2 + (speak ? 12.f : 10.f) * lv * (.35f + .65f * wob);
            const float v = .5f + .5f * sinf(WrapF(i * dtk - kPi / 2 + p3));
            const float* col = lut_[static_cast<int>(v * 63)];
            const float ca = g_tick[i][0], sa = g_tick[i][1];
            // From (ax, ay) outwards by (bax, bay).
            const float ax = cx + ca * (R - len * .5f), ay = cy + sa * (R - len * .5f);
            const float bax = ca * len, bay = sa * len;
            const float inv_bb = FastRecip(len * len);
            const bool level = fabsf(bay) < 1e-3f;
            const float inv_bay = level ? 0.f : 1 / bay;
            const int ya = std::max(strip_y0_, static_cast<int>(FloorF(std::min(ay, ay + bay) - kReach)));
            const int yb = std::min(strip_y1_ - 1, static_cast<int>(FloorF(std::max(ay, ay + bay) + kReach)));
            for (int py = ya; py <= yb; py++) {
                const float y = py + .5f;
                // The part of the stroke's spine within reach of this row.
                float ta = 0, tb = 1;
                if (!level) {
                    ta = (y - kReach - ay) * inv_bay;
                    tb = (y + kReach - ay) * inv_bay;
                    if (ta > tb) std::swap(ta, tb);
                    ta = std::max(ta, 0.f);
                    tb = std::min(tb, 1.f);
                    if (ta > tb) continue;
                }
                const float sx0 = ax + bax * ta, sx1 = ax + bax * tb;
                const int xa = std::max(0, static_cast<int>(FloorF(std::min(sx0, sx1) - kReach)));
                const int xb = std::min(w_ - 1, static_cast<int>(FloorF(std::max(sx0, sx1) + kReach)));
                const float pay = y - ay;
                for (int px = xa; px <= xb; px++) {
                    const float pax = px + .5f - ax;
                    const float h = Clamp((pax * bax + pay * bay) * inv_bb, 0, 1);
                    const float d = Len(pax - bax * h, pay - bay * h) - kHalfW;
                    const float cov = Clamp(.5f - d, 0, 1) * al;
                    if (cov <= .002f) continue;
                    blend(px, py, col, cov);
                }
            }
        }
        return;
    }

    constexpr int kSegs = 36;
    uint8_t seg_col[kSegs];
    const float segp = kTau / kSegs;
    const float seg_rot = Wrap(t * .8);
    if (mode == Ring::kCelebrate) {
        const float p2t = Wrap(2 * t);
        for (int i = 0; i < kSegs; i++) {
            const float a0 = i * segp + seg_rot;
            seg_col[i] = static_cast<uint8_t>((.5f + .5f * sinf(WrapF(2 * a0 + p2t))) * 63);
        }
    }
    const float head = Wrap(t * 4.2);
    const float comet = 1.1f * kPi;
    // Per-pixel divisions become multiplications by these.
    const float inv_tau = 1 / kTau, inv_comet = 1 / comet, inv_segp = kSegs / kTau;
    const float pulse = .6f + .4f * sinf(Wrap(t * 6));
    const float flash = .35f + .65f * std::max(0.f, sinf(Wrap(t * 5)));
    // How far from the circle a pixel can still get some cover: the stroke's
    // half width plus half a pixel.
    const float reach = mode == Ring::kCelebrate ? 2.5f : 3.f;
    const float ro = R + reach, ri = std::max(0.f, R - reach);

    const int ya = std::max(strip_y0_, static_cast<int>(floorf(cy - ro)));
    const int yb = std::min(strip_y1_, static_cast<int>(ceilf(cy + ro)));
    for (int py = ya; py < yb; py++) {
        const float dy = py + .5f - cy;
        if (fabsf(dy) >= ro) continue;
        const float xo = sqrtf(ro * ro - dy * dy);
        const float xi = fabsf(dy) < ri ? sqrtf(ri * ri - dy * dy) : 0.f;
        // One span across the top and bottom of the ring, two through the middle.
        for (int span = 0; span < 2; span++) {
            float sa, sb;
            if (xi > 0) {
                sa = span == 0 ? cx - xo : cx + xi;
                sb = span == 0 ? cx - xi : cx + xo;
            } else {
                if (span == 1) break;
                sa = cx - xo;
                sb = cx + xo;
            }
            const int xa = std::max(0, static_cast<int>(floorf(sa - .5f)));
            const int xb = std::min(w_, static_cast<int>(ceilf(sb + .5f)));
            for (int px = xa; px < xb; px++) {
                const float dx = px + .5f - cx;
                const float r = Len(dx, dy), u = r - R;
                if (fabsf(u) > reach) continue;
                const float ang = FastAtan2(dy, dx);
                float d, alpha = al;
                const float* col = cb_;
                switch (mode) {
                    case Ring::kComet: {
                        float ph = head - ang;
                        ph -= kTau * FloorF(ph * inv_tau);
                        const float behind = ph > kPi ? ph - kTau : ph;   // < 0: ahead of the head
                        if (behind * r < -1.f || ph > comet) continue;
                        const float f = std::max(0.f, behind) * inv_comet;
                        alpha = al * (1 - f) * Clamp(behind * r + .5f, 0, 1);
                        col = lut_[static_cast<int>(f * 63)];
                        d = fabsf(u) - 2.5f;
                        break;
                    }
                    case Ring::kCelebrate: {
                        float ps = ang - seg_rot;
                        ps -= kTau * FloorF(ps * inv_tau);
                        const int k = std::min(kSegs - 1, static_cast<int>(ps * inv_segp));
                        const float s = (ps - k * segp) * R, arc = .62f * segp * R;
                        const float to_next = segp * R - s;
                        const bool next = s > arc && to_next < s - arc;
                        const float along = s > arc ? std::min(s - arc, to_next) : 0.f;
                        d = Len(along, u) - 2.f;
                        col = lut_[seg_col[next ? (k + 1) % kSegs : k]];
                        alpha = al * pulse;
                        break;
                    }
                    default:
                        d = fabsf(u) - 2.5f;
                        alpha = al * flash;
                        break;
                }
                const float cov = Clamp(.5f - d, 0, 1) * alpha;
                if (cov <= .002f) continue;
                blend(px, py, col, cov);
            }
        }
    }
}

template <bool k888>
int Engine::DrawT(Rect* out, int max_out) {
    sdf_evals_ = 0;
    px_flat_ = px_glow_ = px_edge_ = 0;
    Rect cur[kMaxRects];
    int ncur = 0;
    for (const Xform& X : xf_) {
        if (X.x1 > X.x0 && X.y1 > X.y0) cur[ncur++] = {X.x0, X.y0, X.x1, X.y1};
    }
    for (int i = 0; i < nsparks_ && ncur < kMaxRects; i++) {
        const Spark& s = sparks_[i];
        const int r = static_cast<int>(ceilf(10 * s.s));
        const Rect rc = {std::max(0, static_cast<int>(s.x) - r), std::max(strip_y0_, static_cast<int>(s.y) - r),
                         std::min(w_, static_cast<int>(s.x) + r + 1),
                         std::min(strip_y1_, static_cast<int>(s.y) + r + 1)};
        if (rc.x1 > rc.x0 && rc.y1 > rc.y0) cur[ncur++] = rc;
    }
    const bool ring_vis = ring_alpha_ >= .02f && ring_shown_ != Ring::kOff;
    if (ring_vis) ncur += RingRects(cur + ncur, kMaxRects - ncur);
    ring_was_visible_ = ring_vis;

    // Everything drawn last time or this time gets cleared and repainted.
    Rect dirty[2 * kMaxRects];
    int nd = 0;
    if (full_clear_) {
        dirty[nd++] = {0, std::min(clear_y0_, strip_y0_), w_, std::max(clear_y1_, strip_y1_)};
        full_clear_ = false;
    } else {
        for (int i = 0; i < nprev_; i++) dirty[nd++] = prev_[i];
        for (int i = 0; i < ncur; i++) {
            bool dup = false;
            for (int j = 0; j < nd && !dup; j++) {
                dup = dirty[j].x0 == cur[i].x0 && dirty[j].y0 == cur[i].y0 && dirty[j].x1 == cur[i].x1 &&
                      dirty[j].y1 == cur[i].y1;
            }
            if (!dup) dirty[nd++] = cur[i];
        }
    }
    clear_y0_ = strip_y0_;
    clear_y1_ = strip_y1_;
    int n = 0;
    dirty_px_ = 0;
    for (int i = 0; i < nd; i++) {
        Rect r = dirty[i];
        r.x0 = std::max(r.x0, 0);
        r.x1 = std::min(r.x1, w_);
        r.y0 = std::max(r.y0, 0);
        r.y1 = std::min(r.y1, h_);
        if (r.x1 <= r.x0 || r.y1 <= r.y0) continue;
        ClearRect<k888>(r);
        dirty_px_ += static_cast<uint32_t>((r.x1 - r.x0) * (r.y1 - r.y0));
        dirty[n++] = r;
    }
    RenderEye<k888>(0);
    RenderEye<k888>(1);
    RenderSparks<k888>();
    RenderRing<k888>();

    std::copy(cur, cur + ncur, prev_);
    nprev_ = ncur;
    last_draw_t_ = t_;
    std::copy(cur_sig_, cur_sig_ + kSig, sig_);
    sig_valid_ = true;

    if (n <= max_out) {
        std::copy(dirty, dirty + n, out);
        return n;
    }
    if (max_out <= 0) return 0;
    Rect u = dirty[0];
    for (int i = 1; i < n; i++) {
        u.x0 = std::min(u.x0, dirty[i].x0);
        u.y0 = std::min(u.y0, dirty[i].y0);
        u.x1 = std::max(u.x1, dirty[i].x1);
        u.y1 = std::max(u.y1, dirty[i].y1);
    }
    out[0] = u;
    return 1;
}

int Engine::Draw(Rect* out, int max_out) {
    if (canvas_ == nullptr) return 0;
    return fmt_ == PixelFormat::kRgb888 ? DrawT<true>(out, max_out) : DrawT<false>(out, max_out);
}

// ---- benchmark ---------------------------------------------------------------------------

void Engine::Benchmark(uint32_t (*now)(), void* canvas, int width, int height, int stride_px, float* work,
                       size_t work_floats, void* engine_mem, BenchResult* r) {
    BlockProf* const blocks = r->blocks;
    const int max_blocks = r->max_blocks;
    uint32_t (*const block_now)() = r->block_now != nullptr ? r->block_now : now;
    void (*const pm_start)(int) = r->pm_start;
    void (*const pm_stop)(int, uint32_t*) = r->pm_stop;
    const int pm_configs = std::min(r->pm_configs, 8);
    *r = BenchResult();
    Engine* e = engine_mem != nullptr ? new (engine_mem) Engine() : new Engine();
    e->Init(canvas, PixelFormat::kRgb565, width, height, stride_px, work, work_floats, 1);
    e->SetDeterministic(true);
    e->SetBackground(0x101214);
    // A short canvas (an internal-RAM test) holds just the face, from row 0.
    e->SetStrip(height > 200 ? 26 : 0, height);
    for (int i = 0; i < 120; i++) e->Advance(1 / 60.0);

    constexpr int kN = 2048;
    float acc = 0;
    uint32_t t0 = now();
    for (int i = 0; i < kN; i++) acc += sqrtf(1.f + i * .37f);
    r->sqrt_lib = (now() - t0) / kN;
    t0 = now();
    for (int i = 0; i < kN; i++) acc += FastSqrt(1.f + i * .37f);
    r->sqrt_fast = (now() - t0) / kN;
    t0 = now();
    for (int i = 0; i < kN; i++) acc += (1.f + i) / (3.f + i * .5f);
    r->div_lib = (now() - t0) / kN;
    t0 = now();
    for (int i = 0; i < kN; i++) acc += expf(-i * .001f);
    r->exp_lib = (now() - t0) / kN;
    t0 = now();
    for (int i = 0; i < kN; i++) acc += e->Dist(0, (i & 63) - 32.f, ((i >> 6) & 63) - 32.f);
    r->dist_idle = (now() - t0) / kN;

    // Timings keep the fastest of a few tries: whatever else runs on this core
    // (the network, audio) can only add to them.
    constexpr int kTries = 5;
    Rect rects[kMaxRects];
    r->draw_full = UINT32_MAX;
    for (int k = 0; k < kTries; k++) {
        e->Invalidate();
        t0 = now();
        e->Draw(rects, kMaxRects);
        r->draw_full = std::min(r->draw_full, now() - t0);
    }
    r->evals = e->sdf_evals_;
    r->px_flat = e->px_flat_;
    r->px_glow = e->px_glow_;
    r->px_edge = e->px_edge_;
    r->px_dirty = e->dirty_px_;

    // The same frame again, piece by piece.
    const Rect strip = {0, e->strip_y0_, e->w_, e->strip_y1_};
    r->clear = r->eye0 = r->eye1 = UINT32_MAX;
    for (int k = 0; k < kTries; k++) {
        t0 = now();
        e->ClearRect<false>(strip);
        r->clear = std::min(r->clear, now() - t0);
        t0 = now();
        e->RenderEye<false>(0);
        r->eye0 = std::min(r->eye0, now() - t0);
        t0 = now();
        e->RenderEye<false>(1);
        r->eye1 = std::min(r->eye1, now() - t0);
    }
    // The block loop by phase in cycles, fastest of the tries per phase.
    for (int k = 0; k < 8; k++) r->phase_cyc[k] = UINT32_MAX;
    for (int t = 0; t < kTries; t++) {
        e->ClearRect<false>(strip);
        for (int k = 0; k < 8; k++) e->prof_phase_[k] = 0;
        e->prof_now_ = now;
        e->RenderEye<false>(0);
        e->RenderEye<false>(1);
        e->prof_now_ = nullptr;
        for (int k = 0; k < 8; k++) r->phase_cyc[k] = std::min(r->phase_cyc[k], e->prof_phase_[k]);
    }
    for (int k = 0; k < 8; k++) e->prof_phase_[k] = 0;
    e->prof_setup_[0] = e->prof_setup_[1] = 0;
    if (blocks != nullptr) {
        e->ClearRect<false>(strip);
        e->prof_now_ = block_now;
        e->prof_ = blocks;
        e->prof_max_ = max_blocks;
        e->RenderEye<false>(0);
        e->RenderEye<false>(1);
        r->blocks = blocks;
        r->max_blocks = max_blocks;
        r->nblocks = e->prof_n_;
        r->setup[0] = e->prof_setup_[0];
        r->setup[1] = e->prof_setup_[1];
        for (int k = 0; k < 8; k++) r->phase[k] = e->prof_phase_[k];
        e->prof_now_ = nullptr;
    }
    for (int k = 0; k < pm_configs && pm_start != nullptr && pm_stop != nullptr; k++) {
        r->pm[k][0] = r->pm[k][1] = UINT32_MAX;
        for (int t = 0; t < 3; t++) {
            uint32_t v[2];
            e->ClearRect<false>(strip);
            pm_start(k);
            e->RenderEye<false>(0);
            pm_stop(k, v);
            r->pm[k][0] = std::min(r->pm[k][0], v[0]);
            r->pm[k][1] = std::min(r->pm[k][1], v[1]);
        }
    }

    uint32_t sum = 0;
    for (int k = 0; k < 8; k++) {
        e->SetGaze(false, k * 1.5f, 0);
        for (int j = 0; j < 6; j++) e->Advance(1 / 60.0);
        t0 = now();
        e->Draw(rects, kMaxRects);
        sum += now() - t0;
    }
    r->draw_move = sum / 8;

    // The ring on its own, faded fully in.
    const Ring modes[3] = {Ring::kListen, Ring::kComet, Ring::kCelebrate};
    uint32_t* out[3] = {&r->ring_listen, &r->ring_comet, &r->ring_celebrate};
    for (int m = 0; m < 3; m++) {
        e->SetRing(modes[m]);
        for (int j = 0; j < 90; j++) e->Advance(1 / 60.0);
        *out[m] = UINT32_MAX;
        for (int t = 0; t < kTries; t++) {
            t0 = now();
            e->RenderRing<false>();
            *out[m] = std::min(*out[m], now() - t0);
        }
    }
    Rect rr[12];
    const int nr = e->RingRects(rr, 12);
    for (int i = 0; i < nr; i++) r->ring_px += (rr[i].x1 - rr[i].x0) * (rr[i].y1 - rr[i].y0);
    e->SetRing(Ring::kOff);
    for (int j = 0; j < 90; j++) e->Advance(1 / 60.0);

    e->SetExact(true);
    e->Invalidate();
    t0 = now();
    e->Draw(rects, kMaxRects);
    r->draw_exact = now() - t0;
    r->evals_exact = e->sdf_evals_;
    r->sink = acc;
    if (engine_mem != nullptr) {
        e->~Engine();
    } else {
        delete e;
    }
}

}  // namespace eyes
