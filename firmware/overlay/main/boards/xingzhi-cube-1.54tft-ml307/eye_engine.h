// Signed-distance-field eyes for the 240x240 cube.
//
// Every shape is a distance field: morphing blends distances, and the same
// distance gives the anti-aliased edge, the glow and the bevel lighting. The
// rounded box, segment, arc, heart and star distance functions follow Inigo
// Quilez's 2D collection (https://iquilezles.org/articles/distfunctions2d/,
// MIT-licensed in his Shadertoy examples). Colours blend in Björn Ottosson's
// Oklab space (https://bottosson.github.io/posts/oklab/; the same matrices are
// in CSS Color 4).
//
// Plain C++ with no ESP-IDF or LVGL dependency, so tools/eyesim can build and
// render the exact same code on a desktop.

#ifndef EYE_ENGINE_H
#define EYE_ENGINE_H

#include <cstddef>
#include <cstdint>

namespace eyes {

// What the eyes turn into. Idle is the plain glass capsule, which the lids,
// blink and gaze shape further.
enum class Shape : uint8_t {
    kIdle,
    kHappy,   // ^ ^
    kLove,    // two beating hearts
    kStar,    // star eyes, stepping round
    kThink,   // loading arcs
    kDone,    // check mark, drawn in one stroke
    kFail,    // cross, drawn stroke by stroke
    kWide,    // round, startled
    kCount,
};

// Two-stop gradient per mood, top colour then bottom colour.
enum class Palette : uint8_t {
    kIdle, kListen, kThink, kSpeak, kHappy, kLove, kStar, kDone, kFail,
    kSad, kAngry, kWide, kSleepy, kCount,
};

// The status ring around the face.
enum class Ring : uint8_t {
    kOff,
    kListen,     // ticks that swell with the voice
    kSpeak,      // livelier ticks
    kComet,      // one arc chasing round: connecting / starting up
    kCelebrate,  // segmented ring, pulsing
    kAlert,      // flashing full ring
};

// Idle-shape lid pose, design pixels (the face is designed at scale 1).
struct LidPose {
    float outer = 0;   // top lid drop at the outer corner
    float inner = 0;   // top lid drop at the inner corner
    float bottom = 0;  // lower lid pushed up (cheeks)
};

struct Rect {
    int x0, y0, x1, y1;   // half-open
};

enum class PixelFormat : uint8_t { kRgb565, kRgb888 };

class Engine {
public:
    static constexpr int kMaxRects = 40;
    static constexpr int kMaxDim = 320;

    // `work` is scratch for the distance grid; WorkFloats() tells how much.
    // RGB565 is native-endian; RGB888 is byte order R, G, B.
    void Init(void* canvas, PixelFormat fmt, int width, int height, int stride_px,
              float* work, size_t work_floats, uint32_t seed);
    static size_t WorkFloats() { return kWorkFloats; }

    void SetStrip(int y0, int y1);          // rows the face may use
    void SetBackground(uint32_t rgb888);
    void SetShape(Shape s);
    void SetPalette(Palette p);
    void SetLids(const LidPose& lids);
    // wander: look around on its own; otherwise hold (x, y) with a slow sway.
    void SetGaze(bool wander, float x, float y);
    void SetRing(Ring r);
    void SetVoice(bool active);
    void SetBlinking(bool on) { blinking_ = on; }
    // Tests: no random glances, blinks, micro-saccades or sparkles.
    void SetDeterministic(bool on) { deterministic_ = on; }
    // Comparing against the JS mock-up: its glow stops dead instead of fading.
    void SetReference(bool on) { reference_ = on; }
    // Evaluate every pixel exactly instead of on the 2 px grid. Slow; tests.
    void SetExact(bool on) { exact_ = on; }

    // Advance the simulation, from a millisecond clock or by a fixed step of at
    // most 50 ms. True when something moved enough to be worth a frame.
    bool Tick(uint32_t now_ms);
    bool Advance(double dt);
    // Renders into the canvas and returns the rectangles that changed.
    int Draw(Rect* rects, int max_rects);
    // Forget what is on the canvas: next Draw clears and repaints the strip.
    void Invalidate() { full_clear_ = true; }

    Shape shape() const { return shape_; }
    Palette palette() const { return palette_; }
    Ring ring() const { return ring_; }
    double time() const { return t_; }
    double last_draw_time() const { return last_draw_t_; }   // what the canvas shows
    uint32_t sdf_evals() const { return sdf_evals_; }      // last Draw
    uint32_t dirty_pixels() const { return dirty_px_; }    // last Draw

    // Times the building blocks on this CPU, for tuning. `now` is any tick
    // counter (the device passes its cycle counter); results are in its units,
    // per call or per draw. Runs a private engine on the scratch canvas given,
    // built in `engine_mem` (sizeof(Engine) bytes) if not null, else on the heap.
    // Per 8x8 block of one RenderEye, when profiling.
    struct BlockProf {
        uint32_t cycles;
        uint16_t flat, glow, edge, evals;
        uint8_t kind;   // 1 all inside (flat fill), 2 mixed
        uint8_t eye;
    };
    struct BenchResult {
        BlockProf* blocks = nullptr;   // caller's buffer, filled for one full draw
        uint32_t (*block_now)() = nullptr;   // counter for the block profile; default `now`
        int max_blocks = 0;
        int nblocks = 0;
        uint32_t setup[2] = {0, 0};    // RenderEye before the block loop, per eye
        // The block loop of both eyes by phase: centres, flat blocks, nodes,
        // glow blocks, sorting, gradients and exact evaluation, lit shading, the rest.
        uint32_t phase[8] = {};
        uint32_t phase_cyc[8] = {};    // the same in cycles
        uint32_t evals_exact = 0;
        // Optional hardware counters around one RenderEye(0): for each config
        // k < pm_configs, pm_start(k) before and pm_stop(k, pm[k]) after.
        void (*pm_start)(int k) = nullptr;
        void (*pm_stop)(int k, uint32_t out[2]) = nullptr;
        int pm_configs = 0;
        uint32_t pm[8][2] = {};
        uint32_t sqrt_lib, sqrt_fast, div_lib, exp_lib, dist_idle;
        uint32_t draw_full, draw_move, draw_exact;
        uint32_t evals, px_dirty, px_flat, px_glow, px_edge;   // one full idle draw
        uint32_t clear, eye0, eye1;                            // its parts
        uint32_t ring_listen, ring_comet, ring_celebrate;      // RenderRing alone
        uint32_t ring_px;                                      // pixels in its box
        float sink;
    };
    static void Benchmark(uint32_t (*now)(), void* canvas, int width, int height, int stride_px,
                          float* work, size_t work_floats, void* engine_mem, BenchResult* out);

private:
    static constexpr int kMaxTerms = 3;
    static constexpr int kGridMax = 84;   // grid nodes per side, 2 px apart
    static constexpr size_t kWorkFloats = kGridMax * kGridMax;
    static constexpr int kSig = 40;

    struct Term {
        uint8_t kind;   // Shape, or the drop circle
        float w;
        float a, b, c;  // drop circle: centre x, centre y, radius
    };
    struct Morph {
        Term from[kMaxTerms];
        int nfrom = 0;
        uint8_t to = 0;
        Term drop{};
        uint8_t phase = 2;     // 1 collapse into the drop, 2 bloom out
        uint8_t style = 0;
        bool via = false, draw = false, pure = true;
        float e = 1, q = 1, qv = 0;
        double t0 = 0;
        double glint = -1;
    };
    struct Xform {
        float cx, cy;          // canvas px
        float sx, sy, sm;      // design -> px scale per axis, and for distances
        float inv_sx, inv_sy;
        float lip;             // bound on how fast the distance changes per px
        int x0, y0, x1, y1;    // render box, half-open
    };
    struct Spark {
        float x, y, s;         // canvas px, scale
        double t0;
    };

    static Term DropFor(Shape s);
    float EvalShape(uint8_t kind, float x, float y, int eye) const;
    float EvalTerm(const Term& t, float x, float y, int eye) const;
    float EvalMorph(const Morph& m, float x, float y, int eye) const;
    float Dist(int eye, float lx, float ly) const;
    void DistN(int eye, int n, const float* lx, const float* ly, float* out) const;
    void TermExtent(const Term& t, float& ex, float& ey) const;
    void MorphBounds(const Morph& m, float& ex, float& ey, float& lip) const;
    int Snapshot(const Morph& m, Term* out) const;
    void StartMorph(Shape s);
    void Direct();
    void StepMorphs(float dt);
    float BlinkAmount();
    void StepGaze(float dt);
    void StepLids(float dt);
    void StepShapeAnim(float dt);
    void StepLayout(float dt);
    void StepSparks();
    void UpdateColors(float dt);
    void ComputeXforms();
    void Signature(float* sig) const;
    float RingRadius() const;
    int RingRects(Rect* out, int max) const;
    float Rand();
    float Rand(float a, float b) { return a + (b - a) * Rand(); }

    template <bool k888>
    int DrawT(Rect* out, int max_out);
    template <bool k888>
    void RenderEye(int eye);
    template <bool k888>
    void RenderSparks();
    template <bool k888>
    void RenderRing();
    template <bool k888>
    void ClearRect(const Rect& r);
    template <bool k888>
    void ReadPixel(int x, int y, float& r, float& g, float& b) const;
    template <bool k888>
    void WritePixel(int x, int y, float r, float g, float b);

    // RenderEye's pieces. PixCtx holds what every pixel of one eye needs; the
    // block functions are kept out of line so each loop gets the registers to
    // itself (one big function spilled nearly everything).
    struct PixCtx {
        int eye;
        int x0, y0, nw, nh;             // render box origin, node grid size
        float cx, cy, inv_s;
        float ga, gp;                   // light sweep after a morph; ga 0: none
        float lx, ly, lz, hx, hy, hz;   // key light, half vector
        float flat_br, flat_ad;
        float cb[3], bg[3], en, cmax;
        const float* glow;
        bool overlap;                   // eye 1, over eye 0's box below
        int ox1, oy0, oy1;
        float flat_max, glow_min, rim_min;
        bool exact_lit;                 // lit pixels evaluated outright, see LitExact
    };
    struct Blk {
        int bx, by, bw, bh;
        int i0, i1, j0, j1;             // its grid nodes
        bool ovl;                       // may lie over eye 0's pixels
    };
    template <bool k888>
    void Under(const PixCtx& c, int px, int py, float& r, float& g, float& b) const;
    float Glint(const PixCtx& c, int px, int py) const;
    void FlatBase(const PixCtx& c, int py, float* fb) const;
    template <bool k888>
    void PutGlow(const PixCtx& c, int px, int py, float d);
    template <bool k888>
    void PutGlowBin(const PixCtx& c, int px, int py, float fb);
    template <bool k888>
    void PutFlat(const PixCtx& c, int px, int py, const float* fb, float rh);
    // A lit pixel's lighting: brightness, highlight, rim (0 where none).
    struct Lit {
        float br, sp, rm;
    };
    Lit Light(const PixCtx& c, float d, float gx, float gy) const;
    template <bool k888>
    void PutLit(const PixCtx& c, int px, int py, float d, float gx, float gy);
    void FillNodes(const PixCtx& c, int ia, int ib, int ja, int jb);
    template <bool k888>
    void FlatBlock(const PixCtx& c, const Blk& b);
    template <bool k888>
    void GlowBlock(const PixCtx& c, const Blk& b);
    template <bool k888, bool kFast>
    int SortBlock(const PixCtx& c, const Blk& b);
    void NodeGrads(const PixCtx& c, const Blk& b);
    void LitInterp(const PixCtx& c, const Blk& b, int nlit);
    void LitExact(const PixCtx& c, const Blk& b, int nlit);
    bool OpenIdle() const;
    template <bool k888>
    void LitShade(const PixCtx& c, const Blk& b, int nlit);
    template <bool k888, bool kFast>
    void OpenEye(const PixCtx& c);
    // One row of OpenEye's lit pixels, and what they need.
    struct OpenRow {
        int py, x0, nlit, xo;          // glow left of column xo goes onto eye 0's
        const uint16_t* lit;           // columns, from x0
        const float* dr;               // this row's distances
        const float* dn;               // the next row's
        int h0lo, h0hi, h1lo, h1hi;    // columns those rows skipped
        float* memo;                   // per column: d, gx, gy, then its Lit
        const float* lx;               // per column: x from the eye's centre
        float ax, ay, sm;              // BoxRow's scales
        float* buf;                    // scratch, 7 floats per lit pixel
    };
    template <bool k888, bool kFast>
    void LitRow(const PixCtx& c, const OpenRow& r);
    void Phase(int k);

    // canvas
    void* canvas_ = nullptr;
    PixelFormat fmt_ = PixelFormat::kRgb565;
    int w_ = 0, h_ = 0, stride_ = 0;
    int strip_y0_ = 0, strip_y1_ = 0;
    int clear_y0_ = 0, clear_y1_ = 0;   // rows the next full clear covers
    float* grid_ = nullptr;
    uint32_t rng_ = 1;
    float bg_[3] = {0, 0, 0};
    uint16_t bg565_ = 0;

    // clock
    bool started_ = false;
    uint32_t last_ms_ = 0;
    double t_ = 0;            // seconds of simulated time

    // inputs
    Shape shape_ = Shape::kIdle;
    int pending_shape_ = -1;
    Palette palette_ = Palette::kIdle;
    LidPose lids_target_;
    bool wander_ = true;
    float hold_x_ = 0, hold_y_ = 0;
    Ring ring_ = Ring::kOff;
    bool voice_ = false;
    bool blinking_ = true;
    bool deterministic_ = false;
    bool reference_ = false;
    bool exact_ = false;

    // morphing, one per eye
    Morph morph_[2];
    double enter_ = 0;

    // shape animation
    float blink_ = 0, zone_lid_ = 0, zone_lid_v_ = 0, wd_ = 1, inv_wd_ = 1;
    LidPose lids_;
    LidPose lids_v_;
    float rc_ = 1, rs_ = 0, ta_ = 0, tb_ = 1;   // think arcs
    float sc_ = 1, ss_ = 0;                     // star rotation
    float beat_ = 0;
    float prog_[2][2] = {{1, 1}, {1, 1}};       // [done/fail][eye] stroke progress
    float pop_[2][2] = {{0, 0}, {0, 0}};
    float pk_ = 1, pk_v_ = 0;                   // startle pop
    float kx_ = 1, ky_ = 1, yo_ = 0, shx_ = 0, en_ = 1;

    // blink
    bool bl_on_ = false, bl_dbl_ = false, bl_half_ = false;
    double bl_start_ = 0, next_blink_ = 1.5;

    // gaze
    float tx_ = 0, ty_ = 0;
    double pick_at_ = 0, picked_ = -1, last_pick_seen_ = -1, zone_at_ = 7;
    bool zoning_ = false;
    float mx_ = 0, my_ = 0;
    double next_micro_ = 0;
    float ex_ = 0, ey_ = 0, evx_ = 0, evy_ = 0;
    float fx_ = 0, fy_ = 0, fvx_ = 0, fvy_ = 0;
    float gx_ = 0, gy_ = 0;

    // layout springs
    float scale_ = 1, scale_v_ = 0;
    float mid_ = 120, mid_v_ = 0;
    bool layout_init_ = false;

    // colour, Oklab
    float oa_[3] = {0, 0, 0}, ob_[3] = {0, 0, 0};
    float lut_[64][3] = {};
    float cb_[3] = {0, 0, 0};
    bool colors_init_ = false;
    bool color_moving_ = true;

    // ring
    float ring_alpha_ = 0;
    float voice_lv_ = .3f;
    Ring ring_shown_ = Ring::kOff;   // style being faded, survives kOff
    bool ring_was_visible_ = false;

    // sparkles (hearts)
    Spark sparks_[8];
    int nsparks_ = 0;
    bool beat_on_ = false;

    // RenderEye's per-block scratch (too big for the LVGL task's stack).
    static constexpr int kBlockPx = 8;
    struct Scratch {
        static constexpr int kBatch = kBlockPx * kBlockPx;   // >= 7x7 nodes and kMaxDim / 8 centres
        float qx[kBatch], qy[kBatch], qd[kBatch];
        int qi[kBatch];
        float dcs[kMaxDim / kBlockPx + 1];
        uint8_t cls[kBlockPx * kBlockPx];
        float dv[kBlockPx * kBlockPx], gxv[kBlockPx * kBlockPx], gyv[kBlockPx * kBlockPx];
        uint8_t lp[kBlockPx * kBlockPx];
        // Per node: gradient, second differences, and the larger of those.
        float gxn[kBlockPx / 2 + 1][kBlockPx / 2 + 1], gyn[kBlockPx / 2 + 1][kBlockPx / 2 + 1];
        float cxn[kBlockPx / 2 + 1][kBlockPx / 2 + 1], cyn[kBlockPx / 2 + 1][kBlockPx / 2 + 1];
        float sdn[kBlockPx / 2 + 1][kBlockPx / 2 + 1];
        float ex[kBlockPx + 1][kBlockPx + 1];      // exact distances, one past each block edge
        uint8_t need[kBlockPx + 1][kBlockPx + 1];
        // The dither for column bx + lx of any block, by row & 3: blocks start
        // x0 + 8k, so it is the same for every block of an eye.
        float dth[4][kBlockPx];
        uint32_t dsw[4][kBlockPx];
    };
    Scratch scratch_;

    // rendering
    Xform xf_[2] = {};
    Rect prev_[kMaxRects];
    int nprev_ = 0;
    bool full_clear_ = true;
    mutable uint32_t sdf_evals_ = 0;
    // Profiling hooks for Benchmark(); off (null) in normal use.
    uint32_t (*prof_now_)() = nullptr;
    BlockProf* prof_ = nullptr;
    int prof_n_ = 0, prof_max_ = 0;
    uint32_t prof_setup_[2] = {0, 0};
    uint32_t prof_phase_[8] = {};   // block loop by phase, see RenderEye
    uint32_t ph_t_ = 0;
    int ph_cur_ = 7;
    uint32_t px_flat_ = 0, px_glow_ = 0, px_edge_ = 0;   // last Draw, by path
    uint32_t dirty_px_ = 0;
    float sig_[kSig] = {};
    float cur_sig_[kSig] = {};
    bool sig_valid_ = false;
    double last_draw_t_ = -1;
    uint8_t row_vi_[kMaxDim] = {};
    float row_hl_[kMaxDim] = {};
    float col_hl_[kMaxDim] = {};
};

}  // namespace eyes

#endif  // EYE_ENGINE_H
