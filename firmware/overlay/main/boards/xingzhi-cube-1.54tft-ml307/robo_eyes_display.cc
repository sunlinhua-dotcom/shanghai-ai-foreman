#include "robo_eyes_display.h"

#include "application.h"
#include "assets.h"
#include "board.h"
#include "assets/lang_config.h"
#include "display/lvgl_display/lvgl_theme.h"
#include "lvgl_private.h"

#include <esp_cpu.h>
#include <esp_lvgl_port.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_random.h>
#include <esp_timer.h>
#include <perfmon.h>
#include <xtensa/xt_perf_consts.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>

#define TAG "RoboEyesDisplay"

// Last-resort font: the one linked into the app. It only carries the ~890
// glyphs the stock UI strings need, so it is a fallback, never the first pick.
LV_FONT_DECLARE(BUILTIN_TEXT_FONT);

// Optional 20 px full-CJK font, present only in the bigger-subtitle assets
// image. See NOTES.md ("字幕字号").
// Any of these will do, biggest first. Whichever one the assets partition
// actually carries becomes the chrome + echo font.
static const char* const kSmallCjkFontAssets[] = {
    "font_puhui_common_20_4.bin",
    "font_puhui_common_16_4.bin",
    "font_puhui_common_14_1.bin",
};

namespace {

using eyes::LidPose;
using eyes::Palette;
using eyes::Ring;
using eyes::Shape;

// ---- palette (fixed dark theme, see NOTES.md) ---------------------------
constexpr uint32_t kColorBg = 0x101214;      // deep grey ground, the eyes' too
constexpr uint32_t kColorEye = 0xF2EEE6;     // warm white
constexpr uint32_t kColorCoral = 0xE0876A;   // accent, used sparingly
constexpr uint32_t kColorDim = 0x7C838A;     // secondary text

// ---- layout (240 x 240) -------------------------------------------------
// The canvas is the whole screen and the eyes own every row of it. The top bar,
// the text band and the status strip are opaque objects parked on top, and the
// eyes are told to live inside whatever rows are left over -- so glyphs and
// eyes never share a pixel, and the eyes get back every row nobody is using.
constexpr int kScreenW = 240;
constexpr int kScreenH = 240;
constexpr int kTopBarHeight = 26;
constexpr int kCanvasY = 0;
constexpr int kCanvasW = kScreenW;
constexpr int kCanvasH = kScreenH;
constexpr int kEyeTop = kTopBarHeight;   // first canvas row the eyes may use

constexpr int kTextZoneW = kScreenW;
constexpr int kTextPad = 4;
constexpr int kTextLineW = 224;
// The band is sized from the text instead of being a fixed slab: one echo line
// plus as many subtitle lines as fit. Past that LV_LABEL_LONG_DOT clips.
constexpr int kBandMaxH = 150;
constexpr int kStatusPadV = 5;           // above and below the status line
constexpr int kEyeAreaMin = 56;          // never squeeze the eyes below this
constexpr int kEyeAreaFull = kScreenH - kTopBarHeight;   // 214

// 50 fps while something moves. Still eyes cost nothing -- the engine only
// draws when a frame would differ.
constexpr int kFramePeriodMs = 20;
constexpr int kPowerSaveFramePeriodMs = 200;
// A frame may start this much early: LVGL's task wakes on events and 10 ms
// ticks, and the next wake could be a tick away.
constexpr int64_t kFrameEarlyUs = 4000;

constexpr uint32_t kFlashMs = 3000;      // ring after "done" / an error
constexpr uint32_t kLingerMs = 3000;     // a feeling outlasts its turn this long
constexpr uint32_t kOverrideMs = 8000;   // `eye <name>` from the USB console
constexpr uint32_t kDetMs = 120000;      // `eye det`, however it was left

// The 21 emotion values the server can send in an `llm` message, four for
// finished / failed tasks, and the icon names the app's own alerts pass
// instead of an emotion. Unknown names are neutral. See NOTES.md.
//   lids: top-lid drop at the outer / inner corner, cheek lift (design px)
//   look: hold the gaze at (gx, gy) design px; the eyes roam about +-26 x +-16
const EmotionStyle kEmotionStyles[] = {
    // name                  shape          palette           lids          look    gx   gy  flash             error
    {"neutral",              Shape::kIdle,  Palette::kCount,  {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"happy",                Shape::kHappy, Palette::kHappy,  {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"laughing",             Shape::kHappy, Palette::kHappy,  {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"funny",                Shape::kHappy, Palette::kHappy,  {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"delicious",            Shape::kHappy, Palette::kHappy,  {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"winking",              Shape::kHappy, Palette::kHappy,  {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"silly",                Shape::kHappy, Palette::kHappy,  {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"loving",               Shape::kLove,  Palette::kLove,   {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"kissy",                Shape::kLove,  Palette::kLove,   {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"surprised",            Shape::kStar,  Palette::kStar,   {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"shocked",              Shape::kWide,  Palette::kWide,   {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"thinking",             Shape::kThink, Palette::kThink,  {0, 0, 0},    false,   0,   0, Ring::kOff,       false},
    {"done",                 Shape::kDone,  Palette::kDone,   {0, 0, 0},    false,   0,   0, Ring::kCelebrate, false},
    {"success",              Shape::kDone,  Palette::kDone,   {0, 0, 0},    false,   0,   0, Ring::kCelebrate, false},
    {"fail",                 Shape::kFail,  Palette::kFail,   {0, 0, 0},    false,   0,   0, Ring::kAlert,     false},
    {"error",                Shape::kFail,  Palette::kFail,   {0, 0, 0},    false,   0,   0, Ring::kAlert,     false},
    {"sad",                  Shape::kIdle,  Palette::kSad,    {18, 0, 0},   true,    0,   8, Ring::kOff,       false},
    {"crying",               Shape::kIdle,  Palette::kSad,    {22, 0, 0},   true,    0,  10, Ring::kOff,       false},
    {"angry",                Shape::kIdle,  Palette::kAngry,  {0, 20, 0},   false,   0,   0, Ring::kOff,       false},
    {"sleepy",               Shape::kIdle,  Palette::kSleepy, {30, 30, 0},  true,    0,   4, Ring::kOff,       false},
    {"cool",                 Shape::kIdle,  Palette::kCount,  {22, 22, 8},  false,   0,   0, Ring::kOff,       false},
    {"relaxed",              Shape::kIdle,  Palette::kCount,  {14, 14, 6},  false,   0,   0, Ring::kOff,       false},
    {"confident",            Shape::kIdle,  Palette::kCount,  {6, 6, 6},    false,   0,   0, Ring::kOff,       false},
    {"embarrassed",          Shape::kIdle,  Palette::kCount,  {8, 0, 8},    true,  -12,  10, Ring::kOff,       false},
    {"confused",             Shape::kIdle,  Palette::kCount,  {0, 10, 0},   true,  -14, -10, Ring::kOff,       false},
    {"circle_xmark",         Shape::kFail,  Palette::kFail,   {0, 0, 0},    false,   0,   0, Ring::kAlert,     true},
    {"triangle_exclamation", Shape::kFail,  Palette::kFail,   {0, 0, 0},    false,   0,   0, Ring::kAlert,     true},
    {"cloud_slash",          Shape::kFail,  Palette::kFail,   {0, 0, 0},    false,   0,   0, Ring::kAlert,     true},
};
const EmotionStyle* const kNeutral = &kEmotionStyles[0];

// nullptr when the name is not in the table.
const EmotionStyle* FindEmotion(const char* name) {
    for (const auto& style : kEmotionStyles) {
        if (strcmp(style.name, name) == 0) return &style;
    }
    return nullptr;
}

inline uint32_t NowMs() {
    return static_cast<uint32_t>(esp_timer_get_time() / 1000);
}

inline uint16_t Rgb565(uint32_t rgb888) {
    const uint32_t r = (rgb888 >> 16) & 0xFF;
    const uint32_t g = (rgb888 >> 8) & 0xFF;
    const uint32_t b = rgb888 & 0xFF;
    return static_cast<uint16_t>(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}

// ---- `eye demo`: the same tour `eyesim live` renders ----------------------
struct Cue {
    float t = 0;
    int shape = -1, palette = -1, ring = -1, voice = -1, y1 = -1;
    bool set_lids = false, set_gaze = false, wander = true;
    LidPose lids;
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
Cue At(float t) {
    Cue c;
    c.t = t;
    return c;
}

constexpr float kDemoSec = 34;
const Cue kDemoCues[] = {
    At(0).S(Shape::kIdle).P(Palette::kIdle).R(Ring::kOff).Lids(0, 0, 0).Wander().Strip(kScreenH),
    // listening: ticks swell while the voice is on
    At(4.5f).P(Palette::kListen).R(Ring::kListen).Look(0, -4),
    At(5.2f).V(true), At(6.0f).V(false), At(6.4f).V(true), At(7.4f).V(false),
    At(8.0f).S(Shape::kThink).P(Palette::kThink).R(Ring::kOff),
    // speaking with subtitles: the strip shrinks, the ring has no room
    At(10.4f).S(Shape::kHappy).P(Palette::kHappy).R(Ring::kSpeak).Look(0, 2).Strip(150),
    At(13.0f).S(Shape::kLove).P(Palette::kLove),
    At(15.4f).S(Shape::kStar).P(Palette::kStar).R(Ring::kOff).Strip(kScreenH),
    At(17.6f).S(Shape::kDone).P(Palette::kDone).R(Ring::kCelebrate),
    At(20.2f).S(Shape::kFail).P(Palette::kFail).R(Ring::kAlert),
    At(22.6f).S(Shape::kWide).P(Palette::kWide).R(Ring::kOff),
    At(24.4f).S(Shape::kIdle).P(Palette::kSad).Lids(18, 0, 0).Look(0, 8),
    At(26.6f).P(Palette::kAngry).Lids(0, 20, 0).Look(0, 0),
    At(28.6f).P(Palette::kSleepy).Lids(30, 30, 0),
    At(30.6f).P(Palette::kIdle).Lids(0, 0, 0).Wander(),
};
constexpr int kDemoCueCount = sizeof(kDemoCues) / sizeof(kDemoCues[0]);

const char* const kRingNames[] = {"off", "listen", "speak", "comet", "celebrate", "alert"};

// LVGL's task: above the wake word's AFE worker (1), with which it would
// otherwise take 10 ms turns on core 1, and below every audio task.
constexpr UBaseType_t kLvglPriority = 2;

// The engine's core. Core 0 carries the custom wake word (MultiNet runs inside
// the audio input task, priority 8), which took the engine from 12 ms a frame
// to 25. On core 1 it shares with LVGL, which sleeps through each SPI transfer.
// A task that touches the FPU is pinned where it first did on the S3 anyway, so
// no affinity would mean a random core.
constexpr BaseType_t kEngineCore = 1;

// The SPI transfer LVGL is waiting for is over.
SemaphoreHandle_t g_flush_done = nullptr;

// From the SPI interrupt: what esp_lvgl_port's own callback does, and wake
// LVGL's task.
bool OnFlushDone(esp_lcd_panel_io_handle_t, esp_lcd_panel_io_event_data_t*, void* ctx) {
    lv_display_flush_ready(static_cast<lv_display_t*>(ctx));
    BaseType_t woken = pdFALSE;
    xSemaphoreGiveFromISR(g_flush_done, &woken);
    return woken == pdTRUE;
}

// LVGL waits for a transfer asleep, leaving the core to others, instead of
// spinning on the flag for the 8 ms or so a frame's transfers take.
void WaitFlush(lv_display_t* disp) {
    while (disp->flushing) {
        xSemaphoreTake(g_flush_done, pdMS_TO_TICKS(50));
    }
}

}  // namespace

RoboEyesDisplay::RoboEyesDisplay(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_handle_t panel,
                                 int width, int height, int offset_x, int offset_y,
                                 bool mirror_x, bool mirror_y, bool swap_xy)
    : SpiLcdDisplay(panel_io, panel, width, height, offset_x, offset_y,
                    mirror_x, mirror_y, swap_xy) {
    // The palette below is fixed; the theme object is only consulted for fonts,
    // which assets.bin may swap at runtime. Pin it to "dark" so any font the
    // asset loader installs on the dark theme is the one we read.
    auto* dark = LvglThemeManager::GetInstance().GetTheme("dark");
    if (dark != nullptr) current_theme_ = dark;
    emotion_ = kNeutral;
}

RoboEyesDisplay::~RoboEyesDisplay() {
    if (engine_task_ != nullptr) {
        engine_quit_ = true;
        xSemaphoreGive(engine_go_);
        xSemaphoreTake(engine_done_, portMAX_DELAY);
        engine_task_ = nullptr;
    }
    if (engine_go_ != nullptr) vSemaphoreDelete(engine_go_);
    if (engine_done_ != nullptr) vSemaphoreDelete(engine_done_);
    if (display_ != nullptr) {
        lv_display_remove_event_cb_with_user_data(display_, RefrEventCb, this);
    }
    if (frame_timer_ != nullptr) {
        lv_timer_delete(frame_timer_);
        frame_timer_ = nullptr;
    }
    if (frame_wake_ != nullptr) {
        esp_timer_stop(frame_wake_);
        esp_timer_delete(frame_wake_);
        frame_wake_ = nullptr;
    }
    // canvas_ is aliased to content_, which ~LcdDisplay() deletes.
    if (canvas_buf_ != nullptr) {
        heap_caps_free(canvas_buf_);
        canvas_buf_ = nullptr;
    }
    if (back_buf_ != nullptr) {
        heap_caps_free(back_buf_);
        back_buf_ = nullptr;
    }
    if (eye_work_ != nullptr) {
        heap_caps_free(eye_work_);
        eye_work_ = nullptr;
    }
    if (snap_.px != nullptr) {
        heap_caps_free(snap_.px);
        snap_.px = nullptr;
    }
}

// ---------------------------------------------------------------------------
// UI construction
// ---------------------------------------------------------------------------

void RoboEyesDisplay::SetupUI() {
    if (setup_ui_called_) {
        ESP_LOGW(TAG, "SetupUI() called twice, ignoring");
        return;
    }
    Display::SetupUI();
    DisplayLockGuard lock(this);
    EnableDoubleBuffer();

    auto* theme = static_cast<LvglTheme*>(current_theme_);
    const lv_font_t* icon_font = theme->icon_font()->font();
    ResolveFonts();

    lv_obj_t* screen = lv_screen_active();
    // Nothing else is supposed to have drawn on the active screen yet, but a
    // leftover widget here would sit *under* the eyes and show through. Start
    // from an empty screen and say so if anything had to go.
    const uint32_t stale_children = lv_obj_get_child_count(screen);
    if (stale_children > 0) {
        ESP_LOGW(TAG, "active screen already had %u children, clearing them",
                 (unsigned)stale_children);
        lv_obj_clean(screen);
    }
    lv_obj_set_style_bg_color(screen, lv_color_hex(kColorBg), 0);
    lv_obj_set_style_bg_opa(screen, LV_OPA_COVER, 0);
    lv_obj_set_style_text_color(screen, lv_color_hex(kColorEye), 0);
    lv_obj_set_style_text_font(screen, ui_font_, 0);
    lv_obj_set_style_pad_all(screen, 0, 0);

    BuildEyeCanvas(screen);
    BuildTextZone(screen);
    BuildTopBar(screen, icon_font);
    BuildBottomBar(screen);

    // Low battery banner: LvglDisplay::UpdateStatusBar() drives it.
    low_battery_popup_ = lv_obj_create(screen);
    lv_obj_set_scrollbar_mode(low_battery_popup_, LV_SCROLLBAR_MODE_OFF);
    lv_obj_set_size(low_battery_popup_, 200, 40);
    lv_obj_align(low_battery_popup_, LV_ALIGN_BOTTOM_MID, 0, -8);
    lv_obj_set_style_bg_color(low_battery_popup_, lv_color_hex(kColorCoral), 0);
    lv_obj_set_style_border_width(low_battery_popup_, 0, 0);
    lv_obj_set_style_radius(low_battery_popup_, 12, 0);
    low_battery_label_ = lv_label_create(low_battery_popup_);
    lv_obj_set_style_text_font(low_battery_label_, ui_font_, 0);
    lv_label_set_text(low_battery_label_, Lang::Strings::BATTERY_NEED_CHARGE);
    lv_obj_set_style_text_color(low_battery_label_, lv_color_hex(0x101214), 0);
    lv_obj_center(low_battery_label_);
    lv_obj_add_flag(low_battery_popup_, LV_OBJ_FLAG_HIDDEN);

    now_ms_ = NowMs();
    eye_area_h_ = kEyeAreaFull;
    ApplyPose(EyeUiState::kBoot);

    // Every one of the 240 rows belongs to an opaque object now; repaint the
    // lot once so nothing from before this UI can survive underneath.
    lv_obj_invalidate(screen);

    engine_go_ = xSemaphoreCreateBinary();
    engine_done_ = xSemaphoreCreateBinary();
    if (engine_go_ == nullptr || engine_done_ == nullptr ||
        xTaskCreatePinnedToCore(EngineTask, "eyes", 6144, this, 1, &engine_task_,
                                kEngineCore) != pdPASS) {
        engine_task_ = nullptr;
        ESP_LOGW(TAG, "no engine task: the eyes draw on LVGL's");
    }

    g_flush_done = xSemaphoreCreateBinary();
    if (g_flush_done != nullptr && panel_io_ != nullptr && display_ != nullptr) {
        const esp_lcd_panel_io_callbacks_t cbs = {.on_color_trans_done = OnFlushDone};
        if (esp_lcd_panel_io_register_event_callbacks(panel_io_, &cbs, display_) == ESP_OK) {
            lv_display_set_flush_wait_cb(display_, WaitFlush);
        }
    }
    TaskHandle_t lvgl_task = xTaskGetHandle("taskLVGL");
    if (lvgl_task != nullptr) {
        vTaskPrioritySet(lvgl_task, kLvglPriority);
    }

    // LVGL looks in on every pass; OnFrame keeps the time itself.
    frame_period_ms_ = kFramePeriodMs;
    const esp_timer_create_args_t wake_args = {
        .callback = FrameWakeCb,
        .arg = this,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "eye_frame",
        .skip_unhandled_events = true,
    };
    if (esp_timer_create(&wake_args, &frame_wake_) != ESP_OK) {
        frame_wake_ = nullptr;
    }
    frame_timer_ = lv_timer_create(FrameTimerCb, 1, this);
    if (display_ != nullptr) {
        // LVGL's own 33 ms refresh would show only every other eye frame.
        lv_timer_t* refr = lv_display_get_refr_timer(display_);
        if (refr != nullptr) {
            lv_timer_set_period(refr, kFramePeriodMs);
        }
        // For `eye stats`: how long LVGL takes to render and push a frame.
        lv_display_add_event_cb(display_, RefrEventCb, LV_EVENT_RENDER_START, this);
        lv_display_add_event_cb(display_, RefrEventCb, LV_EVENT_REFR_READY, this);
    }
    stats_.since_ms = now_ms_;
    ESP_LOGI(TAG, "Eyes UI ready: canvas %dx%d, %d ms frames, grid in %s RAM", kCanvasW, kCanvasH,
             kFramePeriodMs, eye_work_internal_ ? "internal" : "PSRAM");
}

void RoboEyesDisplay::ResolveFonts() {
    auto* theme = static_cast<LvglTheme*>(current_theme_);
    const lv_font_t* theme_font =
        theme != nullptr && theme->text_font() != nullptr ? theme->text_font()->font() : nullptr;
    if (theme_font == nullptr) {
        theme_font = &BUILTIN_TEXT_FONT;
    }
    subtitle_font_ = theme_font;
    observed_theme_font_ = theme_font;

    // Look for a small full-CJK face in the assets partition. The theme font is
    // whatever the image was built with (30 px in the big-subtitle image), which
    // is far too tall for a 26 px bar.
    const lv_font_t* small = nullptr;
    for (const char* name : kSmallCjkFontAssets) {
        void* ptr = nullptr;
        size_t size = 0;
        if (!Assets::GetInstance().GetAssetData(name, ptr, size) || ptr == nullptr) {
            continue;
        }
        auto font = std::make_shared<LvglCBinFont>(ptr);
        if (font->font() == nullptr) {
            continue;
        }
        extra_ui_font_ = font;
        small = font->font();
        break;
    }

    ui_font_ = small != nullptr ? small : theme_font;
    // Never let a tall face into the top bar. BUILTIN_TEXT_FONT only knows ~200
    // CJK glyphs, but the chrome prints stock strings, so it can live with that.
    if (static_cast<int>(ui_font_->line_height) > kTopBarHeight) {
        ui_font_ = &BUILTIN_TEXT_FONT;
    }
    // The echo line prints whatever the user just said, i.e. arbitrary Chinese.
    // BUILTIN_TEXT_FONT would render most of it blank -- that is the "my words
    // never show up" bug -- so it is never a candidate here.
    echo_font_ = small != nullptr ? small : theme_font;

    ESP_LOGI(TAG, "fonts: ui line_height=%d, echo line_height=%d, subtitle line_height=%d",
             (int)ui_font_->line_height, (int)echo_font_->line_height,
             (int)subtitle_font_->line_height);
}

// Assets::Apply() happens in Application::CheckAssetsVersion(), long after
// SetupUI(). Until then the theme still carries the ~890-glyph built-in font,
// which cannot render arbitrary Chinese. Re-resolve once the swap shows up.
void RoboEyesDisplay::MaybeRefreshFonts() {
    auto* theme = static_cast<LvglTheme*>(current_theme_);
    if (theme == nullptr || theme->text_font() == nullptr) return;
    const lv_font_t* now = theme->text_font()->font();
    if (now == nullptr || now == observed_theme_font_) return;

    // Keep the outgoing cbin font alive until the widgets stop pointing at it.
    std::shared_ptr<LvglFont> previous = extra_ui_font_;
    extra_ui_font_.reset();
    ui_font_ = nullptr;
    echo_font_ = nullptr;
    ResolveFonts();
    ApplyFontsToWidgets();
}

void RoboEyesDisplay::ApplyFontsToWidgets() {
    lv_obj_set_style_text_font(lv_screen_active(), ui_font_, 0);
    if (time_label_ != nullptr) lv_obj_set_style_text_font(time_label_, ui_font_, 0);
    if (status_label_ != nullptr) lv_obj_set_style_text_font(status_label_, ui_font_, 0);
    if (notification_label_ != nullptr) {
        lv_obj_set_style_text_font(notification_label_, ui_font_, 0);
    }
    if (low_battery_label_ != nullptr) {
        lv_obj_set_style_text_font(low_battery_label_, ui_font_, 0);
    }
    if (chat_message_label_ != nullptr) {
        lv_obj_set_style_text_font(chat_message_label_, subtitle_font_, 0);
    }
    if (user_label_ != nullptr) {
        lv_obj_set_style_text_font(user_label_, echo_font_, 0);
    }
    // Line heights just changed, so the band has to be measured again.
    ApplyPose(ui_state_);
}

void RoboEyesDisplay::BuildEyeCanvas(lv_obj_t* screen) {
    const int32_t stride = lv_draw_buf_width_to_stride(kCanvasW, LV_COLOR_FORMAT_RGB565);
    const size_t buf_size = static_cast<size_t>(stride) * kCanvasH;
    canvas_buf_ = static_cast<uint16_t*>(
        heap_caps_aligned_alloc(64, buf_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (canvas_buf_ == nullptr) {
        ESP_LOGW(TAG, "PSRAM canvas alloc failed, falling back to internal RAM");
        canvas_buf_ = static_cast<uint16_t*>(heap_caps_aligned_alloc(64, buf_size, MALLOC_CAP_8BIT));
    }
    if (canvas_buf_ == nullptr) {
        ESP_LOGE(TAG, "Cannot allocate %u byte eye canvas", (unsigned)buf_size);
        return;
    }
    ESP_LOGI(TAG, "Eye canvas: %u bytes, stride %d", (unsigned)buf_size, (int)stride);
    canvas_stride_px_ = stride / 2;
    const uint16_t bg = Rgb565(kColorBg);
    for (size_t i = 0; i < buf_size / 2; i++) {
        canvas_buf_[i] = bg;
    }
    // Without it the engine draws on canvas_buf_, one frame at a time.
    back_buf_ = static_cast<uint16_t*>(heap_caps_aligned_alloc(64, buf_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (back_buf_ != nullptr) {
        memcpy(back_buf_, canvas_buf_, buf_size);
    }

    // The engine's distance grid is hit thousands of times a frame. Internal RAM
    // when it can be spared (steady state leaves ~110 KB free before this); the
    // engine falls back to exact per-pixel evaluation if there is no grid at all.
    const size_t work_bytes = eyes::Engine::WorkFloats() * sizeof(float);
    if (heap_caps_get_free_size(MALLOC_CAP_INTERNAL) > work_bytes + 96 * 1024) {
        eye_work_ = static_cast<float*>(heap_caps_malloc(work_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        eye_work_internal_ = eye_work_ != nullptr;
    }
    if (eye_work_ == nullptr) {
        eye_work_ = static_cast<float*>(heap_caps_malloc(work_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    }
    eyes_.Init(back_buf_ != nullptr ? back_buf_ : canvas_buf_, eyes::PixelFormat::kRgb565, kCanvasW, kCanvasH, canvas_stride_px_,
               eye_work_, eye_work_ != nullptr ? eyes::Engine::WorkFloats() : 0, esp_random());
    eyes_.SetBackground(kColorBg);
    eyes_.SetStrip(kEyeTop, kEyeTop + kEyeAreaFull);

    canvas_ = lv_canvas_create(screen);
    lv_canvas_set_buffer(canvas_, canvas_buf_, kCanvasW, kCanvasH, LV_COLOR_FORMAT_RGB565);
    lv_obj_set_pos(canvas_, 0, kCanvasY);
    lv_obj_set_size(canvas_, kCanvasW, kCanvasH);
    lv_obj_remove_flag(canvas_, LV_OBJ_FLAG_SCROLLABLE);
    // ~LcdDisplay() frees content_; reuse that slot so the canvas is cleaned up.
    content_ = canvas_;
}

// The text band. It is a sibling of the canvas created *after* it, so it draws
// on top, and it is opaque, so the eyes cannot bleed through it. It still only
// ever covers rows the eyes have been told to stay out of.
void RoboEyesDisplay::BuildTextZone(lv_obj_t* screen) {
    text_zone_ = lv_obj_create(screen);
    // Size and position are decided per layout pass in ApplyTextLayout.
    lv_obj_set_size(text_zone_, kTextZoneW, 1);
    lv_obj_set_pos(text_zone_, 0, kScreenH - 1);
    lv_obj_set_style_radius(text_zone_, 0, 0);
    lv_obj_set_style_bg_color(text_zone_, lv_color_hex(kColorBg), 0);
    lv_obj_set_style_bg_opa(text_zone_, LV_OPA_COVER, 0);
    lv_obj_set_style_border_width(text_zone_, 0, 0);
    lv_obj_set_style_pad_all(text_zone_, 0, 0);
    lv_obj_set_style_layout(text_zone_, LV_LAYOUT_NONE, 0);
    lv_obj_remove_flag(text_zone_, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_add_flag(text_zone_, LV_OBJ_FLAG_HIDDEN);

    // What the microphone heard, one dim line. Dropping this is what made the
    // user's own words invisible before.
    user_label_ = lv_label_create(text_zone_);
    lv_obj_set_width(user_label_, kTextLineW);
    lv_label_set_long_mode(user_label_, LV_LABEL_LONG_DOT);
    lv_obj_set_style_text_align(user_label_, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_color(user_label_, lv_color_hex(kColorDim), 0);
    lv_obj_set_style_text_font(user_label_, echo_font_, 0);
    lv_obj_set_style_text_line_space(user_label_, 0, 0);
    lv_label_set_text(user_label_, "");
    lv_obj_add_flag(user_label_, LV_OBJ_FLAG_HIDDEN);

    // What the assistant is saying, or a system alert.
    chat_message_label_ = lv_label_create(text_zone_);
    lv_obj_set_width(chat_message_label_, kTextLineW);
    lv_label_set_long_mode(chat_message_label_, LV_LABEL_LONG_DOT);
    lv_obj_set_style_text_align(chat_message_label_, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_color(chat_message_label_, lv_color_hex(kColorEye), 0);
    lv_obj_set_style_text_font(chat_message_label_, subtitle_font_, 0);
    lv_obj_set_style_text_line_space(chat_message_label_, 2, 0);
    lv_label_set_text(chat_message_label_, "");
    lv_obj_add_flag(chat_message_label_, LV_OBJ_FLAG_HIDDEN);
}

// Measures how tall `text` renders at kTextLineW in `font`. Pure measurement,
// no layout pass, so it is safe to call while deciding where things go.
int RoboEyesDisplay::MeasureTextH(const char* text, const lv_font_t* font,
                                  int line_space) const {
    if (text == nullptr || *text == '\0' || font == nullptr) {
        return 0;
    }
    lv_point_t size;
    lv_text_get_size(&size, text, font, 0, line_space, kTextLineW, LV_TEXT_FLAG_NONE);
    return size.y;
}

// Is there anything worth giving the bottom strip rows for? Read it straight
// off the labels: LvglDisplay flips their HIDDEN flags behind our back (the
// notification timer un-hides the status label 3 s later without telling us).
bool RoboEyesDisplay::StatusStripWanted() const {
    if (notification_label_ != nullptr &&
        !lv_obj_has_flag(notification_label_, LV_OBJ_FLAG_HIDDEN)) {
        const char* t = lv_label_get_text(notification_label_);
        if (t != nullptr && *t != '\0') {
            return true;
        }
    }
    if (status_label_ != nullptr &&
        !lv_obj_has_flag(status_label_, LV_OBJ_FLAG_HIDDEN)) {
        const char* t = lv_label_get_text(status_label_);
        if (t != nullptr && *t != '\0') {
            return true;
        }
    }
    return false;
}

// Lays the bottom stack -- [text band][status strip] -- out from the bottom of
// the screen upwards, sized from the text that is actually in it, and hands
// every row above it to the eyes. Both parts collapse to nothing when they have
// nothing to show, which is the normal case: then the eyes get all 214 rows.
void RoboEyesDisplay::ApplyTextLayout(bool show_user, bool show_subtitle) {
    if (text_zone_ == nullptr) {
        return;
    }

    // --- status strip: one line of chrome, only when there is chrome text ---
    int status_h = 0;
    if (ui_font_ != nullptr && StatusStripWanted()) {
        status_h = static_cast<int>(ui_font_->line_height) + 2 * kStatusPadV;
    }

    // --- text band: exactly as tall as the text needs -----------------------
    // The echo is deliberately kept to one line; a long utterance gets an
    // ellipsis rather than pushing the subtitle off the screen.
    const int user_h = (show_user && echo_font_ != nullptr)
                           ? static_cast<int>(echo_font_->line_height) : 0;
    int sub_h = show_subtitle ? MeasureTextH(subtitle_text_.c_str(), subtitle_font_, 2) : 0;
    const int gap = (user_h > 0 && sub_h > 0) ? 4 : 0;
    int band_h = 0;
    if (user_h > 0 || sub_h > 0) {
        band_h = kTextPad + user_h + gap + sub_h + kTextPad;
        int room = kScreenH - kTopBarHeight - kEyeAreaMin - status_h;
        if (room > kBandMaxH) {
            room = kBandMaxH;
        }
        if (room < 0) {
            room = 0;
        }
        if (band_h > room) {
            // Too much to say: give the subtitle what is left and let
            // LV_LABEL_LONG_DOT put an ellipsis on the rest.
            band_h = room;
            sub_h = band_h - 2 * kTextPad - user_h - gap;
            if (sub_h < 0) {
                sub_h = 0;
            }
        }
    }

    // Rows may just have changed hands; the eye engine clears whatever it
    // gives up or takes over.
    band_h_ = band_h;
    status_h_ = status_h;

    // --- place it all -------------------------------------------------------
    const int status_y = kScreenH - status_h;
    const int band_y = status_y - band_h;

    if (status_bar_ != nullptr) {
        lv_obj_set_size(status_bar_, kScreenW, status_h > 0 ? status_h : 1);
        lv_obj_set_pos(status_bar_, 0, status_h > 0 ? status_y : kScreenH - 1);
        if (status_h > 0) {
            lv_obj_remove_flag(status_bar_, LV_OBJ_FLAG_HIDDEN);
        } else {
            lv_obj_add_flag(status_bar_, LV_OBJ_FLAG_HIDDEN);
        }
        if (status_label_ != nullptr) {
            lv_obj_align(status_label_, LV_ALIGN_CENTER, 0, 0);
        }
        if (notification_label_ != nullptr) {
            lv_obj_align(notification_label_, LV_ALIGN_CENTER, 0, 0);
        }
    }

    text_zone_shown_ = band_h > 0;
    lv_obj_set_size(text_zone_, kTextZoneW, band_h > 0 ? band_h : 1);
    lv_obj_set_pos(text_zone_, 0, band_h > 0 ? band_y : kScreenH - 1);
    if (text_zone_shown_) {
        lv_obj_remove_flag(text_zone_, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_add_flag(text_zone_, LV_OBJ_FLAG_HIDDEN);
    }

    const int x = (kTextZoneW - kTextLineW) / 2;
    int y = kTextPad;
    if (user_h > 0) {
        lv_obj_set_pos(user_label_, x, y);
        lv_obj_set_height(user_label_, user_h);
        lv_obj_remove_flag(user_label_, LV_OBJ_FLAG_HIDDEN);
        y += user_h + gap;
    } else {
        lv_obj_add_flag(user_label_, LV_OBJ_FLAG_HIDDEN);
    }
    if (sub_h > 0) {
        lv_obj_set_pos(chat_message_label_, x, y);
        lv_obj_set_height(chat_message_label_, sub_h);
        lv_obj_remove_flag(chat_message_label_, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_add_flag(chat_message_label_, LV_OBJ_FLAG_HIDDEN);
    }

    // Everything above the band belongs to the eyes.
    SetEyeArea(band_y - kEyeTop);
}

void RoboEyesDisplay::SetEyeArea(int height) {
    // Never let the band starve the eyes into a letterbox: below this they stop
    // reading as eyes at all, and the band clips its text instead.
    if (height < kEyeAreaMin) {
        height = kEyeAreaMin;
    }
    if (height == eye_area_h_) {
        return;
    }
    eye_area_h_ = height;
    // The engine re-fits the face with a spring and clears the rows it leaves.
    // `eye demo` moves the strip itself and hands it back when it ends.
    if (!demo_) {
        WaitEngine();
        eyes_.SetStrip(kEyeTop, kEyeTop + height);
    }
}

void RoboEyesDisplay::BuildTopBar(lv_obj_t* screen, const lv_font_t* icon_font) {
    top_bar_ = lv_obj_create(screen);
    lv_obj_set_size(top_bar_, 240, kTopBarHeight);
    lv_obj_set_pos(top_bar_, 0, 0);
    lv_obj_set_style_radius(top_bar_, 0, 0);
    lv_obj_set_style_bg_color(top_bar_, lv_color_hex(kColorBg), 0);
    lv_obj_set_style_bg_opa(top_bar_, LV_OPA_COVER, 0);
    lv_obj_set_style_border_width(top_bar_, 0, 0);
    lv_obj_set_style_pad_all(top_bar_, 0, 0);
    lv_obj_set_style_layout(top_bar_, LV_LAYOUT_NONE, 0);
    lv_obj_remove_flag(top_bar_, LV_OBJ_FLAG_SCROLLABLE);

    network_label_ = lv_label_create(top_bar_);
    lv_label_set_text(network_label_, "");
    lv_obj_set_style_text_font(network_label_, icon_font, 0);
    lv_obj_set_style_text_color(network_label_, lv_color_hex(kColorDim), 0);
    lv_obj_align(network_label_, LV_ALIGN_LEFT_MID, 8, 0);

    time_label_ = lv_label_create(top_bar_);
    lv_label_set_text(time_label_, "--:--");
    lv_obj_set_style_text_font(time_label_, ui_font_, 0);
    lv_obj_set_style_text_color(time_label_, lv_color_hex(kColorEye), 0);
    lv_obj_align(time_label_, LV_ALIGN_CENTER, 0, 0);

    battery_label_ = lv_label_create(top_bar_);
    lv_label_set_text(battery_label_, "");
    lv_obj_set_style_text_font(battery_label_, icon_font, 0);
    lv_obj_set_style_text_color(battery_label_, lv_color_hex(kColorDim), 0);
    lv_obj_align(battery_label_, LV_ALIGN_RIGHT_MID, -8, 0);

    mute_label_ = lv_label_create(top_bar_);
    lv_label_set_text(mute_label_, "");
    lv_obj_set_style_text_font(mute_label_, icon_font, 0);
    lv_obj_set_style_text_color(mute_label_, lv_color_hex(kColorDim), 0);
    lv_obj_align(mute_label_, LV_ALIGN_RIGHT_MID, -34, 0);

    // Coral dot: the one place the accent colour appears in the top bar.
    link_dot_ = lv_obj_create(top_bar_);
    lv_obj_set_size(link_dot_, 8, 8);
    lv_obj_set_style_radius(link_dot_, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_style_bg_color(link_dot_, lv_color_hex(kColorCoral), 0);
    lv_obj_set_style_border_width(link_dot_, 0, 0);
    lv_obj_set_style_pad_all(link_dot_, 0, 0);
    lv_obj_remove_flag(link_dot_, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_align(link_dot_, LV_ALIGN_RIGHT_MID, -58, 0);
    lv_obj_set_style_bg_opa(link_dot_, LV_OPA_TRANSP, 0);
}

void RoboEyesDisplay::BuildBottomBar(lv_obj_t* screen) {
    // One line of chrome along the very bottom. Not a permanent strip any more:
    // ApplyTextLayout sizes it from the font and hides it whenever there is
    // nothing to say, which is most of the time.
    status_bar_ = lv_obj_create(screen);
    lv_obj_set_size(status_bar_, kScreenW, 1);
    lv_obj_set_pos(status_bar_, 0, kScreenH - 1);
    lv_obj_set_style_radius(status_bar_, 0, 0);
    lv_obj_set_style_bg_color(status_bar_, lv_color_hex(kColorBg), 0);
    lv_obj_set_style_bg_opa(status_bar_, LV_OPA_COVER, 0);
    lv_obj_set_style_border_width(status_bar_, 0, 0);
    lv_obj_set_style_pad_all(status_bar_, 0, 0);
    lv_obj_set_style_layout(status_bar_, LV_LAYOUT_NONE, 0);
    lv_obj_remove_flag(status_bar_, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_add_flag(status_bar_, LV_OBJ_FLAG_HIDDEN);

    status_label_ = lv_label_create(status_bar_);
    lv_obj_set_width(status_label_, kTextLineW);
    lv_label_set_long_mode(status_label_, LV_LABEL_LONG_SCROLL_CIRCULAR);
    lv_obj_set_style_text_align(status_label_, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_font(status_label_, ui_font_, 0);
    lv_obj_set_style_text_color(status_label_, lv_color_hex(kColorDim), 0);
    lv_label_set_text(status_label_, Lang::Strings::INITIALIZING);
    lv_obj_align(status_label_, LV_ALIGN_CENTER, 0, 0);

    notification_label_ = lv_label_create(status_bar_);
    lv_obj_set_width(notification_label_, kTextLineW);
    lv_label_set_long_mode(notification_label_, LV_LABEL_LONG_WRAP);
    lv_obj_set_style_text_align(notification_label_, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_font(notification_label_, ui_font_, 0);
    lv_obj_set_style_text_color(notification_label_, lv_color_hex(kColorCoral), 0);
    lv_label_set_text(notification_label_, "");
    lv_obj_align(notification_label_, LV_ALIGN_CENTER, 0, 0);
    lv_obj_add_flag(notification_label_, LV_OBJ_FLAG_HIDDEN);
}

// ---------------------------------------------------------------------------
// Pose / state
// ---------------------------------------------------------------------------

EyeUiState RoboEyesDisplay::ResolveUiState() const {
    switch (Application::GetInstance().GetDeviceState()) {
        case kDeviceStateSpeaking:
            // "Speaking" starts at the transcript, not at the first sentence:
            // for the seconds in between the box is thinking, and must look it.
            return awaiting_reply_ ? EyeUiState::kThinking : EyeUiState::kSpeaking;
        case kDeviceStateListening:
            return thinking_emotion_ ? EyeUiState::kThinking : EyeUiState::kListening;
        case kDeviceStateConnecting:
            return EyeUiState::kThinking;
        case kDeviceStateIdle:
            return thinking_emotion_ ? EyeUiState::kThinking : EyeUiState::kIdle;
        case kDeviceStateStarting:
        case kDeviceStateWifiConfiguring:
        case kDeviceStateActivating:
        case kDeviceStateUpgrading:
            return EyeUiState::kBoot;
        default:
            return EyeUiState::kIdle;
    }
}

void RoboEyesDisplay::ApplyPose(EyeUiState state) {
    // SetStatus()/ShowNotification() can land before SetupUI() has built any of
    // this; nothing to lay out yet.
    if (canvas_ == nullptr) {
        return;
    }
    // Text and eyes never share pixels. ApplyTextLayout parks the band at the
    // bottom and hands the eyes whatever rows are left; the engine re-fits the
    // face to them.
    const bool show_user = !user_text_.empty();
    const bool show_sub = !subtitle_text_.empty() &&
                          (subtitle_is_system_ || state == EyeUiState::kSpeaking);
    ApplyTextLayout(show_user, show_sub);
    ui_state_ = state;
}

void RoboEyesDisplay::UpdateEyes() {
    if (demo_) {
        StepDemo();
        return;
    }
    if (linger_until_ms_ != 0 && static_cast<int32_t>(now_ms_ - linger_until_ms_) >= 0) {
        linger_until_ms_ = 0;
        emotion_ = kNeutral;
        emotion_ms_ = now_ms_;
    }
    if (deterministic_ && now_ms_ - det_ms_ >= kDetMs) {
        // A test that forgets to turn it off must not leave the face frozen.
        deterministic_ = false;
        eyes_.SetDeterministic(false);
        ESP_LOGI(TAG, "eye det: off (after %u s)", (unsigned)(kDetMs / 1000));
    }
    if (deterministic_) {
        // `eye det`: plain eyes, centred, nothing random -- to hold a snapshot
        // against `eyesim still idle idle`.
        eyes_.SetShape(Shape::kIdle);
        eyes_.SetPalette(Palette::kIdle);
        eyes_.SetLids(LidPose());
        eyes_.SetGaze(true, 0, 0);
        eyes_.SetRing(Ring::kOff);
        eyes_.SetVoice(false);
        return;
    }

    const EmotionStyle* e = emotion_;
    uint32_t since = now_ms_ - emotion_ms_;
    bool forced = false;
    int ring_override = -1;
    if (override_ != nullptr || override_ring_ >= 0) {
        if (now_ms_ - override_ms_ < kOverrideMs) {
            if (override_ != nullptr) {
                e = override_;
                since = now_ms_ - override_ms_;
                forced = true;
            }
            ring_override = override_ring_;
        } else {
            override_ = nullptr;
            override_ring_ = -1;
        }
    }

    // What the UI state alone asks for.
    Shape shape = Shape::kIdle;
    Palette palette = Palette::kIdle;
    Ring ring = Ring::kOff;
    LidPose lids;
    bool wander = false;
    float gx = 0, gy = 0;
    switch (ui_state_) {
        case EyeUiState::kBoot:
            ring = Ring::kComet;
            wander = true;
            break;
        case EyeUiState::kIdle:
            wander = true;
            break;
        case EyeUiState::kListening:
            palette = Palette::kListen;
            ring = Ring::kListen;
            gy = -4;    // up at whoever is talking
            break;
        case EyeUiState::kThinking:
            shape = Shape::kThink;
            palette = Palette::kThink;
            gx = -8;    // up and away, mulling it over
            gy = -8;
            break;
        case EyeUiState::kSpeaking:
            palette = Palette::kSpeak;
            ring = Ring::kSpeak;
            gy = 2;
            break;
    }

    // Emotions ride on the idle and speaking faces; listening and thinking keep
    // their own. Errors and console overrides show whatever the state.
    const bool show = forced || e->error ||
                      (ui_state_ != EyeUiState::kListening && ui_state_ != EyeUiState::kThinking);
    if (show) {
        if (e->error && since >= kFlashMs) {
            // The flash said it; until the error clears, droop instead of
            // glaring an X at the room.
            e = FindEmotion("sad");
        }
        if (e->shape != Shape::kIdle) shape = e->shape;
        if (e->palette != Palette::kCount) palette = e->palette;
        lids = e->lids;
        if (e->look) {
            wander = false;
            gx = e->gx;
            gy = e->gy;
        }
        if (e->flash != Ring::kOff && since < kFlashMs) ring = e->flash;
    }
    if (ring_override >= 0) ring = static_cast<Ring>(ring_override);

    eyes_.SetShape(shape);
    eyes_.SetPalette(palette);
    eyes_.SetLids(lids);
    eyes_.SetGaze(wander, gx, gy);
    eyes_.SetRing(ring);
    // The listening ring swells while the mic hears a voice.
    eyes_.SetVoice(ui_state_ == EyeUiState::kListening && Application::GetInstance().IsVoiceDetected());
}

void RoboEyesDisplay::StepDemo() {
    const float t = (now_ms_ - demo_t0_ms_) / 1000.f;
    while (demo_next_ < kDemoCueCount && kDemoCues[demo_next_].t <= t) {
        const Cue& c = kDemoCues[demo_next_++];
        if (c.y1 >= 0) eyes_.SetStrip(kEyeTop, c.y1);
        if (c.shape >= 0) eyes_.SetShape(static_cast<Shape>(c.shape));
        if (c.palette >= 0) eyes_.SetPalette(static_cast<Palette>(c.palette));
        if (c.ring >= 0) eyes_.SetRing(static_cast<Ring>(c.ring));
        if (c.voice >= 0) eyes_.SetVoice(c.voice != 0);
        if (c.set_lids) eyes_.SetLids(c.lids);
        if (c.set_gaze) eyes_.SetGaze(c.wander, c.gx, c.gy);
    }
    if (t >= kDemoSec) {
        demo_ = false;
        eyes_.SetStrip(kEyeTop, kEyeTop + eye_area_h_);
        ESP_LOGI(TAG, "eye demo: done");
    }
}

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------

// A second draw buffer: LVGL fills one while the other goes out over SPI.
// With the port's single buffer each chunk waits for its own transfer.
void RoboEyesDisplay::EnableDoubleBuffer() {
    if (display_ == nullptr || lv_display_is_double_buffered(display_)) return;
    lv_draw_buf_t* b1 = lv_display_get_buf_active(display_);
    if (b1 == nullptr || b1->data == nullptr) return;
    const uint32_t bytes = b1->data_size;
    void* b2 = heap_caps_aligned_alloc(CONFIG_LV_DRAW_BUF_ALIGN, bytes, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (b2 == nullptr) {
        ESP_LOGW(TAG, "no RAM for a second LVGL draw buffer (%u B)", (unsigned)bytes);
        return;
    }
    // Not while a transfer out of the first one is still running.
    for (int i = 0; i < 50 && display_->flushing; i++) vTaskDelay(1);
    lv_display_set_buffers(display_, b1->data, b2, bytes, LV_DISPLAY_RENDER_MODE_PARTIAL);
    ESP_LOGI(TAG, "LVGL draw buffers: 2 x %u B", (unsigned)bytes);
}

void RoboEyesDisplay::EngineTask(void* arg) {
    auto* self = static_cast<RoboEyesDisplay*>(arg);
    for (;;) {
        xSemaphoreTake(self->engine_go_, portMAX_DELAY);
        if (self->engine_quit_) break;
        const int64_t t0 = esp_timer_get_time();
        self->engine_n_ = self->eyes_.Draw(self->engine_rects_, eyes::Engine::kMaxRects);
        const int64_t t1 = esp_timer_get_time();
        self->engine_us_ = static_cast<uint32_t>(t1 - t0);
        self->engine_t0_us_ = t0;
        self->engine_t1_us_ = t1;
        xSemaphoreGive(self->engine_done_);
    }
    xSemaphoreGive(self->engine_done_);
    vTaskDelete(nullptr);
}

// The frame just ticked: to the engine task, which with a canvas of its own
// draws it while LVGL shows the one before; otherwise drawn and up at once.
void RoboEyesDisplay::StartDraw() {
    engine_busy_ = true;
    engine_kick_us_ = esp_timer_get_time();
    if (engine_task_ != nullptr) {
        xSemaphoreGive(engine_go_);
        if (back_buf_ != nullptr) return;
    } else {
        const int64_t t0 = esp_timer_get_time();
        engine_n_ = eyes_.Draw(engine_rects_, eyes::Engine::kMaxRects);
        engine_us_ = static_cast<uint32_t>(esp_timer_get_time() - t0);
    }
    WaitEngine();
}

// Puts up what the engine drew, if it has finished or `wait`. Anything that
// touches the engine outside OnFrame waits for it first, under LVGL's lock.
bool RoboEyesDisplay::CollectEngine(bool wait) {
    if (!engine_busy_) return true;
    if (engine_task_ != nullptr && xSemaphoreTake(engine_done_, wait ? portMAX_DELAY : 0) != pdTRUE) {
        return false;
    }
    engine_busy_ = false;
    const int64_t c0 = esp_timer_get_time();
    for (int i = 0; i < engine_n_; i++) {
        const eyes::Rect& r = engine_rects_[i];
        if (back_buf_ != nullptr) {
            const size_t bytes = static_cast<size_t>(r.x1 - r.x0) * 2;
            for (int y = r.y0; y < r.y1; y++) {
                const size_t o = static_cast<size_t>(y) * canvas_stride_px_ + r.x0;
                memcpy(canvas_buf_ + o, back_buf_ + o, bytes);
            }
        }
        lv_area_t area;
        area.x1 = r.x0;
        area.y1 = r.y0 + kCanvasY;
        area.x2 = r.x1 - 1;
        area.y2 = r.y1 - 1 + kCanvasY;
        lv_obj_invalidate_area(canvas_, &area);
    }
    shown_t_ = eyes_.last_draw_time();
    Stats& s = stats_;
    const uint32_t cu = static_cast<uint32_t>(esp_timer_get_time() - c0);
    s.copy_us += cu;
    s.copy_max = std::max(s.copy_max, cu);
    s.draws++;
    s.draw_us += engine_us_;
    s.draw_us_max = std::max(s.draw_us_max, engine_us_);
    s.evals += eyes_.sdf_evals();
    s.evals_max = std::max(s.evals_max, eyes_.sdf_evals());
    s.px += eyes_.dirty_pixels();
    s.px_max = std::max(s.px_max, eyes_.dirty_pixels());
    if ((s.draws & 63) == 1) {
        s.stack_min = std::min<uint32_t>(s.stack_min, uxTaskGetStackHighWaterMark(nullptr));
        if (engine_task_ != nullptr) {
            s.engine_stack_min = std::min<uint32_t>(s.engine_stack_min, uxTaskGetStackHighWaterMark(engine_task_));
        }
    }
    return true;
}

void RoboEyesDisplay::FrameWakeCb(void*) {
    lvgl_port_task_wake(LVGL_PORT_EVENT_USER, nullptr);
}

void RoboEyesDisplay::FrameTimerCb(lv_timer_t* timer) {
    static_cast<RoboEyesDisplay*>(lv_timer_get_user_data(timer))->OnFrame();
}

void RoboEyesDisplay::OnFrame() {
    if (canvas_buf_ == nullptr) return;
    // A frame starts once it is due; the frame timer wakes LVGL's task for it.
    const int64_t now_us = esp_timer_get_time();
    if (now_us < next_frame_us_ - kFrameEarlyUs) return;
    // Up with what was drawn since the last frame; the engine is idle from
    // here until this frame is handed to it. If it is still drawing, wait:
    // going back to the port's loop costs a 10 ms tick.
    const bool shown = engine_busy_;
    CollectEngine(true);
    if (shown && engine_task_ != nullptr && back_buf_ != nullptr) {
        Stats& s = stats_;
        const uint32_t q = static_cast<uint32_t>(engine_t0_us_ - engine_kick_us_);
        const uint32_t up = static_cast<uint32_t>(esp_timer_get_time() - engine_t1_us_);
        s.splits++;
        s.q_us += q;
        s.q_max = std::max(s.q_max, q);
        s.up_us += up;
        s.up_max = std::max(s.up_max, up);
    }
    const int64_t period_us = frame_period_ms_ * 1000LL;
    next_frame_us_ = std::max(next_frame_us_ + period_us, now_us + period_us / 2);
    if (frame_wake_ != nullptr) {
        esp_timer_stop(frame_wake_);
        esp_timer_start_once(frame_wake_, static_cast<uint64_t>(next_frame_us_ - now_us));
    }
    now_ms_ = NowMs();

    const EyeUiState state = ResolveUiState();
    if (state != ui_state_) {
        ApplyPose(state);
    }
    UpdateEyes();

    stats_.frames++;
    if (!eyes_.Tick(now_ms_)) {
        stats_.moving_t0 = -1;
    } else {
        const int64_t t0 = esp_timer_get_time();
        if (stats_.moving_t0 >= 0) {
            const uint32_t step = static_cast<uint32_t>(t0 - stats_.moving_t0);
            stats_.steps++;
            stats_.step_us += step;
            stats_.step_us_max = std::max(stats_.step_us_max, step);
            if (step > 1000000 / 30) stats_.steps_slow++;
        }
        stats_.moving_t0 = t0;
        StartDraw();
    }
    // What just went up goes out now, while the engine draws the next one,
    // not whenever LVGL's own refresh comes round.
    if (shown) {
        const uint32_t pre = static_cast<uint32_t>(esp_timer_get_time() - now_us);
        stats_.pre_us += pre;
        stats_.pre_max = std::max(stats_.pre_max, pre);
    }
    if (shown && display_ != nullptr) {
        lv_refr_now(display_);
    }
    if (shown) {
        const uint32_t on = static_cast<uint32_t>(esp_timer_get_time() - now_us);
        stats_.on_us += on;
        stats_.on_max = std::max(stats_.on_max, on);
    }

    if (snap_request_ != 0) {
        CaptureSnapshot();
    }
    if (bench_request_) {
        bench_request_ = false;
        WaitEngine();
        RunBenchmark();
    }
}

// `eye bench`: the engine's building blocks timed on a private engine and canvas,
// between two frames of the live one (it shares the live engine's scratch grid).
// Hardware counter pairs for `eye bench`, around one eye's render.
struct PmConfig {
    uint16_t s0, m0, s1, m1;
    const char* n0;
    const char* n1;
};
static const PmConfig kPm[] = {
    {XTPERF_CNT_CYCLES, XTPERF_MASK_CYCLES, XTPERF_CNT_INSN, XTPERF_MASK_INSN_ALL, "cycles", "instructions"},
    {XTPERF_CNT_D_STALL, XTPERF_MASK_D_STALL_ALL, XTPERF_CNT_I_STALL, XTPERF_MASK_I_STALL_ALL, "data stall",
     "insn stall"},
    {XTPERF_CNT_BUBBLES, XTPERF_MASK_BUBBLES_ALL, XTPERF_CNT_EXR, XTPERF_MASK_EXR_ALL, "bubbles", "exceptions"},
    {XTPERF_CNT_BUBBLES, XTPERF_MASK_BUBBLES_R_HOLD_REG_DEP, XTPERF_CNT_BUBBLES, XTPERF_MASK_BUBBLES_CTI,
     "reg-dep hold", "branch bubbles"},
    {XTPERF_CNT_I_STALL, XTPERF_MASK_I_STALL_FAST_L32R, XTPERF_CNT_D_STALL, XTPERF_MASK_D_STALL_CACHE_MISS,
     "l32r stall", "dcache-miss stall"},
    {XTPERF_CNT_I_STALL, XTPERF_MASK_I_STALL_CACHE_MISS, XTPERF_CNT_INSN, XTPERF_MASK_INSN_BRANCH_TAKEN,
     "icache-miss stall", "taken branches"},
    {XTPERF_CNT_BUBBLES, XTPERF_MASK_BUBBLES_R_HOLD_D_CACHE_MISS, XTPERF_CNT_BUBBLES,
     XTPERF_MASK_BUBBLES_R_HOLD_STORE_RELEASE, "dcache-miss hold", "store-release hold"},
    {XTPERF_CNT_D_STALL, XTPERF_MASK_D_STALL_STORE_BUF_FULL | XTPERF_MASK_D_STALL_STORE_BUF_CONFLICT,
     XTPERF_CNT_D_STALL, XTPERF_MASK_D_STALL_BUSY | XTPERF_MASK_D_STALL_BANK_CONFLICT, "store-buffer stall",
     "busy/bank stall"},
};
static constexpr int kPmConfigs = sizeof(kPm) / sizeof(kPm[0]);

static void PmStart(int k) {
    xtensa_perfmon_stop();
    xtensa_perfmon_init(0, kPm[k].s0, kPm[k].m0, 0, 0);
    xtensa_perfmon_init(1, kPm[k].s1, kPm[k].m1, 0, 0);
    xtensa_perfmon_reset(0);
    xtensa_perfmon_reset(1);
    xtensa_perfmon_start();
}

static void PmStop(int, uint32_t out[2]) {
    xtensa_perfmon_stop();
    out[0] = xtensa_perfmon_value(0);
    out[1] = xtensa_perfmon_value(1);
}

void RoboEyesDisplay::RunBenchmark() {
    const uint32_t mhz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
    constexpr int kMaxBlocks = 1024;
    auto* blocks = static_cast<eyes::Engine::BlockProf*>(
        heap_caps_malloc(kMaxBlocks * sizeof(eyes::Engine::BlockProf), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    void* mem = heap_caps_malloc(sizeof(eyes::Engine), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    // Three runs: the full-size canvas in PSRAM as in use, then a face-sized
    // canvas in PSRAM and the same in internal RAM, to price PSRAM access.
    struct Run {
        const char* name;
        int h;
        uint32_t caps;
    } runs[3] = {
        {"240x240 PSRAM", kCanvasH, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT},
        {"240x120 PSRAM", 120, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT},
        {"240x120 internal", 120, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT},
    };
    ESP_LOGI(TAG, "eye bench: internal heap free %u KB, largest block %u KB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024));
    for (int k = 0; k < 3; k++) {
        const size_t bytes = static_cast<size_t>(canvas_stride_px_) * runs[k].h * 2;
        void* canvas = heap_caps_malloc(bytes, runs[k].caps);
        if (canvas == nullptr || eye_work_ == nullptr || mem == nullptr) {
            ESP_LOGW(TAG, "eye bench (%s): out of memory", runs[k].name);
            heap_caps_free(canvas);
            continue;
        }
        eyes::Engine::BenchResult r;
        if (k == 0) {
            r.blocks = blocks;
            r.max_blocks = blocks != nullptr ? kMaxBlocks : 0;
            r.pm_start = PmStart;
            r.pm_stop = PmStop;
            r.pm_configs = kPmConfigs;
            // The block profile counts instructions: counter 0 on INSN.
            r.block_now = []() -> uint32_t { return xtensa_perfmon_value(0); };
            xtensa_perfmon_stop();
            xtensa_perfmon_init(0, XTPERF_CNT_INSN, XTPERF_MASK_INSN_ALL, 0, 0);
            xtensa_perfmon_reset(0);
            xtensa_perfmon_start();
        }
        eyes::Engine::Benchmark([]() -> uint32_t { return esp_cpu_get_cycle_count(); }, canvas, kCanvasW,
                                runs[k].h, canvas_stride_px_, eye_work_, eyes::Engine::WorkFloats(), mem, &r);
        heap_caps_free(canvas);
        ESP_LOGI(TAG, "eye bench, canvas %s. cycles per call: sqrtf %u, fast sqrt %u, divide %u, expf %u, "
                      "idle distance %u",
                 runs[k].name, (unsigned)r.sqrt_lib, (unsigned)r.sqrt_fast, (unsigned)r.div_lib,
                 (unsigned)r.exp_lib, (unsigned)r.dist_idle);
        ESP_LOGI(TAG, "  idle draw: full %u us (clear %u, eyes %u + %u), moving %u us, exact %u us",
                 (unsigned)(r.draw_full / mhz), (unsigned)(r.clear / mhz), (unsigned)(r.eye0 / mhz),
                 (unsigned)(r.eye1 / mhz), (unsigned)(r.draw_move / mhz), (unsigned)(r.draw_exact / mhz));
        ESP_LOGI(TAG, "  full draw: %u evals, %u px cleared, %u flat, %u glow, %u edge", (unsigned)r.evals,
                 (unsigned)r.px_dirty, (unsigned)r.px_flat, (unsigned)r.px_glow, (unsigned)r.px_edge);
        ESP_LOGI(TAG, "  ring alone: listen %u us, comet %u us, celebrate %u us, %u px in its boxes",
                 (unsigned)(r.ring_listen / mhz), (unsigned)(r.ring_comet / mhz),
                 (unsigned)(r.ring_celebrate / mhz), (unsigned)r.ring_px);
        ESP_LOGI(TAG, "  exact draw: %u evals", (unsigned)r.evals_exact);
        for (int c = 0; c < kPmConfigs && k == 0; c++) {
            ESP_LOGI(TAG, "  eye 0 counters: %s %u, %s %u", kPm[c].n0, (unsigned)r.pm[c][0], kPm[c].n1,
                     (unsigned)r.pm[c][1]);
        }
        if (r.nblocks > 0) {
            // Blocks by kind: 1 flat fill, 2 mixed, 3 all glow.
            uint32_t n[4] = {}, c[4] = {};
            for (int i = 0; i < r.nblocks; i++) {
                const int kd = r.blocks[i].kind & 3;
                n[kd]++;
                c[kd] += r.blocks[i].cycles;
            }
            ESP_LOGI(TAG, "  profile (instructions): setup %u + %u; blocks flat %u (%u), mixed %u (%u), glow %u (%u)",
                     (unsigned)r.setup[0], (unsigned)r.setup[1], (unsigned)n[1], (unsigned)c[1], (unsigned)n[2],
                     (unsigned)c[2], (unsigned)n[3], (unsigned)c[3]);
        }
        if (k == 0) {
            // Phase names are the block path's; the open eye's (Engine::OpenEye)
            // are in brackets.
            ESP_LOGI(TAG, "  phases (instructions): centres %u, flat blocks [lighting] %u, nodes [distances] %u, "
                          "glow blocks %u, sort [paint rows] %u, gradients+exact [gather] %u, lit [colour] %u, other %u",
                     (unsigned)r.phase[0], (unsigned)r.phase[1], (unsigned)r.phase[2], (unsigned)r.phase[3],
                     (unsigned)r.phase[4], (unsigned)r.phase[5], (unsigned)r.phase[6], (unsigned)r.phase[7]);
            ESP_LOGI(TAG, "  phases (us): centres %u, flat blocks [lighting] %u, nodes [distances] %u, "
                          "glow blocks %u, sort [paint rows] %u, gradients+exact [gather] %u, lit [colour] %u, other %u",
                     (unsigned)(r.phase_cyc[0] / mhz), (unsigned)(r.phase_cyc[1] / mhz),
                     (unsigned)(r.phase_cyc[2] / mhz), (unsigned)(r.phase_cyc[3] / mhz),
                     (unsigned)(r.phase_cyc[4] / mhz), (unsigned)(r.phase_cyc[5] / mhz),
                     (unsigned)(r.phase_cyc[6] / mhz), (unsigned)(r.phase_cyc[7] / mhz));
        }
    }
    heap_caps_free(blocks);
    heap_caps_free(mem);
}

void RoboEyesDisplay::RefrEventCb(lv_event_t* e) {
    auto* self = static_cast<RoboEyesDisplay*>(lv_event_get_user_data(e));
    Stats& s = self->stats_;
    if (lv_event_get_code(e) == LV_EVENT_RENDER_START) {
        s.render_t0 = esp_timer_get_time();
    } else if (s.render_t0 >= 0) {
        // REFR_READY: rendered and handed to the SPI bus.
        const uint32_t us = static_cast<uint32_t>(esp_timer_get_time() - s.render_t0);
        s.render_t0 = -1;
        s.renders++;
        s.render_us += us;
        s.render_us_max = std::max(s.render_us_max, us);
    }
}

// Integers only: the firmware's printf is newlib-nano (no %f, no 64-bit).
void RoboEyesDisplay::PrintStats() {
    Stats& s = stats_;
    const uint32_t now = NowMs();
    const uint32_t ms = std::max<uint32_t>(1, now - s.since_ms);
    const uint32_t d = std::max<uint32_t>(1, s.draws);
    const uint32_t r = std::max<uint32_t>(1, s.renders);
    ESP_LOGI(TAG, "eye stats over %u ms: %u draws of %u frames = %u.%u draws/s",
             (unsigned)ms, (unsigned)s.draws, (unsigned)s.frames,
             (unsigned)(s.draws * 1000 / ms), (unsigned)(s.draws * 10000 / ms % 10));
    ESP_LOGI(TAG, "  engine draw: avg %u us, max %u us; sdf evals avg %u max %u; px avg %u max %u",
             (unsigned)(s.draw_us / d), (unsigned)s.draw_us_max, (unsigned)(s.evals / d),
             (unsigned)s.evals_max, (unsigned)(s.px / d), (unsigned)s.px_max);
    ESP_LOGI(TAG, "  lvgl render+flush: %u refreshes, avg %u us, max %u us",
             (unsigned)s.renders, (unsigned)(s.render_us / r), (unsigned)s.render_us_max);
    const uint32_t st = std::max<uint32_t>(1, s.steps);
    ESP_LOGI(TAG, "  while moving: %u frame steps, avg %u us (%u fps), max %u us, %u slower than 30 fps",
             (unsigned)s.steps, (unsigned)(s.step_us / st),
             (unsigned)(s.step_us > 0 ? 1000000ull * st / s.step_us : 0), (unsigned)s.step_us_max,
             (unsigned)s.steps_slow);
    const uint32_t sp = std::max<uint32_t>(1, s.splits);
    ESP_LOGI(TAG, "  per frame: engine starts %u us after (max %u), put up %u us after done (max %u); "
             "a frame's OnFrame %u us (max %u)",
             (unsigned)(s.q_us / sp), (unsigned)s.q_max, (unsigned)(s.up_us / sp), (unsigned)s.up_max,
             (unsigned)(s.on_us / sp), (unsigned)s.on_max);
    ESP_LOGI(TAG, "  copy to the shown canvas %u us (max %u); OnFrame before the refresh %u us (max %u)",
             (unsigned)(s.copy_us / std::max<uint32_t>(1, s.draws)), (unsigned)s.copy_max,
             (unsigned)(s.pre_us / sp), (unsigned)s.pre_max);
    ESP_LOGI(TAG, "  stack free min: lvgl task %u B, engine task %u B; grid in %s; heap internal %u KB (min %u), psram %u KB",
             (unsigned)(s.stack_min == UINT32_MAX ? 0 : s.stack_min),
             (unsigned)(s.engine_stack_min == UINT32_MAX ? 0 : s.engine_stack_min),
             eye_work_internal_ ? "internal" : "PSRAM",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    ESP_LOGI(TAG, "  now: ui %d, shape %d, palette %d, ring %d, strip %d..%d, emotion %s",
             (int)ui_state_, (int)eyes_.shape(), (int)eyes_.palette(), (int)eyes_.ring(),
             kEyeTop, kEyeTop + eye_area_h_, emotion_->name);
    const uint32_t stack_min = s.stack_min, engine_stack_min = s.engine_stack_min;
    s = Stats();
    s.since_ms = now;
    s.stack_min = stack_min;
    s.engine_stack_min = engine_stack_min;
}

// ---------------------------------------------------------------------------
// USB debug console
// ---------------------------------------------------------------------------

void RoboEyesDisplay::EyeCommand(const std::string& args) {
    DisplayLockGuard lock(this);
    WaitEngine();
    const uint32_t now = NowMs();
    if (args.empty() || args == "stats") {
        PrintStats();
        return;
    }
    if (args == "off") {
        override_ = nullptr;
        override_ring_ = -1;
        if (demo_) {
            demo_ = false;
            eyes_.SetStrip(kEyeTop, kEyeTop + eye_area_h_);
        }
        deterministic_ = false;
        eyes_.SetDeterministic(false);
        ESP_LOGI(TAG, "eye: back to normal");
        return;
    }
    if (args == "bench") {
        bench_request_ = true;
        return;
    }
    if (args == "demo") {
        demo_ = true;
        demo_t0_ms_ = now;
        demo_next_ = 0;
        ESP_LOGI(TAG, "eye demo: %d s", (int)kDemoSec);
        return;
    }
    if (args == "det") {
        deterministic_ = !deterministic_;
        det_ms_ = now;
        eyes_.SetDeterministic(deterministic_);
        if (deterministic_) {
            ESP_LOGI(TAG, "eye det: on for %u s", (unsigned)(kDetMs / 1000));
        } else {
            ESP_LOGI(TAG, "eye det: off");
        }
        return;
    }
    if (args.rfind("ring ", 0) == 0) {
        const std::string name = args.substr(5);
        for (int i = 0; i < 6; i++) {
            if (name == kRingNames[i]) {
                override_ring_ = i;
                override_ms_ = now;
                ESP_LOGI(TAG, "eye: ring %s for %u s", kRingNames[i], (unsigned)(kOverrideMs / 1000));
                return;
            }
        }
        ESP_LOGW(TAG, "eye: ring off | listen | speak | comet | celebrate | alert");
        return;
    }
    // Shape names work too.
    static const char* const kAliases[][2] = {
        {"idle", "neutral"}, {"love", "loving"}, {"star", "surprised"},
        {"think", "thinking"}, {"wide", "shocked"},
    };
    const char* name = args.c_str();
    for (const auto& a : kAliases) {
        if (args == a[0]) name = a[1];
    }
    const EmotionStyle* style = FindEmotion(name);
    if (style == nullptr) {
        ESP_LOGW(TAG, "eye: unknown '%s' (emotion | ring <mode> | demo | det | off | stats | bench)", args.c_str());
        return;
    }
    override_ = style;
    override_ms_ = now;
    ESP_LOGI(TAG, "eye: %s for %u s", style->name, (unsigned)(kOverrideMs / 1000));
}

void RoboEyesDisplay::RequestSnapshot(bool whole_screen) {
    DisplayLockGuard lock(this);
    snap_request_ = whole_screen ? 2 : 1;
}

bool RoboEyesDisplay::TakeSnapshot(EyeSnapshot* out) {
    DisplayLockGuard lock(this);
    if (snap_.px == nullptr) {
        return false;
    }
    *out = snap_;
    snap_ = EyeSnapshot();
    return true;
}

// Runs on the LVGL task, between frames, so the copy is never half-drawn.
void RoboEyesDisplay::CaptureSnapshot() {
    const bool whole = snap_request_ == 2;
    snap_request_ = 0;
    if (snap_.px != nullptr) {      // an earlier one nobody collected
        heap_caps_free(snap_.px);
        snap_ = EyeSnapshot();
    }
    EyeSnapshot s;
    s.w = kCanvasW;
    s.h = kCanvasH;
    s.whole_screen = whole;
    s.t = shown_t_;
    s.strip_y0 = kEyeTop;
    s.strip_y1 = kEyeTop + eye_area_h_;
    s.px = static_cast<uint16_t*>(
        heap_caps_malloc(static_cast<size_t>(s.w) * s.h * 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (s.px == nullptr) {
        ESP_LOGW(TAG, "snap: out of memory");
        return;
    }
    if (whole) {
        lv_draw_buf_t* buf = lv_snapshot_take(lv_screen_active(), LV_COLOR_FORMAT_RGB565);
        if (buf == nullptr || static_cast<int>(buf->header.w) < s.w || static_cast<int>(buf->header.h) < s.h) {
            ESP_LOGW(TAG, "snap: lv_snapshot_take failed");
            if (buf != nullptr) lv_draw_buf_destroy(buf);
            heap_caps_free(s.px);
            return;
        }
        for (int y = 0; y < s.h; y++) {
            memcpy(s.px + y * s.w, buf->data + y * buf->header.stride, s.w * 2);
        }
        lv_draw_buf_destroy(buf);
    } else {
        for (int y = 0; y < s.h; y++) {
            memcpy(s.px + y * s.w, canvas_buf_ + y * canvas_stride_px_, s.w * 2);
        }
    }
    snap_ = s;
}

// ---------------------------------------------------------------------------
// Display virtuals
// ---------------------------------------------------------------------------

void RoboEyesDisplay::SetEmotion(const char* emotion) {
    if (emotion == nullptr) return;
    DisplayLockGuard lock(this);
    const EmotionStyle* style = FindEmotion(emotion);
    if (style == nullptr) {
        style = kNeutral;   // the boot/OTA icons, or something new from the server
    }
    const uint32_t now = NowMs();
    // The app says "neutral" the moment a turn ends. A feeling stays on the idle
    // face a little longer instead of being wiped mid-smile; thinking and
    // errors have nothing to linger for.
    if (style == kNeutral && emotion_ != kNeutral && !emotion_->error &&
        emotion_->shape != Shape::kThink &&
        Application::GetInstance().GetDeviceState() == kDeviceStateIdle) {
        if (linger_until_ms_ == 0) {
            linger_until_ms_ = now + kLingerMs;
        }
        return;
    }
    linger_until_ms_ = 0;
    // A repeated alert flashes again.
    if (style != emotion_ || style->flash != Ring::kOff) {
        emotion_ms_ = now;
    }
    emotion_ = style;
    thinking_emotion_ = style->shape == Shape::kThink;
    ApplyPose(ResolveUiState());
}

void RoboEyesDisplay::SetChatMessage(const char* role, const char* content) {
    DisplayLockGuard lock(this);
    if (text_zone_ == nullptr) return;

    const bool empty = content == nullptr || *content == '\0';
    // "user" is the speech-to-text echo. It gets its own dim line at the top of
    // the band; dropping it is what made the user's own words invisible.
    if (role != nullptr && strcmp(role, "user") == 0) {
        user_text_ = empty ? std::string() : content;
        lv_label_set_text(user_label_, user_text_.c_str());
        // The box has heard you and handed the turn to the model.
        awaiting_reply_ = !empty;
        RefreshStatus();
        return;
    }
    if (empty) {
        subtitle_text_.clear();
        subtitle_is_system_ = false;
        lv_label_set_text(chat_message_label_, "");
        ApplyPose(ui_state_);
        return;
    }
    // "assistant" is a TTS sentence, shown while speaking. "system" is an alert
    // -- activation code, OTA progress, errors -- readable in any state.
    if (role == nullptr) return;
    if (strcmp(role, "system") == 0) {
        subtitle_is_system_ = true;
    } else if (strcmp(role, "assistant") == 0) {
        subtitle_is_system_ = false;
    } else {
        return;
    }
    subtitle_text_ = content;
    lv_label_set_text(chat_message_label_, subtitle_text_.c_str());
    if (!subtitle_is_system_) {
        awaiting_reply_ = false;   // the reply has started coming out
    }
    RefreshStatus();
}

void RoboEyesDisplay::ClearChatMessages() {
    DisplayLockGuard lock(this);
    subtitle_text_.clear();
    user_text_.clear();
    subtitle_is_system_ = false;
    awaiting_reply_ = false;
    if (chat_message_label_ != nullptr) {
        lv_label_set_text(chat_message_label_, "");
    }
    if (user_label_ != nullptr) {
        lv_label_set_text(user_label_, "");
    }
    ApplyPose(ui_state_);
}

void RoboEyesDisplay::ShowNotification(const char* notification, int duration_ms) {
    LvglDisplay::ShowNotification(notification, duration_ms);
    DisplayLockGuard lock(this);
    ApplyPose(ui_state_);
}

void RoboEyesDisplay::SetStatus(const char* status) {
    if (status == nullptr) return;
    {
        DisplayLockGuard lock(this);
        raw_status_ = status;
        // Connecting / listening / standby all mean a turn just began or ended,
        // so nothing is pending. Without this an errored turn -- one that never
        // produces a sentence -- would sit on "稍等..." forever.
        if (strcmp(status, Lang::Strings::LISTENING) == 0 ||
            strcmp(status, Lang::Strings::CONNECTING) == 0 ||
            strcmp(status, Lang::Strings::STANDBY) == 0) {
            awaiting_reply_ = false;
        }
    }
    RefreshStatus();
}

// The eyes alone turned out not to tell the user anything: pressing the button
// gave several silent seconds that read as "nothing happened". So the strip now
// keeps a word for every running state, and only idle stays wordless.
void RoboEyesDisplay::RefreshStatus() {
    std::string text;
    {
        DisplayLockGuard lock(this);
        if (raw_status_ == Lang::Strings::STANDBY) {
            text.clear();                       // idle: let the eyes have the screen
        } else if (raw_status_ == Lang::Strings::SPEAKING) {
            // Real speech is already on screen as a subtitle; the only thing
            // worth a word here is the silent stretch before it.
            text = awaiting_reply_ ? Lang::Strings::THINKING : "";
        } else if (raw_status_ == Lang::Strings::LISTENING) {
            text = awaiting_reply_ ? Lang::Strings::THINKING : raw_status_;
        } else {
            text = raw_status_;                 // OTA progress, activation, errors
        }
    }
    LvglDisplay::SetStatus(text.c_str());

    DisplayLockGuard lock(this);
    // The strip just appeared or vanished; hand the rows over. (The LVGL port
    // mutex is recursive, so taking it again inside ApplyPose is fine.)
    ApplyPose(ui_state_);
}

void RoboEyesDisplay::UpdateStatusBar(bool update_all) {
    // The base class writes the wall clock into status_label_ when idle; we put
    // it in the top bar instead, so keep its 10 s timer from ever elapsing.
    last_status_update_time_ = std::chrono::system_clock::now();
    LvglDisplay::UpdateStatusBar(update_all);

    DisplayLockGuard lock(this);
    MaybeRefreshFonts();
    if (time_label_ != nullptr) {
        time_t now = time(nullptr);
        struct tm tm_now;
        localtime_r(&now, &tm_now);
        if (tm_now.tm_year >= 2025 - 1900) {
            char buf[8];
            strftime(buf, sizeof(buf), "%H:%M", &tm_now);
            lv_label_set_text(time_label_, buf);
        }
    }
    if (link_dot_ != nullptr) {
        // Derived from the existing device state + network icon: no new channel.
        lv_opa_t opa = LV_OPA_TRANSP;
        switch (Application::GetInstance().GetDeviceState()) {
            case kDeviceStateConnecting:
            case kDeviceStateListening:
            case kDeviceStateSpeaking:
                opa = LV_OPA_COVER;
                break;
            case kDeviceStateIdle:
                opa = network_icon_ != nullptr ? LV_OPA_50 : LV_OPA_TRANSP;
                break;
            default:
                opa = LV_OPA_TRANSP;
                break;
        }
        lv_obj_set_style_bg_opa(link_dot_, opa, 0);
    }
    // The base class's own 3 s timer swaps the notification label back for the
    // status label without telling us. Catch the strip appearing or vanishing
    // here rather than relaying out every second, which would cancel blinks.
    if (StatusStripWanted() != (status_h_ > 0)) {
        ApplyPose(ui_state_);
    }
}

void RoboEyesDisplay::SetTheme(Theme* theme) {
    // This UI ships one hand-picked dark palette; the stock light/dark switch
    // has nothing to act on here. Accepted and ignored on purpose.
    (void)theme;
}

void RoboEyesDisplay::SetPowerSaveMode(bool on) {
    LvglDisplay::SetPowerSaveMode(on);
    DisplayLockGuard lock(this);
    frame_period_ms_ = on ? kPowerSaveFramePeriodMs : kFramePeriodMs;
}

void RoboEyesDisplay::SetScreenOff(bool off) {
    // Holding LVGL's lock means it is between passes, and a pass only ends once
    // its flush is out, so nothing is on the bus while the panel is switched.
    DisplayLockGuard lock(this);
    if (off == screen_off_) return;
    screen_off_ = off;
    if (off) {
        if (frame_timer_ != nullptr) lv_timer_pause(frame_timer_);
        if (frame_wake_ != nullptr) esp_timer_stop(frame_wake_);
        esp_lcd_panel_disp_on_off(panel_, false);
    } else {
        esp_lcd_panel_disp_on_off(panel_, true);
        next_frame_us_ = 0;  // the next pass draws straight away
        if (frame_timer_ != nullptr) lv_timer_resume(frame_timer_);
        lvgl_port_task_wake(LVGL_PORT_EVENT_USER, nullptr);
    }
}
