// Eyes display for the 无名科技星智 1.54" (ML307) cube.
//
// Replaces the stock chat/emoji UI with a pair of glass eyes drawn from signed
// distance fields (eye_engine.h), plus a thin top bar and a bottom status strip.
// Everything below the UI -- audio, ML307/Wi-Fi, protocol, OTA -- is untouched;
// this only overrides the Display virtuals.

#ifndef ROBO_EYES_DISPLAY_H
#define ROBO_EYES_DISPLAY_H

#include "display/lcd_display.h"
#include "display/lvgl_display/lvgl_font.h"
#include "eye_engine.h"

#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#include <memory>
#include <string>

// Which pose the eyes hold. Derived from the app's DeviceState, plus the
// "thinking" emotion which has no device state of its own.
enum class EyeUiState {
    kBoot,
    kIdle,
    kListening,
    kThinking,
    kSpeaking,
};

// How one server-side `emotion` value (or one of the app's alert icons) shows.
struct EmotionStyle {
    const char* name;
    eyes::Shape shape;       // kIdle keeps the glass capsule, bent by the lids
    eyes::Palette palette;   // kCount keeps the UI state's colour
    eyes::LidPose lids;      // idle shape only, design px
    bool look;               // hold the gaze at (gx, gy) instead of the state's
    float gx, gy;            // design px, + is right / down
    eyes::Ring flash;        // ring shown for the first seconds, kOff for none
    bool error;              // after the flash, droop instead of keeping the shape
};

// One frame copied off the screen for the USB `snap` command. Native RGB565,
// rows packed; free `px` with heap_caps_free().
struct EyeSnapshot {
    uint16_t* px = nullptr;
    int w = 0, h = 0;
    bool whole_screen = false;   // false: the eye canvas alone
    double t = 0;                // engine time of the frame on the canvas
    int strip_y0 = 0, strip_y1 = 0;
};

class RoboEyesDisplay : public SpiLcdDisplay {
public:
    RoboEyesDisplay(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_handle_t panel,
                    int width, int height, int offset_x, int offset_y,
                    bool mirror_x, bool mirror_y, bool swap_xy);
    ~RoboEyesDisplay();

    void SetupUI() override;
    void SetEmotion(const char* emotion) override;
    void SetChatMessage(const char* role, const char* content) override;
    void ClearChatMessages() override;
    void SetStatus(const char* status) override;
    void ShowNotification(const char* notification, int duration_ms = 3000) override;
    using LvglDisplay::ShowNotification;   // keep the std::string overload visible
    void UpdateStatusBar(bool update_all = false) override;
    void SetTheme(Theme* theme) override;
    void SetPowerSaveMode(bool on) override;
    // Idle screen-off (the board decides when): panel off and no eye frames at
    // all; `false` turns the panel back on and restarts them.
    void SetScreenOff(bool off);

    // USB debug console (see the board file). `args` is what followed "eye ":
    // an emotion name, "ring <mode>", "demo", "det", "off", "stats" or "bench".
    void EyeCommand(const std::string& args);
    // The copy is made by the LVGL task at its next frame; poll TakeSnapshot().
    void RequestSnapshot(bool whole_screen);
    bool TakeSnapshot(EyeSnapshot* out);

private:
    static void FrameTimerCb(lv_timer_t* timer);
    static void RefrEventCb(lv_event_t* e);
    void OnFrame();
    void EnableDoubleBuffer();

    void ResolveFonts();
    void MaybeRefreshFonts();
    void ApplyFontsToWidgets();
    void BuildTopBar(lv_obj_t* screen, const lv_font_t* icon_font);
    void BuildEyeCanvas(lv_obj_t* screen);
    void BuildBottomBar(lv_obj_t* screen);
    void BuildTextZone(lv_obj_t* screen);

    EyeUiState ResolveUiState() const;
    // Lays the text out for `state` and remembers it.
    void ApplyPose(EyeUiState state);
    // Turns UI state + emotion + debug overrides into engine inputs. Every frame.
    void UpdateEyes();
    void StepDemo();
    void CaptureSnapshot();
    void RunBenchmark();
    void PrintStats();
    // Sizes the bottom stack from the text actually in it and hands every row
    // above it to the eyes.
    void ApplyTextLayout(bool show_user, bool show_subtitle);
    // Hands the eyes a shorter (or the full) strip to live in.
    void SetEyeArea(int height);
    // How tall `text` renders at the band width. Measurement only, no layout.
    int MeasureTextH(const char* text, const lv_font_t* font, int line_space) const;
    // Is there chrome text worth giving the bottom strip a row for?
    bool StatusStripWanted() const;
    // Re-renders the bottom strip from raw_status_ + awaiting_reply_.
    void RefreshStatus();

    const EmotionStyle* emotion_ = nullptr;
    uint32_t emotion_ms_ = 0;         // when it arrived, for the flash
    // The app says "neutral" the moment a turn ends; the last emotion stays on
    // the idle face until this time instead. 0 = not lingering.
    uint32_t linger_until_ms_ = 0;
    EyeUiState ui_state_ = EyeUiState::kBoot;
    bool thinking_emotion_ = false;
    // The protocol flips to "speaking" the moment the transcript comes back,
    // seconds before the model has said anything. True from the STT echo until
    // the first sentence arrives -- i.e. while the box is actually thinking.
    bool awaiting_reply_ = false;
    // Last status string the app handed us, before our own re-wording.
    std::string raw_status_;

    eyes::Engine eyes_;
    float* eye_work_ = nullptr;       // the engine's distance grid
    bool eye_work_internal_ = false;
    uint16_t* canvas_buf_ = nullptr;
    int canvas_stride_px_ = 0;
    lv_obj_t* canvas_ = nullptr;
    lv_timer_t* frame_timer_ = nullptr;

    // The engine draws on core 0, into a canvas of its own, while LVGL puts
    // the frame before on screen from canvas_buf_. LVGL's task is on core 1,
    // as is the wake word's audio work, which takes 10 ms turns with it.
    // Each frame starts by copying what the engine drew onto canvas_buf_.
    static void EngineTask(void* arg);
    void StartDraw();
    bool CollectEngine(bool wait);
    void WaitEngine() { CollectEngine(true); }
    TaskHandle_t engine_task_ = nullptr;
    SemaphoreHandle_t engine_go_ = nullptr;
    SemaphoreHandle_t engine_done_ = nullptr;
    uint16_t* back_buf_ = nullptr;   // the engine's canvas; null: it draws on canvas_buf_
    bool engine_quit_ = false;
    bool engine_busy_ = false;       // a frame drawn, or being drawn, and not up yet
    int engine_n_ = 0;
    uint32_t engine_us_ = 0;
    int64_t engine_kick_us_ = 0, engine_t0_us_ = 0, engine_t1_us_ = 0;
    eyes::Rect engine_rects_[eyes::Engine::kMaxRects];
    double shown_t_ = 0;             // the time of what canvas_buf_ shows

    // Frames are timed to the microsecond: an esp_timer wakes LVGL's task
    // when the next is due, rather than the 10 ms FreeRTOS tick.
    static void FrameWakeCb(void* arg);
    esp_timer_handle_t frame_wake_ = nullptr;
    int frame_period_ms_ = 0;
    int64_t next_frame_us_ = 0;
    bool screen_off_ = false;
    uint32_t now_ms_ = 0;

    // Debug overrides from the USB console.
    const EmotionStyle* override_ = nullptr;
    int override_ring_ = -1;          // eyes::Ring, -1 = none
    uint32_t override_ms_ = 0;         // when it was set; lasts kOverrideMs
    bool demo_ = false;
    uint32_t demo_t0_ms_ = 0;
    int demo_next_ = 0;
    bool deterministic_ = false;
    uint32_t det_ms_ = 0;              // when `eye det` went on; lasts kDetMs
    int snap_request_ = 0;            // 1 canvas, 2 whole screen
    bool bench_request_ = false;
    EyeSnapshot snap_;

    struct Stats {
        uint32_t since_ms = 0;
        uint32_t frames = 0, draws = 0;
        uint64_t draw_us = 0, evals = 0, px = 0;
        uint32_t draw_us_max = 0, evals_max = 0, px_max = 0;
        uint32_t renders = 0;
        uint64_t render_us = 0;
        uint32_t render_us_max = 0;
        int64_t render_t0 = -1;
        uint32_t stack_min = UINT32_MAX;
        uint32_t engine_stack_min = UINT32_MAX;
        // Frame to frame while the eyes keep moving: from one draw's start to
        // the next's, when no frame in between was still.
        int64_t moving_t0 = -1;
        uint32_t steps = 0, steps_slow = 0, step_us_max = 0;
        uint64_t step_us = 0;
        // Where a moving frame's time goes: handed over -> engine starts ->
        // engine done -> put up by the next frame; and OnFrame's own time.
        uint64_t q_us = 0, up_us = 0, on_us = 0, copy_us = 0, pre_us = 0;
        uint32_t q_max = 0, up_max = 0, on_max = 0, splits = 0, copy_max = 0, pre_max = 0;
    } stats_;

    lv_obj_t* time_label_ = nullptr;
    lv_obj_t* link_dot_ = nullptr;

    // Opaque band along the bottom of the eye area. Text never shares pixels
    // with the eyes: when the band is up, the eyes get the strip above it.
    lv_obj_t* text_zone_ = nullptr;
    lv_obj_t* user_label_ = nullptr;   // one dim line: what the mic heard

    // Chrome runs at ~20 px, the subtitle at whatever the assets partition can
    // give us (20 px stock, 30 px with the optional assets image). See NOTES.md.
    const lv_font_t* ui_font_ = nullptr;
    const lv_font_t* subtitle_font_ = nullptr;
    // The echo line is whatever the user just said, i.e. arbitrary Chinese, so
    // it must never fall back to the 206-CJK built-in font -- that is what made
    // the user's own words come out blank.
    const lv_font_t* echo_font_ = nullptr;
    // assets.bin is applied well after SetupUI(), so the theme font we see at
    // build-the-UI time is still the small built-in one. Watch for the swap.
    const lv_font_t* observed_theme_font_ = nullptr;
    std::shared_ptr<LvglFont> extra_ui_font_;

    bool text_zone_shown_ = false;
    int eye_area_h_ = 0;             // height of the strip the eyes live in
    int band_h_ = 0;                 // current text band height, 0 = collapsed
    int status_h_ = 0;               // current status strip height, 0 = collapsed
    // System messages (activation code, OTA progress, alerts) also use the
    // subtitle, but unlike TTS text they show in every state.
    bool subtitle_is_system_ = false;
    std::string subtitle_text_;
    std::string user_text_;
};

#endif  // ROBO_EYES_DISPLAY_H
