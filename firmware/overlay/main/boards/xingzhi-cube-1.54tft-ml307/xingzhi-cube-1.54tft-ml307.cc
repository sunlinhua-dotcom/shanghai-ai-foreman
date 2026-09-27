#include "dual_network_board.h"
#include "codecs/no_audio_codec.h"
#include "robo_eyes_display.h"
#include "system_reset.h"
#include "application.h"
#include "button.h"
#include "config.h"
#include "settings.h"
#include "power_save_timer.h"
#include "led/single_led.h"
#include "assets/lang_config.h"
#include "../xingzhi-cube-1.54tft-wifi/power_manager.h"

#include <esp_log.h>
#include <esp_lcd_panel_vendor.h>

#include <driver/rtc_io.h>
#include <hal/usb_serial_jtag_ll.h>
#include <esp_heap_caps.h>
#include <esp_rom_crc.h>
#include <esp_sleep.h>
#include <esp_timer.h>

#include <algorithm>
#include <mutex>

#define TAG "XINGZHI_CUBE_1_54TFT_ML307"

class XINGZHI_CUBE_1_54TFT_ML307 : public DualNetworkBoard {
private:
    Button boot_button_;
    Button volume_up_button_;
    Button volume_down_button_;
    RoboEyesDisplay* display_;
    PowerSaveTimer* power_save_timer_;
    PowerManager* power_manager_;
    esp_lcd_panel_io_handle_t panel_io_ = nullptr;
    esp_lcd_panel_handle_t panel_ = nullptr;
    int rotation_ = 0;
    std::mutex screen_mutex_;
    bool screen_off_ = false;
    static constexpr int kShutdownBatteryLevel = 10;  // 没插电且电量不高于这个才真关机
    // 音量加减的按住状态和「这一轮是组合键」标记。按键回调都在同一个 esp_timer 任务里跑，不用加锁。
    bool vol_up_held_ = false;
    bool vol_down_held_ = false;
    bool vol_combo_ = false;

    // 音量加减一起按 = 换声音（9/27 下午用户要的，原来在长按 BOOT 上，跟转屏冲突）。
    // 像被唤醒一样连上服务器（listen detect text=__voice__），服务器轮到下一种声音，用新声音报一声就挂回待机。
    void NextVoiceCombo() {
        vol_combo_ = true;
        // 按键回调跑在 esp_timer 任务里（栈 3.5 KB），连网这类活挪到主循环做
        Application::GetInstance().Schedule([this]() {
            power_save_timer_->WakeUp();
            auto& app = Application::GetInstance();
            if (app.GetDeviceState() == kDeviceStateIdle) {
                ESP_LOGI(TAG, "Volume up + down: next voice");
                app.WakeWordInvoke("__voice__");
            }
        });
    }

    // 盒子横放 / 倒放时让 ST7789 自己转（MADCTL 的 MV/MX/MY），不耗 CPU；屏幕是
    // 240×240，布局一个像素都不用动。编号 = 盒子从出厂竖放逆时针转了几个 90°，
    // 画面反着转回来保持正立（Adafruit / TFT_eSPI 的 ST7789 旋转表同一套；
    // 9/23 真机：顺时针横放要 3，照表猜的 1 差了 180°）。
    // 这块玻璃只占控制器 240×320 显存的一角：行序或列序反过来时可见区挪到
    // 80..319，要补 80 的偏移，不补就是一条 80 px 的花屏。
    // 长按 BOOT 再转 90°，存在 NVS display/rotation，断电不丢。
    void ApplyRotation(int rotation) {
        struct Mode { bool swap_xy, mirror_x, mirror_y; int gap_x, gap_y; };
        static const Mode kModes[4] = {
            {false, false, false, 0, 0},    // 0 竖放（出厂）
            {true, true, false, 0, 0},      // 1 逆时针转 90° 横放
            {false, true, true, 0, 80},     // 2 倒放
            {true, false, true, 80, 0},     // 3 顺时针转 90° 横放（用户的摆法）
        };
        const Mode& m = kModes[rotation & 3];
        esp_lcd_panel_swap_xy(panel_, m.swap_xy);
        esp_lcd_panel_mirror(panel_, m.mirror_x, m.mirror_y);
        esp_lcd_panel_set_gap(panel_, m.gap_x, m.gap_y);
    }

    // 显示对象建好以后才能调：lvgl_port_add_disp 建显示时会按构造参数把方向重设一遍
    void SetRotation(int rotation) {
        rotation_ = rotation & 3;
        DisplayLockGuard lock(display_);
        ApplyRotation(rotation_);
        lv_obj_invalidate(lv_screen_active());   // 旧方向画进显存的内容不会自己转，整屏重画
    }

    // 插在电脑上时的遥控：USB 串口敲一行命令（net wifi / net 4g / reboot / rot 0-3；调眼睛用
    // eye <表情> / eye ring <光环> / eye demo / eye det / eye off / eye stats 和 snap [lv]），人不在盒子旁边
    // 也能切网络。USB-Serial-JTAG 只是副控制台（只出不进），这里直接轮询它的接收 FIFO，不装驱动，
    // 不影响日志；打开串口本身会让盒子复位，所以要等它起来再敲。
    static void UsbCommandTask(void* arg) {
        auto* self = static_cast<XINGZHI_CUBE_1_54TFT_ML307*>(arg);
        char line[48];
        size_t n = 0;
        while (true) {
            uint8_t buf[16];
            uint32_t got = usb_serial_jtag_ll_rxfifo_data_available() ? usb_serial_jtag_ll_read_rxfifo(buf, sizeof(buf)) : 0;
            if (got == 0) {
                vTaskDelay(pdMS_TO_TICKS(100));
                continue;
            }
            for (uint32_t i = 0; i < got; i++) {
                char c = static_cast<char>(buf[i]);
                if (c == '\r' || c == '\n') {
                    line[n] = '\0';
                    if (n > 0) {
                        self->HandleUsbCommand(line);
                    }
                    n = 0;
                } else if (n < sizeof(line) - 1) {
                    line[n++] = c;
                }
            }
        }
    }

    void HandleUsbCommand(const std::string& cmd) {
        ESP_LOGW(TAG, "USB command: %s", cmd.c_str());
        if (cmd == "net wifi" || cmd == "net 4g") {
            bool want_wifi = cmd == "net wifi";
            if ((GetNetworkType() == NetworkType::WIFI) == want_wifi) {
                ESP_LOGW(TAG, "USB command: already on %s", want_wifi ? "wifi" : "4g");
                return;
            }
            SwitchNetworkType();    // 存 NVS、屏幕提示、重启
        } else if (cmd == "reboot") {
            Application::GetInstance().Reboot();
        } else if (cmd.rfind("rot ", 0) == 0) {
            int r = atoi(cmd.c_str() + 4);
            Application::GetInstance().Schedule([this, r]() {
                SetRotation(r);
                Settings settings("display", true);
                settings.SetInt("rotation", rotation_);
                ESP_LOGI(TAG, "Screen rotation -> %d", rotation_);
            });
        } else if (cmd == "eye" || cmd.rfind("eye ", 0) == 0) {
            display_->EyeCommand(cmd.size() > 4 ? cmd.substr(4) : std::string());
        } else if (cmd == "snap" || cmd == "snap lv") {
            SendSnapshot(cmd == "snap lv");
        } else {
            ESP_LOGW(TAG, "USB command: unknown (net wifi | net 4g | reboot | rot 0-3 | eye ... | snap [lv])");
        }
    }

    // 直接往 USB-Serial-JTAG 的发送 FIFO 里写：走控制台的话每个字节还要陪 115200 波特的 UART
    // 走一遍，一帧要十几秒。没人在读就放弃。
    static bool UsbWrite(const char* s, size_t n) {
        int64_t last = esp_timer_get_time();
        while (n > 0) {
            if (usb_serial_jtag_ll_txfifo_writable()) {
                const uint32_t k = usb_serial_jtag_ll_write_txfifo(reinterpret_cast<const uint8_t*>(s), n);
                s += k;
                n -= k;
                last = esp_timer_get_time();
            } else {
                usb_serial_jtag_ll_txfifo_flush();
                if (esp_timer_get_time() - last > 500 * 1000) {
                    return false;
                }
            }
        }
        usb_serial_jtag_ll_txfifo_flush();
        return true;
    }

    // snap：眼睛画布（snap lv 是 LVGL 画出来的整屏）按原样 RGB565 小端、base64 分行吐到 USB 串口，
    // scripts/eyesim/snap.py 拼回去再跟 eyesim still 对。吐的时候日志静音，免得插进行里。
    void SendSnapshot(bool whole_screen) {
        display_->RequestSnapshot(whole_screen);
        EyeSnapshot snap;
        bool ok = false;
        for (int i = 0; i < 100 && !ok; i++) {
            vTaskDelay(pdMS_TO_TICKS(20));
            ok = display_->TakeSnapshot(&snap);
        }
        if (!ok) {
            ESP_LOGW(TAG, "snap: no frame");
            return;
        }
        const size_t bytes = static_cast<size_t>(snap.w) * snap.h * 2;
        const uint8_t* data = reinterpret_cast<const uint8_t*>(snap.px);
        const uint32_t crc = esp_rom_crc32_le(0, data, bytes);
        const uint32_t t_s = static_cast<uint32_t>(snap.t);
        const uint32_t t_us = static_cast<uint32_t>((snap.t - t_s) * 1e6);

        // 忙等 FIFO 的时候降到最低优先级，别把 LVGL 和空闲任务饿着。
        const UBaseType_t prio = uxTaskPriorityGet(nullptr);
        vTaskPrioritySet(nullptr, tskIDLE_PRIORITY);
        esp_log_level_set("*", ESP_LOG_NONE);
        vTaskDelay(pdMS_TO_TICKS(100));   // 已经在路上的日志先走完

        static const char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        char line[112];
        int n = snprintf(line, sizeof(line), "\n#SNAP %s %d %d t=%u.%06u strip=%d,%d crc=%08x\n",
                         snap.whole_screen ? "screen" : "canvas", snap.w, snap.h, (unsigned)t_s,
                         (unsigned)t_us, snap.strip_y0, snap.strip_y1, (unsigned)crc);
        bool sent = UsbWrite(line, n);
        unsigned seq = 0;
        for (size_t off = 0; sent && off < bytes; off += 57, seq++) {
            const size_t k = std::min<size_t>(57, bytes - off);
            n = snprintf(line, sizeof(line), "#S%u ", seq);
            for (size_t i = 0; i < k; i += 3) {
                const uint8_t* p = data + off + i;
                const uint32_t v = (p[0] << 16) | ((i + 1 < k ? p[1] : 0) << 8) | (i + 2 < k ? p[2] : 0);
                line[n++] = kB64[(v >> 18) & 63];
                line[n++] = kB64[(v >> 12) & 63];
                line[n++] = i + 1 < k ? kB64[(v >> 6) & 63] : '=';
                line[n++] = i + 2 < k ? kB64[v & 63] : '=';
            }
            line[n++] = '\n';
            sent = UsbWrite(line, n);
        }
        if (sent) {
            n = snprintf(line, sizeof(line), "#SNAP END %u\n", seq);
            UsbWrite(line, n);
        }

        esp_log_level_set("*", static_cast<esp_log_level_t>(CONFIG_LOG_DEFAULT_LEVEL));
        vTaskPrioritySet(nullptr, prio);
        heap_caps_free(snap.px);
        ESP_LOGI(TAG, "snap: %u lines%s", seq, sent ? "" : " (host stopped reading)");
    }

    void InitializePowerManager() {
        // 充电时原来会关掉省电计时器；现在到点只关屏不关机，插不插电都开着，不再跟着充电状态开关
        power_manager_ = new PowerManager(GPIO_NUM_38);
    }

    void InitializePowerSaveTimer() {
        rtc_gpio_init(GPIO_NUM_21);
        rtc_gpio_set_direction(GPIO_NUM_21, RTC_GPIO_MODE_OUTPUT_ONLY);
        rtc_gpio_set_level(GPIO_NUM_21, 1);

        // 闲 60 秒变暗；闲 300 秒原来是断电关机，关着就收不到播报。9/25 用户要「关屏幕，但是不关机」：
        // 现在只关屏，4G 和唤醒词照开，来播报、喊醒、按键都会亮屏。没插电、电量低才真关机，护电池。
        power_save_timer_ = new PowerSaveTimer(-1, 60, 300);
        power_save_timer_->OnEnterSleepMode([this]() {
            GetDisplay()->SetPowerSaveMode(true);
            GetBacklight()->SetBrightness(1);
        });
        power_save_timer_->OnExitSleepMode([this]() {
            SetScreenOff(false);
            GetDisplay()->SetPowerSaveMode(false);
            GetBacklight()->RestoreBrightness();
        });
        power_save_timer_->OnShutdownRequest([this]() {
            if (power_manager_->IsDischarging() &&
                power_manager_->GetBatteryLevel() <= kShutdownBatteryLevel) {
                ESP_LOGI(TAG, "Shutting down");
                rtc_gpio_set_level(GPIO_NUM_21, 0);
                // 启用保持功能，确保睡眠期间电平不变
                rtc_gpio_hold_en(GPIO_NUM_21);
                esp_lcd_panel_disp_on_off(panel_, false); //关闭显示
                esp_deep_sleep_start();
            }
            SetScreenOff(true);  // 过了 300 秒以后每秒都来一次，SetScreenOff 自己去重
        });
        power_save_timer_->SetEnabled(true);
    }

    // 关屏 = 背光灭 + 面板关 + 眼睛不出帧。计时器（每秒）和唤醒（按键、播报、唤醒词）
    // 在不同任务里，用锁排队；关之前再看一眼是不是刚好有人把它叫醒了。
    void SetScreenOff(bool off) {
        std::lock_guard<std::mutex> lock(screen_mutex_);
        if (off == screen_off_) return;
        if (off && !Application::GetInstance().CanEnterSleepMode()) return;
        screen_off_ = off;
        ESP_LOGI(TAG, "Screen %s", off ? "off" : "on");
        if (off) {
            GetBacklight()->SetBrightness(0);
        }
        display_->SetScreenOff(off);
    }

    void InitializeSpi() {
        spi_bus_config_t buscfg = {};
        buscfg.mosi_io_num = DISPLAY_SDA;
        buscfg.miso_io_num = GPIO_NUM_NC;
        buscfg.sclk_io_num = DISPLAY_SCL;
        buscfg.quadwp_io_num = GPIO_NUM_NC;
        buscfg.quadhd_io_num = GPIO_NUM_NC;
        buscfg.max_transfer_sz = DISPLAY_WIDTH * DISPLAY_HEIGHT * sizeof(uint16_t);
        ESP_ERROR_CHECK(spi_bus_initialize(SPI3_HOST, &buscfg, SPI_DMA_CH_AUTO));
    }

    void InitializeButtons() {
        boot_button_.OnClick([this]() {
            power_save_timer_->WakeUp();
            auto& app = Application::GetInstance();
            if (GetNetworkType() == NetworkType::WIFI) {
                if (app.GetDeviceState() == kDeviceStateStarting) {
                    // cast to WifiBoard
                    auto& wifi_board = static_cast<WifiBoard&>(GetCurrentBoard());
                    wifi_board.EnterWifiConfigMode();
                    return;
                }
            }
            app.ToggleChatState();
        });
        boot_button_.OnLongPress([this]() {
            // 长按 BOOT = 转屏 90°。9/27 凌晨一度改成换声音，下午用户说跟他的转屏冲突，改回来；换声音挪到音量加减一起按。
            // 按键回调跑在 esp_timer 任务里（栈 3.5 KB），锁 LVGL、发 SPI、写 NVS 这类活挪到主循环做
            Application::GetInstance().Schedule([this]() {
                power_save_timer_->WakeUp();
                SetRotation(rotation_ + 1);
                Settings settings("display", true);
                settings.SetInt("rotation", rotation_);
                ESP_LOGI(TAG, "Screen rotation -> %d", rotation_);
            });
        });
        boot_button_.OnDoubleClick([this]() {
            auto& app = Application::GetInstance();
            if (app.GetDeviceState() == kDeviceStateStarting || app.GetDeviceState() == kDeviceStateWifiConfiguring) {
                SwitchNetworkType();
            }
        });

        // 音量加减一起按 = 换声音：谁后按下、另一个还按着，就算组合键。
        // 组合键这一轮两个键的单击 / 长按都作废（单击要等松手后才触发），不然音量会先加后减、或者被拉满 / 静音；
        // 下一次单独按下时再清掉标记。
        volume_up_button_.OnPressDown([this]() {
            if (!vol_down_held_) vol_combo_ = false;
            vol_up_held_ = true;
            if (vol_down_held_) NextVoiceCombo();
        });
        volume_up_button_.OnPressUp([this]() { vol_up_held_ = false; });
        volume_down_button_.OnPressDown([this]() {
            if (!vol_up_held_) vol_combo_ = false;
            vol_down_held_ = true;
            if (vol_up_held_) NextVoiceCombo();
        });
        volume_down_button_.OnPressUp([this]() { vol_down_held_ = false; });

        volume_up_button_.OnClick([this]() {
            if (vol_combo_) return;
            power_save_timer_->WakeUp();
            auto codec = GetAudioCodec();
            auto volume = codec->output_volume() + 10;
            if (volume > 100) {
                volume = 100;
            }
            codec->SetOutputVolume(volume);
            GetDisplay()->ShowNotification(Lang::Strings::VOLUME + std::to_string(volume));
        });

        volume_up_button_.OnLongPress([this]() {
            if (vol_combo_) return;
            power_save_timer_->WakeUp();
            GetAudioCodec()->SetOutputVolume(100);
            GetDisplay()->ShowNotification(Lang::Strings::MAX_VOLUME);
        });

        volume_down_button_.OnClick([this]() {
            if (vol_combo_) return;
            power_save_timer_->WakeUp();
            auto codec = GetAudioCodec();
            auto volume = codec->output_volume() - 10;
            if (volume < 0) {
                volume = 0;
            }
            codec->SetOutputVolume(volume);
            GetDisplay()->ShowNotification(Lang::Strings::VOLUME + std::to_string(volume));
        });

        volume_down_button_.OnLongPress([this]() {
            if (vol_combo_) return;
            power_save_timer_->WakeUp();
            GetAudioCodec()->SetOutputVolume(0);
            GetDisplay()->ShowNotification(Lang::Strings::MUTED);
        });
    }

    void InitializeSt7789Display() {
        ESP_LOGD(TAG, "Install panel IO");
        esp_lcd_panel_io_spi_config_t io_config = {};
        io_config.cs_gpio_num = DISPLAY_CS;
        io_config.dc_gpio_num = DISPLAY_DC;
        io_config.spi_mode = 3;
        io_config.pclk_hz = 80 * 1000 * 1000;
        io_config.trans_queue_depth = 10;
        io_config.lcd_cmd_bits = 8;
        io_config.lcd_param_bits = 8;
        ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi(SPI3_HOST, &io_config, &panel_io_));

        ESP_LOGD(TAG, "Install LCD driver");
        esp_lcd_panel_dev_config_t panel_config = {};
        panel_config.reset_gpio_num = DISPLAY_RES;
        panel_config.rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB;
        panel_config.bits_per_pixel = 16;
        ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(panel_io_, &panel_config, &panel_));
        ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_));
        ESP_ERROR_CHECK(esp_lcd_panel_init(panel_));
        ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel_, DISPLAY_SWAP_XY));
        ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel_, DISPLAY_MIRROR_X, DISPLAY_MIRROR_Y));
        ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel_, true));

        display_ = new RoboEyesDisplay(panel_io_, panel_, DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_OFFSET_X, DISPLAY_OFFSET_Y,
            DISPLAY_MIRROR_X, DISPLAY_MIRROR_Y, DISPLAY_SWAP_XY);

        Settings settings("display", false);
        SetRotation(settings.GetInt("rotation", DISPLAY_DEFAULT_ROTATION));
        ESP_LOGI(TAG, "Screen rotation %d", rotation_);
    }

public:
    XINGZHI_CUBE_1_54TFT_ML307() :
        DualNetworkBoard(ML307_TX_PIN, ML307_RX_PIN),
        boot_button_(BOOT_BUTTON_GPIO),
        volume_up_button_(VOLUME_UP_BUTTON_GPIO),
        volume_down_button_(VOLUME_DOWN_BUTTON_GPIO) {
        InitializePowerManager();
        InitializePowerSaveTimer();
        InitializeSpi();
        InitializeButtons();
        InitializeSt7789Display();
        GetBacklight()->RestoreBrightness();
        xTaskCreate(UsbCommandTask, "usb_cmd", 4096, this, 2, nullptr);
    }

    virtual AudioCodec* GetAudioCodec() override {
        static NoAudioCodecSimplex audio_codec(AUDIO_INPUT_SAMPLE_RATE, AUDIO_OUTPUT_SAMPLE_RATE,
            AUDIO_I2S_SPK_GPIO_BCLK, AUDIO_I2S_SPK_GPIO_LRCK, AUDIO_I2S_SPK_GPIO_DOUT, AUDIO_I2S_MIC_GPIO_SCK, AUDIO_I2S_MIC_GPIO_WS, AUDIO_I2S_MIC_GPIO_DIN);
        return &audio_codec;
    }

    virtual Display* GetDisplay() override {
        return display_;
    }
    
    virtual Backlight* GetBacklight() override {
        static PwmBacklight backlight(DISPLAY_BACKLIGHT_PIN, DISPLAY_BACKLIGHT_OUTPUT_INVERT);
        return &backlight;
    }

    virtual bool GetBatteryLevel(int& level, bool& charging, bool& discharging) override {
        // 省电计时器插不插电都开着（到点只关屏不关机），不再跟着充放电开关它
        charging = power_manager_->IsCharging();
        discharging = power_manager_->IsDischarging();
        level = power_manager_->GetBatteryLevel();
        return true;
    }

    virtual void SetPowerSaveLevel(PowerSaveLevel level) override {
        if (level != PowerSaveLevel::LOW_POWER) {
            power_save_timer_->WakeUp();
        }
        DualNetworkBoard::SetPowerSaveLevel(level);
    }
};

DECLARE_BOARD(XINGZHI_CUBE_1_54TFT_ML307);
