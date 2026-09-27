# firmware/ — 星智 1.54 ML307 的眼睛固件

这是一份 overlay：只放改过和新增的文件，盖在上游 [78/xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) 的固定提交上（见 `UPSTREAM_COMMIT`），板型 `xingzhi-cube-1.54tft-ml307`。音频、Wi-Fi / 4G 双网、协议、分区表都沿用上游。

*An overlay on upstream xiaozhi-esp32 for the Xingzhi Cube 1.54 ML307 board: SDF eye UI, custom wake word, announcement inbox, 4G fallbacks. Run `./setup.sh`, then build with ESP-IDF 5.5.*

## 编译

```bash
./setup.sh                                  # 克隆上游、切到固定提交、盖上 overlay/
cd xiaozhi-esp32
. ~/esp/esp-idf-v5.5.2/export.sh            # ESP-IDF 5.5
idf.py set-target esp32s3
idf.py menuconfig                           # 见下面「要填的配置」
python scripts/release.py xingzhi-cube-1.54tft-ml307
```

`release.py` 会读板子目录下的 `config.json`，追加唤醒词这些配置，再编译打包。平时改完代码直接 `idf.py build`，然后 `idf.py -p <串口> flash monitor`。

## 要填的配置

`idf.py menuconfig` → **Xiaozhi Assistant**：

| 项 | 说明 |
|---|---|
| `OTA_URL` | 指向你的服务端：`https://<你的域名>/xiaozhi/ota/`。烧进固件，这样长按 BOOT 重新配网把 NVS 里的地址清掉后，也不会掉回官方云 |
| `XZ_SELF_HOST_DOMAIN` | 只填域名，如 `xz.example.com`。OTA 地址是 `https://<这个域名>/` 开头、并且走 4G 时，固件改用明文 HTTP 连，因为 ML307 的 TLS 跟 Cloudflare 握不上手。留空就不改写 |
| `XZ_4G_RELAY` | 可选，`host:port`。4G 下先走这个明文中转（比如国内的 frp），连不上再退回上面的域名 |

也可以写进 `sdkconfig.defaults`（文件末尾有注释好的三行）。

唤醒词在板子目录的 `config.json`：`dang zang mu zi`，显示成「打桩模子」。想换就改那里；拼音按 MultiNet 的写法，多种念法用 `;` 隔开。

## 按键

| 操作 | 作用 |
|---|---|
| 单击 BOOT | 开始 / 结束对话 |
| 长按 BOOT | 屏幕转 90°（记进 NVS） |
| 开机或配网时双击 BOOT | Wi-Fi 和 4G 之间切换 |
| 音量 + / − | 调音量；长按拉满 / 静音 |
| 音量 + 和 − 一起按 | 换下一种声音（服务端轮换，用新声音报一声） |

闲置 5 分钟后关屏，但耳朵不关：播报来了照样会醒。

## USB 串口命令

插在电脑上时，可以往 USB 串口（USB-Serial-JTAG）敲：

| 命令 | 作用 |
|---|---|
| `net wifi` / `net 4g` | 切网络并重启 |
| `rot N` | 屏幕转到 0–3 |
| `reboot` | 重启 |
| `eye …` | 眼睛调试模式；`eye det` 关掉随机动作，两分钟后自动退出 |
| `snap` / `snap lv` | 把当前画面以 base64 打到串口，配合 `scripts/eyesim/snap.py` 取图 |

## 改了哪些文件

| 文件 | 说明 |
|---|---|
| `main/boards/xingzhi-cube-1.54tft-ml307/eye_engine.{h,cc}` | SDF 眼睛引擎。纯 C++，不依赖 ESP-IDF 和 LVGL |
| `main/boards/xingzhi-cube-1.54tft-ml307/robo_eyes_display.{h,cc}` | `SpiLcdDisplay` 子类：眼睛画布、字幕、顶栏状态条、情绪映射，双核出帧 |
| `main/boards/xingzhi-cube-1.54tft-ml307/xingzhi-cube-1.54tft-ml307.cc` | 按键、转屏、关屏、串口命令，换成眼睛 UI |
| `main/boards/xingzhi-cube-1.54tft-ml307/config.{h,json}` | 默认横放（旋转 3）；唤醒词、1000 Hz tick |
| `main/application.{h,cc}` | 播报收件箱长轮询（有消息就自己唤醒）、连接卡死看门狗、4G 连不上时自检一遍并把结论挂在屏幕上 |
| `main/ota.{h,cc}` | 4G 下连自建服务改走明文和中转，失败时换路 |
| `main/protocols/protocol.{h,cc}` | 记录离上次收到服务器数据多久，给看门狗用（4G 弱信号时连接会半路卡死而不报断开） |
| `main/boards/common/board.cc` | OTA 请求里带上次重启原因，服务端日志里就能看到，不用插串口 |
| `main/Kconfig.projbuild`、`sdkconfig.defaults` | 上面那三个配置项 |
| `main/CMakeLists.txt`、`CMakeLists.txt` | 眼睛 UI 不用 emoji 图，腾出空间；可选 30 px 大字幕（`-DROBOEYES_BIG_SUBTITLE=1`）；版本号 |
| `scripts/build_default_assets.py` | 唤醒词可以写几种念法，用 `;` 隔开 |
| `main/assets/locales/*/language.json` | 新增「稍等...」提示 |
| `scripts/eyesim/` | 桌面模拟器：`./build.sh` 用同一份 `eye_engine.cc` 在电脑上出帧，可以跟真机截图逐像素对比 |

## 许可

整个仓库是 MIT，见 [../LICENSE](../LICENSE)。`overlay/` 里改过的上游文件同样是 MIT，上游的版权声明见 `LICENSE-upstream-MIT`。
