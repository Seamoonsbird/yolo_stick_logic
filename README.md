> 本仓库为智能盲杖的决策程序


# 文件夹结构

```
yolo_stick_logic/
├── main.py                 # 程序入口
├── config.py               # 全局配置
├── models/                 # YOLO 模型权重
│   ├── best.pt
│   └── origin.pt
├── nodes/                  # 四个工作进程
│   ├── camera.py
│   ├── inference.py
│   ├── logic.py
│   ├── recorder.py
│   └── video_audio.py      # 提示音混入录像
├── test/                   # 测试 / 调试脚本
├── sh/                     # Docker 启动脚本
└── out/                    # 录像输出（运行时生成，已忽略）
```

# 各文件作用

## 根目录

| 文件 | 作用 |
| --- | --- |
| `main.py` | 程序入口。创建帧/结果/录制/状态四个队列和停止事件，启动摄像头、推理、逻辑、录制四个进程并把它们串起来；`Ctrl+C` 后统一退出清理。 |
| `config.py` | 唯一的配置文件。摄像头参数、模型路径、置信度/IoU 阈值、队列长度、是否录制、输出目录等都在这里改；`get_output_dir()` 按日期生成并创建 `out/日期/` 目录。 |

## nodes/ — 数据处理链路的四个进程

数据流：`摄像头 → frame_q → 推理 → result_q / recorder_q → 逻辑 → state_q → 录制`

| 文件 | 作用 |
| --- | --- |
| `camera.py` | 摄像头采集。用 `gst-launch-1.0` 子进程读 Jetson CSI 摄像头，把 BGR 裸流还原成 numpy 帧，接口模仿 `cv2.VideoCapture`；按 `config.ROTATE_CCW_90` 在帧入队前做逆时针 90° 旋转，`camera_worker` 不断把最新帧塞进 `frame_q`（满了丢旧帧）。 |
| `inference.py` | YOLO 推理。加载模型并预热、读取模型自带类别名；每帧推理后用 `results.plot()` 画框，检测数据（框/类别/置信度）送 `result_q`，标注后的画面送 `recorder_q`。 |
| `logic.py` | 核心决策（盲杖业务逻辑）。按有效画面尺寸（默认逆时针旋转后 480×640）划分「正前方 / 侧方 / 近距离 / 危险距离」区域，分别处理盲道居中、上下楼梯、水坑井盖、草地/斑马线/木板；带冷却时间去重、TTS 语音播报，并把状态推给 `state_q`；每次真实发声还会把 `{时间戳, WAV, 文案}` 发给录制进程（`audio_q`）。 |
| `recorder.py` | 录制。收标注帧，用 PIL 叠加中文状态、最近告警、时间戳和实测 FPS（解决 OpenCV 中文乱码），按前两帧间隔自动测算真实帧率，写入 `out/日期/record_时间戳.mp4`；结束前调用 `video_audio` 把提示音混成音轨。 |
| `video_audio.py` | 提示音混入录像。读 WAV（8/16/24/32-bit，多声道取平均）→ 重采样 → 按事件偏移叠加成与视频等长的音轨 → ffmpeg 复制视频流 + AAC 编码混流。 |

## models/ — 模型权重

| 文件 | 作用 |
| --- | --- |
| `best.pt` | 当前使用的训练权重，`config.MODEL_PATH` 默认指向它。 |
| `origin.pt` | 原始的/备用模型文件，一般用于对比或重新导出。 |

## test/ — 测试与调试脚本（不参与主流程）

| 文件 | 作用 |
| --- | --- |
| `csi_camera_utils.py` | 独立版的 CSI 摄像头打开工具（带超详细注释），供下面的分割 demo 调用，思路和 `nodes/camera.py` 相同。 |
| `02.segmentation_camera_csi_commented.py` | 实时实例分割演示：加载分割模型 + CSI 摄像头，逐帧推理并显示窗口，按 `q` 退出。 |
| `test.py` | 视频抽帧工具：读取 `output/orange.mkv`，每 15 帧存一张 png，用来制作数据集。 |
| `logic_test.py` | 从另一个项目（抓球小车）拷来的状态机逻辑，依赖的 `GRAB_ZONE_*` 等配置本仓库 `config.py` 里没有，仅作思路参考，不能直接运行。 |
| `led.py` | 树莓派 LED 闪烁测试（`RPi.GPIO`，BOARD 12 脚）。 |
| `serial_test.py` | USB 转 TTL 串口测试工具：列串口、TX-RX 回环自测、按 `AA 02 cmd 55` 帧格式发命令并看回包。需要 `pip install pyserial`。 |

## sh/ — Docker 启动脚本

| 文件 | 作用 |
| --- | --- |
| `run_yolo_docker.sh` | **主启动脚本**。创建 `yolo26_csi` 容器：挂载本项目到 `/ultralytics/yolo_stick_logic`、通摄像头和声卡、开 X11 显示、启用 GPU。 |
| `start_yolo_csi.sh` | 重新进入已存在的 `yolo26_csi` 容器（运行中用 `exec`，已停止用 `start -ai`）。 |
| `yolov11_docker.sh` | 旧版容器启动脚本，挂载 YoloSmartStick 项目。 |
| `run_usb_docker.sh` | 启动 ROS2 基础容器，挂载 USB 摄像头，用于 yahboomcar 开发。 |

## 其他

| 文件 | 作用 |
| --- | --- |
| `.vscode/settings.json` | VS Code 的 Python 环境设置（conda）。 |
| `.gitignore` | 忽略录像、`out/`、`__pycache__`、`.vscode`、`.claude`。 |
| `.gitattributes` | 统一使用 LF 换行（在 Linux/Docker 中运行）。 |

# 运行

```bash
# 在 run_yolo_docker.sh 启动的容器内
python main.py
```

录像保存在 `out/日期/`，按 `Ctrl+C` 停止。

# 画面方向（逆时针旋转 90°）

`config.py` 里：

```python
ROTATE_CCW_90 = True    # 画面逆时针旋转 90°；摄像头方向正立了就改成 False
```

旋转在 `nodes/camera.py` 的 `camera_worker` 里、帧进队列**之前**完成，所以推理框、逻辑判断、录制视频用的都是同一套「已正立」坐标。
旋转后有效画面尺寸变成 `480×640`（`config.FRAME_WIDTH / FRAME_HEIGHT` 会自动互换），`logic.py` 的中线、正前方区域、录制分辨率全部跟着变，不需要另外改。

# 中文字体（Docker 容器内）

录制画面上的中文由 PIL 渲染，容器里必须有中文字体，否则显示成方块。**容器内执行：**

```bash
# 1. 进容器
docker exec -it yolo26_csi_speak bash

# 2. 装字体（Debian/Ubuntu 系，Jetson 镜像通用）
apt-get update
apt-get install -y fonts-wqy-zenhei
fc-cache -fv

# 3. 验证：能列出中文字体即成功
fc-list :lang=zh | head
exit
```

一条命令版（在宿主机执行，不用进容器）：

```bash
docker exec -it yolo26_csi_speak bash -c "apt-get update && apt-get install -y fonts-wqy-zenhei && fc-cache -fv && fc-list :lang=zh"
```

容器没网或 apt 太慢时，从宿主机拷一个中文字体进去：

```bash
# Linux 宿主机
docker cp /path/to/wqy-zenhei.ttc yolo26_csi_speak:/usr/share/fonts/truetype/
# Windows 宿主机（字体在 C:\Windows\Fonts）
docker cp C:\Windows\Fonts\msyh.ttc yolo26_csi_speak:/usr/share/fonts/truetype/
# 随后刷新字体缓存
docker exec yolo26_csi_speak fc-cache -fv
```

启动时 `[Recorder] 中文字体: /usr/share/fonts/...` 打印出真实路径即生效；找不到时 `recorder.py` 会自动扫描 `/usr/share/fonts` 下的 `wqy / noto cjk / msyh / simhei / droid fallback` 等文件，所以拷进任意子目录也能被识别。

> 容器一旦删除重建，apt 装的字体就没了。想永久保留，可以在 `sh/run_yolo_docker_speak.sh` 的 `docker run` 里加一行，把宿主机的字体目录只读挂进去：
>
> ```bash
> -v /usr/share/fonts/truetype/wqy:/usr/share/fonts/truetype/wqy:ro
> ```

# 提示音混入录像

录制结束后，`nodes/video_audio.py` 会把本次**实际播放过**的语音提示按时间轴铺成一条音轨，再用 ffmpeg 和视频合成——输出仍是 `out/日期/record_时间戳.mp4`，播放时能听到提示音。

`config.py` 开关：

```python
MUX_ALERT_AUDIO = True    # False → 只录画面，不生成音轨
AUDIO_MIX_RATE = 22050    # 混音采样率，语音够用；想更通用改 44100
```

原理：`logic.speak()` 每次真实发声时，把 `{时间戳, WAV 路径, 文案}` 塞进 `audio_q`（冷却期内的提醒不会发，因为它本来就没播）；录制进程以**视频首帧时刻为 0 点**算出每条提示的偏移，叠加成与视频等长的音轨，最后
`ffmpeg -c:v copy -c:a aac` 混流（视频流直接复制，不重新编码，画质无损）。

不录制系统声卡，所以**音画不会漂移**，也不用给容器 PulseAudio monitor 权限；提示音文件本身参与混音，所以就算现场喇叭没插、用户没听到，视频里也有声音。

容器需要 ffmpeg：

```bash
# 进容器后
apt-get install -y ffmpeg
# 或从宿主机一条命令
docker exec -it yolo26_csi_speak bash -c "apt-get install -y ffmpeg && ffmpeg -version | head -1"
```

没装 ffmpeg 也不会丢画面：视频照常保存成无声 mp4，音轨单独存成 `record_时间戳.mix.wav`，并打印手动合成命令：

```bash
ffmpeg -i record_2026-10-25_10-00-00.mp4 -i record_2026-10-25_10-00-00.mix.wav -c:v copy -c:a aac out.mp4
```

启动/结束日志可以确认状态：

```
[Recorder] 提示音混入录像: 开 (ffmpeg: /usr/bin/ffmpeg)
[Recorder] 已混入 12 条语音提示 → ./out/2026-10-25/record_2026-10-25_10-00-00.mp4
```

