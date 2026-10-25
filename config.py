"""
全局配置文件。

所有模块通过 `from config import ...` 读取参数，避免硬编码散落各处。
修改配置只需改这一个文件。
"""
import os
from datetime import datetime
from typing import Dict

# ============================================================
# 摄像头参数
# ============================================================

# CSI 摄像头编号（Jetson 上第一个 CSI 口通常是 0）
CAMERA_SENSOR_ID = 0

# 图像分辨率（宽 × 高）
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# 帧率（每秒帧数）
CAMERA_FPS = 30

# 是否使用 CSI 摄像头。
# True  → 走 GStreamer 管线（open_jetson_csi_capture）
# False → 走 cv2.VideoCapture（USB 摄像头或视频文件）
USE_CSI_CAMERA = True

# USB 摄像头的设备索引（仅当 USE_CSI_CAMERA=False 时生效）
USB_CAMERA_INDEX = 0

# 是否把画面逆时针旋转 90°。
# Jetson CSI 摄像头（nvarguscamerasrc）经常是竖装/横装反的，
# 打开后画面方向不对时把这里设为 True 校正。
# 旋转在 nodes/camera.py 里、帧进入队列之前完成，
# 所以推理、逻辑、录制拿到的都是旋转后的同一套坐标。
ROTATE_CCW_90 = True

# ---- 旋转后的有效画面尺寸（下游一律用这两个常量）----
# 逆时针转 90° 后宽高互换：640x480 → 480x640
if ROTATE_CCW_90:
    FRAME_WIDTH, FRAME_HEIGHT = CAMERA_HEIGHT, CAMERA_WIDTH     # 480 x 640
else:
    FRAME_WIDTH, FRAME_HEIGHT = CAMERA_WIDTH, CAMERA_HEIGHT     # 640 x 480

# ============================================================
# YOLO 推理参数
# ============================================================

# 模型文件路径（支持 .pt / .engine / .onnx 等格式）
MODEL_PATH = "./models/best.pt"

# 置信度阈值：低于此值的检测结果直接丢弃
CONF_THRESHOLD = 0.6  # 低于此值的检测结果直接丢弃

# IoU 阈值：NMS（非极大值抑制）时使用
IOU_THRESHOLD = 0.45

# 推理设备
#   "cuda"  → NVIDIA GPU（Jetson 上推荐）
#   "cpu"   → 纯 CPU 推理（备选，速度较慢）
DEVICE = "cuda"

# 是否使用半精度推理（FP16），Jetson GPU 支持可开启以加速
USE_FP16 = True

# 目标类别 ID 列表。留空 [] 表示检测所有类别。
TARGET_CLASSES = []

# 自定义类别名映射（你的 origin.onnx 是自己训练的模型，不是 COCO 80 类）。
# 按你的模型实际类别填写，格式：{类别ID: "名称", ...}
# 未在此映射中的类别 ID 会显示为 "cls_N"。
# 示例（冰球场景）：
# CLASS_NAMES = {0: "stick", 1: "player", 2: "puck", 3: "goal"}
CLASS_NAMES: Dict[int, str] = {}

# ============================================================
# 队列参数
# ============================================================

# 帧队列最大长度：摄像头 → 推理
# 设为 2 保证推理拿到的是最新帧，同时避免内存堆积
FRAME_QUEUE_MAXSIZE = 2

# 结果队列最大长度：推理 → 逻辑
RESULT_QUEUE_MAXSIZE = 2

# 录制队列最大长度：逻辑 → 录制
RECORDER_QUEUE_MAXSIZE = 2

# 状态队列最大长度：逻辑 → 录制（用于叠加状态文字到画面）
STATE_QUEUE_MAXSIZE = 1

# 播报事件队列最大长度：逻辑 → 录制（用于把提示音混进视频）
AUDIO_EVENT_QUEUE_MAXSIZE = 200

# ============================================================
# 录制参数
# ============================================================

# 是否开启视频录制
ENABLE_RECORDER = True

# 录制编码格式
#   "mp4v" → .mp4 文件，通用兼容性好
#   "avc1" → H.264，压缩率高
#   "XVID" → .avi 文件
RECORDER_CODEC = "mp4v"

# 录制帧率由 recorder 根据前两帧实际到达间隔自动测量，无需硬编码。
# 如果头两帧间隔就是 5fps，视频就标 5fps——播放速度 = 真实速度。

# 录制分辨率（跟随旋转后的有效画面尺寸，不可再用 CAMERA_WIDTH/HEIGHT，
# 否则旋转后写出的视频会被拉伸变形）
RECORDER_WIDTH = FRAME_WIDTH
RECORDER_HEIGHT = FRAME_HEIGHT

# ---- 提示音混入录像 ----
# True  → 录完把本次播放过的语音提示按时间轴混成音轨，合成带声音的 mp4
#         （视频流直接复制，不重新编码，画质无损；音频编码 AAC）
# False → 只录画面，不生成音轨
MUX_ALERT_AUDIO = True

# 混音采样率（Hz）。22050 对语音足够、文件更小；想要更通用可改 44100
AUDIO_MIX_RATE = 22050

# 依赖 ffmpeg：容器里没有会降级成「视频无声音 + 单独保存 .wav」，
# 安装：apt-get install -y ffmpeg

# ============================================================
# 输出路径
# ============================================================

# 录像 & 截图保存目录
# {date} 会被替换为日期（如 2026-08-05）
# {time} 会被替换为时间戳（如 2026-08-05_14-30-00）
OUTPUT_DIR = "./out/{date}"

# ============================================================
# 调试 & 日志
# ============================================================

# 是否打印各进程的 FPS（每秒帧数/推理次数）
PRINT_FPS = True

# 是否在画面右上角显示 FPS 文字叠加
SHOW_FPS_ON_FRAME = False

# ============================================================
# 音频 / 语音播报
# ============================================================
# 播报策略（优先级从高到低）:
#   1. 预录人声 WAV —— 放在 SOUNDS_DIR 下，文件名见 nodes/logic.py 的 SOUND_PRESETS。
#      音质最清晰、最可靠，盲杖这类无障碍设备首选。
#   2. espeak-ng 实时合成 —— 当某个预录文件缺失时自动兜底，保证不静默。
#   3. 占位蜂鸣 —— 连 espeak 都不可用时的最后有声反馈。
# 不依赖任何需编译的 Python 音频库（sounddevice/pyaudio/simpleaudio 等）。
# 播放统一走 paplay(PulseAudio)/aplay(ALSA) → 系统默认输出（宿主机 USB 声卡）。

# espeak-ng 中文语音参数（仅作为预录缺失时的兜底合成，适当放慢更清晰）
ESPEAK_VOICE = "zh"      # 普通话
ESPEAK_SPEED = 130       # 语速（词/分钟），兜底合成用，稍慢更清晰
ESPEAK_VOLUME = 150      # 音量 0-200

# 音频资源目录（预录中文 WAV + 占位蜂鸣）。用绝对路径，避免容器 cwd 不同导致找不到。
SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sounds")

# espeak-ng 不可用时的兜底提示音（短蜂鸣），保证至少有声反馈
FALLBACK_WAV = os.path.join(SOUNDS_DIR, "fallback_alert.wav")

# ============================================================
# 工具函数
# ============================================================

def get_output_dir():
    """
    返回格式化后的输出目录路径，自动创建目录。
    用法：
        out_dir = get_output_dir()
        # → "./out/2026-08-05"
    """
    now = datetime.now()
    dir_path = OUTPUT_DIR.format(
        date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(dir_path, exist_ok=True)
    return dir_path