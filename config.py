"""
全局配置文件。

所有模块通过 `from config import ...` 读取参数，避免硬编码散落各处。
修改配置只需改这一个文件。
"""
import os
from datetime import datetime

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

# ============================================================
# YOLO 推理参数
# ============================================================

# 模型文件路径（支持 .pt / .engine / .onnx 等格式）
MODEL_PATH = "./models/origin.pt"

# 置信度阈值：低于此值的检测结果直接丢弃
CONF_THRESHOLD = 0.5

# IoU 阈值：NMS（非极大值抑制）时使用
IOU_THRESHOLD = 0.45

# 推理设备
#   "cuda"  → NVIDIA GPU（Jetson 上推荐）
#   "cpu"   → 纯 CPU 推理（备选，速度较慢）
DEVICE = "cuda"

# 是否使用半精度推理（FP16），Jetson GPU 支持可开启以加速
USE_FP16 = True

# 目标类别 ID 列表。留空 [] 表示检测所有类别。
# YOLO 默认 0=person, 等等。按需要填写。
TARGET_CLASSES = []

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

# 录制帧率
RECORDER_FPS = 30

# 录制分辨率（默认跟随摄像头分辨率）
RECORDER_WIDTH = CAMERA_WIDTH
RECORDER_HEIGHT = CAMERA_HEIGHT

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