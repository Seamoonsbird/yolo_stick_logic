"""
录制进程。

从 recorder_q 接收待录制的帧，从 state_q 接收状态信息（叠加到画面），
写入视频文件。

变更:
  - PIL 渲染中文（替换 OpenCV putText，解决中文问号问题）
  - 文件名带时间戳，不覆盖旧视频
  - 显示最近告警文案
  - VideoWriter FPS 根据头两帧到达间隔动态计算
"""

from __future__ import annotations

import queue
import time
import datetime
import os

import cv2
import numpy as np

from config import (
    RECORDER_CODEC,
    RECORDER_WIDTH,
    RECORDER_HEIGHT,
    get_output_dir,
)

# ==========================================
# 中文字体检测
# ==========================================

# 按优先级排列的 CJK 字体路径（Ubuntu / Jetson 常见位置）
_CJK_FONT_PATHS = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    # Jetson 常见
    "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf",  # 无中文，兜底
]

# PIL 字体缓存
_pil_fonts: dict[int, object] = {}  # size -> ImageFont
_pil_font_path: str | None = None


def _find_cjk_font() -> str | None:
    """查找第一个可用的 CJK 字体。"""
    for path in _CJK_FONT_PATHS:
        if os.path.isfile(path):
            return path
    return None


def _get_font(size: int):
    """获取指定大小的 PIL 字体（带缓存）。"""
    global _pil_font_path
    if size in _pil_fonts:
        return _pil_fonts[size]

    if _pil_font_path is None:
        _pil_font_path = _find_cjk_font()

    try:
        from PIL import ImageFont
        if _pil_font_path:
            font = ImageFont.truetype(_pil_font_path, size)
        else:
            font = ImageFont.load_default()
        _pil_fonts[size] = font
        return font
    except Exception:
        from PIL import ImageFont
        font = ImageFont.load_default()
        _pil_fonts[size] = font
        return font


# ==========================================
# PIL 中文叠加
# ==========================================

def _put_text_pil(frame, text: str, xy: tuple[int, int], *,
                  font_size: int = 20, color: tuple[int, int, int] = (0, 255, 0),
                  anchor: str = "lt"):
    """
    用 PIL 在 OpenCV BGR 帧上绘制文字（支持中文）。
    anchor: "lt"=左上, "rb"=右下, "rt"=右上, "lb"=左下
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # PIL 不可用 → 降级为 OpenCV putText（只能 ASCII）
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
        return

    font = _get_font(font_size)

    # OpenCV BGR → PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 获取文字尺寸用于定位
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x, y = xy
    if anchor == "rb":
        x, y = x - tw, y - th
    elif anchor == "rt":
        x, y = x - tw, y
    elif anchor == "lb":
        x, y = x, y - th

    draw.text((x, y), text, font=font, fill=color[::-1])  # BGR → RGB

    # 写回 OpenCV BGR
    rgb = np.array(pil_img)
    frame[:] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ==========================================
# VideoWriter
# ==========================================

def _build_video_writer(fps: float):
    """
    创建 cv2.VideoWriter，文件名带时间戳避免覆盖。
    """
    out_dir = get_output_dir()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"{out_dir}/record_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*RECORDER_CODEC)  # pyright: ignore[reportAttributeAccessIssue]

    writer = cv2.VideoWriter(
        filepath,
        fourcc,
        fps,
        (RECORDER_WIDTH, RECORDER_HEIGHT),
    )

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件：{filepath}")

    return writer, filepath


# ==========================================
# 画面叠加
# ==========================================

def _draw_overlay(frame, state: dict, frame_count: int, real_fps: float):
    """
    在帧画面上叠加信息（PIL 渲染，全中文支持）。

    左上角: 时间戳 + 帧序号 + 实时 FPS
    右上角: 录制指示灯
    左下角: 最近告警文案（高亮）
    右下角: 检测状态
    """
    h, w = frame.shape[:2]

    # ---- 左上角：时间戳 + 帧序号 + FPS ----
    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    _put_text_pil(frame, f"{ts}  |  frame #{frame_count}  |  {real_fps:.1f} fps",
                  (10, 5), font_size=18, color=(0, 255, 0))

    # ---- 右上角：录制指示灯 ----
    _put_text_pil(frame, "● REC", (w - 10, 5),
                  font_size=22, color=(0, 0, 255), anchor="rt")

    # ---- 右下角：检测状态 ----
    y_bottom = h - 5
    # 先画检测状态（从下往上排列）
    status_keys = [k for k in state.keys() if k not in ("最近提醒",)]
    # 按固定顺序排列：帧、盲道、楼梯、水坑、井盖、草地
    priority_order = ["帧", "盲道", "上楼梯", "下楼梯", "水坑", "井盖", "草地"]
    ordered = [k for k in priority_order if k in state] + \
              [k for k in status_keys if k not in priority_order]

    for key in reversed(ordered):
        value = state[key]
        text = f"{key}: {value}" if value else key
        _put_text_pil(frame, text, (w - 10, y_bottom),
                      font_size=18, color=(0, 255, 0), anchor="rb")
        y_bottom -= 22

    # ---- 左下角：最近提醒（黄色醒目） ----
    alerts = state.get("最近提醒", [])
    if alerts:
        y_alert = h - 5
        for alert_text in reversed(alerts[-3:]):  # 最多显示最近 3 条
            _put_text_pil(frame, f"⚠ {alert_text}", (10, y_alert),
                          font_size=20, color=(0, 255, 255), anchor="lb")
            y_alert -= 26

    return frame


# ==========================================
# 主循环
# ==========================================

def recorder_worker(recorder_q, stop_event, state_q):
    """
    录制主循环。

    工作流程:
        1. 等待第一帧到达
        2. 等待第二帧到达 → 根据时间间隔计算真实 FPS
        3. 以真实 FPS 创建 VideoWriter，写入缓冲的 2 帧
        4. 循环取帧 → 叠加信息 → 写入视频
        5. stop_event 触发后清空残帧、释放资源
    """
    print("[Recorder] 正在等待首帧以测量真实帧率...")

    # 检测中文字体
    font_path = _find_cjk_font()
    if font_path:
        print(f"[Recorder] 中文字体: {font_path}")
    else:
        print("[Recorder] ⚠ 未找到中文字体，中文将降级显示。请安装: apt install fonts-wqy-zenhei")

    current_state: dict = {}
    frame_count = 0

    # ---- 阶段 1：缓冲开头两帧，测量真实 FPS ----
    buffered_frames = []
    t_first = 0.0
    t_second = 0.0

    def _fetch_frame():
        """从 recorder_q 取一帧，超时返回 None。"""
        try:
            return recorder_q.get(timeout=1.0)
        except queue.Empty:
            return None

    # 等第一帧
    while not stop_event.is_set():
        frame = _fetch_frame()
        if frame is not None:
            t_first = time.time()
            frame_count += 1
            frame = _draw_overlay(frame, current_state, frame_count, 0.0)
            buffered_frames.append(frame)
            print("[Recorder] 收到首帧")
            break

    if stop_event.is_set():
        print("[Recorder] 未收到任何帧，退出")
        return

    # 等第二帧
    while not stop_event.is_set():
        frame = _fetch_frame()
        if frame is not None:
            t_second = time.time()
            frame_count += 1
            frame = _draw_overlay(frame, current_state, frame_count, 0.0)
            buffered_frames.append(frame)
            break

    if stop_event.is_set() or len(buffered_frames) < 2:
        print("[Recorder] 帧数不足，退出")
        return

    # 计算真实 FPS
    delta = t_second - t_first
    if delta <= 0:
        delta = 0.2
    real_fps = 1.0 / delta
    real_fps = max(1.0, min(real_fps, 60.0))
    print(f"[Recorder] 实测帧率: {real_fps:.1f} fps (帧间隔: {delta*1000:.0f}ms)")

    # ---- 阶段 2：创建 VideoWriter（文件名带时间戳） ----
    writer, filepath = _build_video_writer(real_fps)
    print(f"[Recorder] 视频文件已创建: {filepath}")

    # 写入缓冲的两帧
    for f in buffered_frames:
        writer.write(f)

    # ---- 阶段 3：主循环 ----
    try:
        while not stop_event.is_set():
            # 非阻塞捞取最新状态
            while True:
                try:
                    current_state = state_q.get_nowait()
                except queue.Empty:
                    break

            # 从录制队列取帧
            try:
                frame = recorder_q.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_count += 1
            frame = _draw_overlay(frame, current_state, frame_count, real_fps)
            writer.write(frame)

        # ---- 阶段 4：退出清空残帧 ----
        print("[Recorder] 收到停止信号，正在清空剩余帧...")
        while True:
            try:
                frame = recorder_q.get_nowait()
            except queue.Empty:
                break
            frame_count += 1
            frame = _draw_overlay(frame, current_state, frame_count, real_fps)
            writer.write(frame)

    finally:
        writer.release()
        print(f"[Recorder] 视频已保存: {filepath}  "
              f"(共 {frame_count} 帧, {real_fps:.1f} fps)")
