"""
录制进程。

从 recorder_q 接收待录制的帧，从 state_q 接收状态信息（叠加到画面），
写入视频文件。

VideoWriter 的 FPS 不是硬编码的，而是根据开头两帧的真实到达间隔
动态计算，确保视频播放速度与实际录制速度一致，不会快进或慢放。
"""
import queue
import time

import cv2

from config import (
    RECORDER_CODEC,
    RECORDER_WIDTH,
    RECORDER_HEIGHT,
    get_output_dir,
)


def _build_video_writer(fps: float):
    """
    创建 cv2.VideoWriter 实例。

    参数:
        fps: 实际测量的帧率（非硬编码的配置值）

    返回 (writer, filepath)。
    """
    out_dir = get_output_dir()
    filepath = f"{out_dir}/record.mp4"

    # fourcc: 四个字符的编码标识。
    # pyright 的 cv2 stub 未包含此 C 扩展函数，忽略类型检查
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


def _draw_overlay(frame, state: dict, frame_count: int, real_fps: float):
    """
    在帧画面上叠加信息。

    左上角：时间戳 + 帧序号 + 实时 FPS（始终显示，不依赖 state_q）
    右上角：录制指示灯
    右下角：state_q 传来的状态信息（有则显示）
    """
    import datetime

    h, w = frame.shape[:2]

    # ---- 左上角：时间戳 + 帧序号 + 实际帧率 ----
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame,
        f"{timestamp}  |  frame #{frame_count}  |  {real_fps:.1f} fps",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),  # 绿色
        2,
        cv2.LINE_AA,
    )

    # ---- 右上角：录制指示灯 ----
    cv2.putText(
        frame,
        "● REC",
        (w - 100, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),  # 红色
        2,
        cv2.LINE_AA,
    )

    # ---- 右下角：state_q 传来的额外状态 ----
    y_offset = h - 20
    for key, value in reversed(list(state.items())):
        text = f"{key}: {value}"
        cv2.putText(
            frame,
            text,
            (w - 250, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y_offset -= 25

    return frame


def recorder_worker(recorder_q, stop_event, state_q):
    """
    录制主循环。

    工作流程:
        1. 等待第一帧到达（不创建文件）
        2. 等待第二帧到达 → 根据时间间隔计算真实 FPS
        3. 以真实 FPS 创建 VideoWriter，写入缓冲的 2 帧
        4. 循环取帧 → 叠加信息 → 写入视频
        5. stop_event 触发后清空残帧、释放资源
    """
    print("[Recorder] 正在等待首帧以测量真实帧率...")

    current_state: dict = {}
    frame_count = 0

    # ---- 阶段 1：缓冲开头两帧，测量真实 FPS ----
    # VideoWriter 需要固定 FPS，但实际推理速度未知（可能是 5fps 也可能是 30fps）。
    # 用头两帧的到达间隔计算真实帧率，保证视频播放速度与真实时间一致。
    buffered_frames = []
    t_first = 0.0   # 初始化让类型检查器满意
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
        delta = 0.2  # 保护：最小间隔 200ms = 5 FPS
    real_fps = 1.0 / delta

    # 限制在合理范围
    real_fps = max(1.0, min(real_fps, 60.0))
    print(f"[Recorder] 实测帧率: {real_fps:.1f} fps (帧间隔: {delta*1000:.0f}ms)")

    # ---- 阶段 2：以真实 FPS 创建 VideoWriter ----
    writer, filepath = _build_video_writer(real_fps)
    print(f"[Recorder] 视频文件已创建: {filepath}")

    # 写入缓冲的两帧（更新叠加以显示真实 FPS）
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