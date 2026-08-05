"""
录制进程。

从 recorder_q 接收待录制的帧，从 state_q 接收状态信息（叠加到画面），
写入视频文件。支持优雅退出和资源清理。
"""
import queue

import cv2

from config import (
    RECORDER_CODEC,
    RECORDER_FPS,
    RECORDER_WIDTH,
    RECORDER_HEIGHT,
    get_output_dir,
)


def _build_video_writer():
    """
    创建 cv2.VideoWriter 实例。

    返回 (writer, filepath)，其中 filepath 用于日志/保存确认。
    """
    out_dir = get_output_dir()
    filepath = f"{out_dir}/record.mp4"

    # fourcc: 四个字符的编码标识。
    # pyright 的 cv2 stub 未包含此 C 扩展函数，忽略类型检查
    fourcc = cv2.VideoWriter_fourcc(*RECORDER_CODEC)  # pyright: ignore[reportAttributeAccessIssue]

    writer = cv2.VideoWriter(
        filepath,
        fourcc,
        RECORDER_FPS,
        (RECORDER_WIDTH, RECORDER_HEIGHT),
    )

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件：{filepath}")

    return writer, filepath


def _draw_overlay(frame, state: dict, frame_count: int):
    """
    在帧画面上叠加信息。

    左上角：时间戳 + 帧序号（始终显示，不依赖 state_q）
    右下角：state_q 传来的状态信息（如比分、时间等，有则显示）

    state 示例：
        {"score": "2:1", "time": "12:34", "period": "Q2"}
    """
    import datetime

    h, w = frame.shape[:2]

    # ---- 左上角：时间戳 + 帧序号 ----
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame,
        f"{timestamp}  |  frame #{frame_count}",
        (10, 30),  # 左上角
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
        1. 创建 VideoWriter
        2. 循环：
           a. 从 recorder_q 非阻塞取帧（超时 0.5s）
           b. 从 state_q 非阻塞取最新状态（只保留最新一条）
           c. 如果拿到了帧 → 叠加状态 → 写入视频
           d. 如果 stop_event 已触发且 recorder_q 已清空 → 退出
        3. 释放 VideoWriter
    """
    print("[Recorder] 正在初始化视频写入器...")

    writer, filepath = _build_video_writer()
    print(f"[Recorder] 视频文件已创建: {filepath}")

    # 当前状态缓存 + 帧计数器
    current_state: dict = {}
    frame_count = 0

    try:
        while not stop_event.is_set():
            # ---- 步骤 1：非阻塞捞取最新状态 ----
            while True:
                try:
                    current_state = state_q.get_nowait()
                except queue.Empty:
                    break

            # ---- 步骤 2：从录制队列取帧 ----
            try:
                frame = recorder_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # ---- 步骤 3：叠加信息（始终执行） ----
            frame_count += 1
            frame = _draw_overlay(frame, current_state, frame_count)

            # ---- 步骤 4：写入视频文件 ----
            writer.write(frame)

        # ================================================
        # 正常退出：stop_event 触发，清空队列中残余帧
        # ================================================
        print("[Recorder] 收到停止信号，正在清空剩余帧...")
        while True:
            try:
                frame = recorder_q.get_nowait()
            except queue.Empty:
                break
            frame_count += 1
            frame = _draw_overlay(frame, current_state, frame_count)
            writer.write(frame)

    finally:
        # ================================================
        # 无论如何都要释放 VideoWriter，避免文件损坏
        # ================================================
        writer.release()
        print(f"[Recorder] 视频已保存: {filepath}")