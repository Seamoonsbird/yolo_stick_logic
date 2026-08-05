"""
推理进程。

从 frame_q 取帧，运行 YOLO 推理，将结果放入 result_q，
同时将帧转发到 recorder_q 供录制。
"""
import queue


def inference_worker(frame_q, result_q, recorder_q, stop_event):
    """
    推理主循环。

    TODO: 集成 YOLO 模型推理。
          当前为透传模式——帧直接转发给 recorder，不做推理。
    """
    print("[Inference] 推理进程已启动（当前为透传模式）")

    while not stop_event.is_set():
        # ---- 步骤 1：从摄像头队列取帧 ----
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # ---- 步骤 2：TODO YOLO 推理 ----
        # results = model(frame, ...)
        # result_q.put(results)

        # ---- 步骤 3：转发帧给录制队列 ----
        if recorder_q is not None:
            # 保持最新帧策略：队列满了就丢旧帧
            if recorder_q.full():
                try:
                    recorder_q.get_nowait()
                except queue.Empty:
                    pass
            recorder_q.put(frame)

    print("[Inference] 推理进程已退出")