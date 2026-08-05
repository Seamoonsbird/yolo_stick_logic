"""
推理进程。

从 frame_q 取帧，运行 YOLO 推理，将检测结果绘制的画面放入 recorder_q，
将原始推理结果放入 result_q 供逻辑进程使用。
"""
import queue
import time

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    DEVICE,
    USE_FP16,
    TARGET_CLASSES,
    CLASS_NAMES,
    PRINT_FPS,
)


def _class_name(cls_id: int) -> str:
    """
    返回类别 ID 对应的名称。

    优先使用 config.CLASS_NAMES（自定义模型），
    未定义则返回 "cls_N" 格式。
    """
    return CLASS_NAMES.get(cls_id, f"cls_{cls_id}")


def _load_model():
    """
    加载 YOLO 模型（支持 .pt / .onnx / .engine 等格式）。

    在子进程内调用，避免通过 pickle 序列化模型（multiprocessing 不支持）。
    """
    print(f"[Inference] 正在加载模型: {MODEL_PATH}")
    print(f"[Inference] 设备: {DEVICE}, FP16: {USE_FP16}")

    model = YOLO(MODEL_PATH)

    # ONNX 模型不需要预热（没有 CUDA kernel 编译），直接开始推理
    is_onnx = MODEL_PATH.endswith(".onnx")
    if is_onnx:
        print("[Inference] ONNX 模型加载完毕，开始推理")
    else:
        # 预热模型：跑一次空推理，触发 CUDA kernel 编译和模型加载到显存
        print("[Inference] 预热模型中（首次推理较慢）...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        model(
            dummy,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            half=USE_FP16 and DEVICE == "cuda",
            verbose=False,
        )
        print("[Inference] 模型加载完毕，开始推理")

    return model


def _draw_boxes(frame: np.ndarray, results) -> np.ndarray:
    """
    在帧上绘制 YOLO 检测框和标签。

    参数:
        frame:   原始帧（不会被修改，返回新副本）
        results: ultralytics Results 对象（单帧）

    返回:
        绘制了检测框的帧副本
    """
    annotated = frame.copy()

    if results.boxes is None:
        return annotated

    for box in results.boxes:
        # 获取边界框坐标（左上角 x1, y1, 右下角 x2, y2）
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # 类别与置信度
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # 如果配置了目标类别过滤，跳过不关心的类别
        if TARGET_CLASSES and cls_id not in TARGET_CLASSES:
            continue

        # 类别名
        label = _class_name(cls_id)
        text = f"{label} {conf:.2f}"

        # 框的颜色（按类别 ID 变化，保证不同类别颜色不同）
        color = _class_color(cls_id)

        # 绘制矩形框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景和文字
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


def _class_color(cls_id: int) -> tuple:
    """
    根据类别 ID 生成固定颜色（BGR 格式）。

    不同类别颜色差异大，便于肉眼区分。
    """
    # 用乘法 + 取模生成色相偏移
    hue = (cls_id * 43) % 180  # OpenCV H 范围 0-179
    # 转为 BGR
    hsv = np.array([[[hue, 200, 200]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def _extract_results(results) -> list:
    """
    从 ultralytics Results 对象中提取检测数据到 CPU 纯 Python 结构。

    必须做这一步：result_q 通过 pickle 序列化跨进程传递，
    GPU tensor（CUDA）无法被 pickle，必须 .cpu() 搬到 CPU 并转成
    普通 Python 类型后再入队。

    返回格式：
        [
            {"box": [x1, y1, x2, y2], "cls": int, "conf": float, "name": str},
            ...
        ]
    """
    detections = []

    if results.boxes is None:
        return detections

    boxes = results.boxes
    xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
    cls_ids = boxes.cls.cpu().tolist() if boxes.cls is not None else []
    confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []

    for i in range(len(xyxy)):
        cls_id = int(cls_ids[i])
        conff = float(confs[i])

        # 类别过滤
        if TARGET_CLASSES and cls_id not in TARGET_CLASSES:
            continue

        detections.append({
            "box": [float(v) for v in xyxy[i]],
            "cls": cls_id,
            "conf": conff,
            "name": _class_name(cls_id),
        })

    return detections


def _print_detections(frame_count: int, t_ms: float, detections: list):
    """
    每帧打印检测结果到控制台，每个物体独占一行。

    输出格式：
        [Inference] #128 | 198ms
          person    0.87  box[ 100, 200, 300, 400]  area= 20000
          car       0.72  box[  50,  60, 150, 200]  area= 14000

    无检测时：
        [Inference] #128 | 198ms | 0 objects
    """
    dt_str = f"{t_ms:.0f}ms"

    if not detections:
        print(f"[Inference] #{frame_count} | {dt_str} | 0 objects")
        return

    # 帧头行
    print(f"[Inference] #{frame_count} | {dt_str}")

    # 每个检测结果一行，列对齐
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        area = (x2 - x1) * (y2 - y1)
        print(
            f"  {d['name']:<12s} {d['conf']:.2f}  "
            f"box[{x1:4.0f},{y1:4.0f},{x2:4.0f},{y2:4.0f}]  "
            f"area={area:7.0f}"
        )


def _put_result(result_q, detections: list):
    """
    将检测结果放入结果队列（保持最新策略）。
    """
    if result_q.full():
        try:
            result_q.get_nowait()
        except queue.Empty:
            pass
    result_q.put(detections)


def _put_frame(recorder_q, frame):
    """
    将标注帧放入录制队列（保持最新策略）。
    """
    if recorder_q is None:
        return
    if recorder_q.full():
        try:
            recorder_q.get_nowait()
        except queue.Empty:
            pass
    recorder_q.put(frame)


def inference_worker(frame_q, result_q, recorder_q, stop_event):
    """
    推理主循环。

    1. 加载 YOLO 模型
    2. 循环取帧 → 推理 → 绘框 → 分发到 result_q / recorder_q
    """
    # ---- 模型加载（在子进程内，避免 pickle） ----
    model = _load_model()
    is_onnx = MODEL_PATH.endswith(".onnx")

    # ---- 帧计数器 & FPS 统计 ----
    frame_count = 0
    fps_window: list[float] = []  # 最近 N 次推理耗时(ms)
    fps_log_time = time.time()

    print("[Inference] 推理主循环已启动")

    while not stop_event.is_set():
        # ---- 步骤 1：从摄像头队列取帧 ----
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # ---- 步骤 2：YOLO 推理 ----
        t_start = time.time()

        # ONNX 模型不支持 half 参数
        if is_onnx:
            results_list = model(
                frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )
        else:
            results_list = model(
                frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                half=USE_FP16 and DEVICE == "cuda",
                verbose=False,
            )

        results = results_list[0]  # 单帧推理，取第一个
        t_infer = (time.time() - t_start) * 1000  # 毫秒
        frame_count += 1

        # ---- 步骤 3：绘制检测框 ----
        annotated = _draw_boxes(frame, results)

        # ---- 步骤 4：提取检测数据（GPU → CPU）并分发 ----
        detections = _extract_results(results)
        _put_result(result_q, detections)     # CPU 数据 → 逻辑进程
        _put_frame(recorder_q, annotated)     # 标注画面 → 录制进程

        # ---- 步骤 5：每帧打印检测结果 ----
        _print_detections(frame_count, t_infer, detections)

        # ---- FPS 汇总统计 ----
        fps_window.append(t_infer)
        if PRINT_FPS and len(fps_window) >= 30:
            now = time.time()
            if now - fps_log_time >= 5.0:
                avg = sum(fps_window) / len(fps_window)
                fps = 1000 / avg if avg > 0 else 0
                print(f"[Inference] === 近 {len(fps_window)} 帧汇总 | "
                      f"平均推理: {avg:.1f}ms | FPS: {fps:.0f} ===")
                fps_window.clear()
                fps_log_time = now

    print(f"[Inference] 推理进程已退出 (共处理 {frame_count} 帧)")