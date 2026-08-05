"""
推理进程。

从 frame_q 取帧，运行 YOLO 推理。标注画面 → recorder_q，检测数据 → result_q。
"""
import queue
import time

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
    return CLASS_NAMES.get(cls_id, f"cls_{cls_id}")


def _load_model():
    """加载 YOLO 模型并预热。"""
    print(f"[Inference] 加载模型: {MODEL_PATH}  (device={DEVICE}, fp16={USE_FP16})")

    model = YOLO(MODEL_PATH)

    print("[Inference] 预热中...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model(dummy, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
          device=DEVICE, half=USE_FP16 and DEVICE == "cuda", verbose=False)
    print("[Inference] 就绪")

    return model


def _extract_results(results) -> list:
    """
    从 ultralytics Results 提取检测数据到 CPU（避开 GPU tensor 无法 pickle 的问题）。
    返回: [{"box": [x1,y1,x2,y2], "cls": int, "conf": float, "name": str}, ...]
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
        if TARGET_CLASSES and cls_id not in TARGET_CLASSES:
            continue
        detections.append({
            "box": [float(v) for v in xyxy[i]],
            "cls": cls_id,
            "conf": float(confs[i]),
            "name": _class_name(cls_id),
        })
    return detections


def _print_detections(frame_count: int, t_ms: float, detections: list):
    """每帧打印检测结果，每物体一行。"""
    dt_str = f"{t_ms:.0f}ms"
    if not detections:
        print(f"[Inference] #{frame_count} | {dt_str} | 0 objects")
        return

    print(f"[Inference] #{frame_count} | {dt_str}")
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        area = (x2 - x1) * (y2 - y1)
        print(f"  {d['name']:<12s} {d['conf']:.2f}  "
              f"box[{x1:4.0f},{y1:4.0f},{x2:4.0f},{y2:4.0f}]  area={area:7.0f}")


def _put_result(result_q, detections: list):
    """检测数据入队（保持最新策略）。"""
    if result_q.full():
        try:
            result_q.get_nowait()
        except queue.Empty:
            pass
    result_q.put(detections)


def _put_frame(recorder_q, frame):
    """标注帧入队（保持最新策略）。"""
    if recorder_q is None:
        return
    if recorder_q.full():
        try:
            recorder_q.get_nowait()
        except queue.Empty:
            pass
    recorder_q.put(frame)


def inference_worker(frame_q, result_q, recorder_q, stop_event):
    """推理主循环。"""
    model = _load_model()

    frame_count = 0
    fps_window = []
    fps_log_time = time.time()

    print("[Inference] 主循环开始")

    while not stop_event.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # YOLO 推理
        t_start = time.time()
        results = model(frame,
                        conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                        device=DEVICE, half=USE_FP16 and DEVICE == "cuda",
                        verbose=False)[0]
        t_infer = (time.time() - t_start) * 1000
        frame_count += 1

        # results.plot() 一键绘框（比手写 _draw_boxes 简洁）
        annotated = results.plot()

        # 提取检测数据（GPU → CPU）并分发
        detections = _extract_results(results)
        _put_result(result_q, detections)
        _put_frame(recorder_q, annotated)

        # 控制台输出
        _print_detections(frame_count, t_infer, detections)

        # FPS 汇总
        fps_window.append(t_infer)
        if PRINT_FPS and len(fps_window) >= 30:
            now = time.time()
            if now - fps_log_time >= 5.0:
                avg = sum(fps_window) / len(fps_window)
                fps = 1000 / avg if avg > 0 else 0
                print(f"[Inference] === {len(fps_window)}帧汇总 | "
                      f"平均 {avg:.0f}ms | FPS {fps:.0f} ===")
                fps_window.clear()
                fps_log_time = now

    print(f"[Inference] 退出 (共 {frame_count} 帧)")