import time
from multiprocessing import Process, Queue, Event
from nodes.camera import camera_worker
from nodes.inference import inference_worker
from nodes.logic import logic_worker
from nodes.recorder import recorder_worker
from config import (
    ENABLE_RECORDER,
    MUX_ALERT_AUDIO,
    SHUTDOWN_TIMEOUT,
    FRAME_QUEUE_MAXSIZE,
    RESULT_QUEUE_MAXSIZE,
    RECORDER_QUEUE_MAXSIZE,
    STATE_QUEUE_MAXSIZE,
    AUDIO_EVENT_QUEUE_MAXSIZE,
)


def main():
    frame_q = Queue(maxsize=FRAME_QUEUE_MAXSIZE)
    result_q = Queue(maxsize=RESULT_QUEUE_MAXSIZE)
    recorder_q = Queue(maxsize=RECORDER_QUEUE_MAXSIZE) if ENABLE_RECORDER else None
    state_q = Queue(maxsize=STATE_QUEUE_MAXSIZE)

    # 播报事件队列：逻辑进程写入「实际播了哪条语音」，录制进程按时间轴混进视频
    audio_q = (Queue(maxsize=AUDIO_EVENT_QUEUE_MAXSIZE)
               if (MUX_ALERT_AUDIO and ENABLE_RECORDER) else None)

    # 系统停止事件
    stop_event = Event()

    # ===========================================
    # 开启多进程（没有使用进程池）
    # ===========================================
    process_cam = Process(target=camera_worker,args=(frame_q, stop_event),name="Camera")
    process_inf = Process(target=inference_worker,args=(frame_q,result_q,recorder_q,stop_event),name="Inference")
    process_log = Process(target=logic_worker,args=(result_q,recorder_q,stop_event,state_q,audio_q),name="Logic")

    print("多进程已创建，即将启动多进程")

    process_cam.daemon = True
    process_cam.start()

    process_inf.daemon = True
    process_inf.start()

    process_log.daemon = True
    process_log.start()

    if ENABLE_RECORDER:
        process_rec = Process(target=recorder_worker,args=(recorder_q,stop_event,state_q,audio_q),name="Record")
        process_rec.daemon = True
        process_rec.start()
        print("开启录制")
    else:
        print("未开启录制")


    try:
        #主循环：监控退出信号
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n检测到用户中断，正在关闭系统...")
        stop_event.set()

    # ================================================
    # 清理资源
    # ================================================
    # 录制进程收尾时要把 mp4 的索引（moov）写进文件，这期间不能再被打断：
    # 这里给每个进程限时等待，重复按 Ctrl+C 也不会中断收尾，只有真的卡死
    # 超过 SHUTDOWN_TIMEOUT 才 terminate（那时文件可能不完整）。
    procs = [process_cam, process_inf, process_log]
    if ENABLE_RECORDER:
        procs.append(process_rec)

    deadline = time.time() + SHUTDOWN_TIMEOUT
    for p in procs:
        while p.is_alive() and time.time() < deadline:
            try:
                p.join(timeout=1.0)
            except KeyboardInterrupt:
                print(f"[系统] 正在等 {p.name} 收尾（视频索引写入中），请等待完成…")
        if p.is_alive():
            print(f"[系统] ⚠ {p.name} 超过 {SHUTDOWN_TIMEOUT}s 未退出，强制结束"
                  f"（正在录制的视频可能不完整）")
            p.terminate()
            p.join(timeout=5)

    print("多进程已关闭")


if __name__ == "__main__":
    main()