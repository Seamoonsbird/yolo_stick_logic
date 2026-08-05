import time
from multiprocessing import Process, Queue, Event
from nodes.camera import camera_worker
from nodes.inference import inference_worker
from nodes.logic import logic_worker
from nodes.recorder import recorder_worker
from config import (
    ENABLE_RECORDER,
    FRAME_QUEUE_MAXSIZE,
    RESULT_QUEUE_MAXSIZE,
    RECORDER_QUEUE_MAXSIZE,
    STATE_QUEUE_MAXSIZE,
)


def main():
    frame_q = Queue(maxsize=FRAME_QUEUE_MAXSIZE)
    result_q = Queue(maxsize=RESULT_QUEUE_MAXSIZE)
    recorder_q = Queue(maxsize=RECORDER_QUEUE_MAXSIZE) if ENABLE_RECORDER else None
    state_q = Queue(maxsize=STATE_QUEUE_MAXSIZE)

    # 系统停止事件
    stop_event = Event()

    # ===========================================
    # 开启多进程（没有使用进程池）
    # ===========================================
    process_cam = Process(target=camera_worker,args=(frame_q, stop_event),name="Camera")
    process_inf = Process(target=inference_worker,args=(frame_q,result_q,recorder_q,stop_event),name="Inference")
    process_log = Process(target=logic_worker,args=(result_q,recorder_q,stop_event,state_q),name="Logic")

    print("多进程已创建，即将启动多进程")

    process_cam.daemon = True
    process_cam.start()

    process_inf.daemon = True
    process_inf.start()

    process_log.daemon = True
    process_log.start()

    if ENABLE_RECORDER:
        process_rec = Process(target=recorder_worker,args=(recorder_q,stop_event,state_q),name="Record")
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

    process_cam.join()
    process_inf.join()
    process_log.join()
    if ENABLE_RECORDER:
        process_rec.join()
    print("多进程已关闭")


if __name__ == "__main__":
    main()