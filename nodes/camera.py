import cv2
import shutil
import subprocess
import numpy as np

class GstPipeCapture:
    # 初始化函数
    def __init__(self, sensor_id=0, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        # 一帧 BGR 数据的总字节数：每个像素占 3 个通道（B、G、R），每通道 1 字节
        self.frame_bytes = width * height * 3
        # 要输入的指令
        cmd = (
            f"gst-launch-1.0 --quiet -e "
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"'video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1' ! "
            f"nvvidconv ! "
            f"'video/x-raw,width={width},height={height},format=BGRx' ! "
            f"videoconvert ! "
            f"'video/x-raw,format=BGR' ! "
            f"fdsink"
        )
        self._proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_bytes * 2,
        )
    # 判断摄像头状态函数
    def isOpened(self):
        return self._proc.poll() is None
    # 读取图像
    def read(self):
        raw = b""
        while len(raw) < self.frame_bytes:
            chunk = self._proc.stdout.read(self.frame_bytes -len(raw))
            if not chunk:
                return False, None
            raw += chunk
        # 将原始字节流解析为 numpy 数组
        # dtype=np.uint8: 每个字节是 0-255 的无符号整数
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        # .copy() 创建独立副本：
        # frombuffer 返回的数组共享底层 buffer，
        # 下一次 read 会覆盖 buffer 内容，所以必须 copy
        return True, frame.copy()
    # 释放摄像头
    def release(self):
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    # 获得摄像头属性
    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    # 设置摄像头属性（空实现，兼容 cv2.VideoCapture.set）。
    def set(self, prop_id, value):
        pass



class _PrefetchCapture:
    def __init__(self, inner, first_frame):
        self._inner = inner
        self._first = first_frame  # 缓存预读帧，非 None 表示还未被消费

    def isOpened(self):
        return self._first is not None or self._inner.isOpened()

    def read(self):
        """
        读取一帧。
        第一次调用时返回缓存的预读帧；
        之后调用全部委托给底层 inner。
        """
        if self._first is not None:
            # 返回预读帧，并将 _first 置为 None 表示已消费
            frame = self._first
            self._first = None
            return True, frame
        # 预读帧已消费，正常从底层读取
        return self._inner.read()

    def release(self):
        """转发给底层 inner"""
        self._inner.release()

    def get(self, prop_id):
        """转发给底层 inner"""
        return self._inner.get(prop_id)

    def set(self, prop_id, value):
        """转发给底层 inner"""
        return self._inner.set(prop_id, value)

def open_jetson_csi_capture(sensor_id=0,width = 640, height = 480,fps = 30):
    if not shutil.which("gst-launch-1.0"):
        raise RuntimeError(
            "gst-launch-1.0 未找到，请确认容器中已安装 GStreamer。"
        )

    import time, select, os

    cap = GstPipeCapture(sensor_id,width,height,fps)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if cap._proc.poll() is not None:
            break

        fd = cap._proc.stdout.fileno()
        try:
            ready, _, _ = select.select([fd],[],[],0.2)
        except (ValueError,OSError):
            ready = False

        if ready:
            ok,frame = cap.read()
            if ok and frame is not None:
                return _PrefetchCapture(cap,frame),"gst-launch-pipe"
            break

        time.sleep(0.1)
    cap.release()
    raise RuntimeError(
        "无法打开 CSI 摄像头。\n"
        "请确认：\n"
        "  1. CSI 摄像头已连接且在容器内 gst-launch 能正常工作\n"
        "  2. Docker 启动时挂载了 /tmp（含 argus_socket）\n"
        "  3. 宿主机 nvargus-daemon 正在运行\n"
        "调试命令（在容器内执行）：\n"
        f"  gst-launch-1.0 nvarguscamerasrc sensor-id={sensor_id} ! "
        f"'video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1' ! nvvidconv ! fakesink"
    )





def camera_worker(frame_q, stop_event):
    print("摄像头正在开启")
    cap, _ = open_jetson_csi_capture()
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_q.full():
            try:
                frame_q.get_nowait()
            except Exception:
                pass
        frame_q.put(frame)

    cap.release()
    print("[Camera] 摄像头已释放")