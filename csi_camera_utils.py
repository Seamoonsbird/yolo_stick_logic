"""
Jetson CSI 摄像头打开工具。

通过 gst-launch-1.0 子进程 + fdsink 把原始 BGR 帧写到 stdout，
Python 端用 numpy 解析。不依赖 OpenCV 的 GStreamer 编译选项。
"""
import shutil
import subprocess

import cv2
import numpy as np


class GstPipeCapture:
    """
    用 gst-launch-1.0 子进程读取 CSI 摄像头的 BGR 原始帧。
    接口兼容 cv2.VideoCapture (isOpened / read / release)。
    """

    def __init__(self, sensor_id=0, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3

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

    def isOpened(self):
        return self._proc.poll() is None

    def read(self):
        raw = b""
        while len(raw) < self.frame_bytes:
            chunk = self._proc.stdout.read(self.frame_bytes - len(raw))
            if not chunk:
                return False, None
            raw += chunk
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        return True, frame.copy()

    def release(self):
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def set(self, prop_id, value):
        pass


class _PrefetchCapture:
    """包装器：保存预读的第一帧，后续 read() 先返回它。"""

    def __init__(self, inner, first_frame):
        self._inner = inner
        self._first = first_frame

    def isOpened(self):
        return self._first is not None or self._inner.isOpened()

    def read(self):
        if self._first is not None:
            frame = self._first
            self._first = None
            return True, frame
        return self._inner.read()

    def release(self):
        self._inner.release()

    def get(self, prop_id):
        return self._inner.get(prop_id)

    def set(self, prop_id, value):
        return self._inner.set(prop_id, value)


def open_jetson_csi_capture(sensor_id=0, width=640, height=480, fps=30):
    """
    返回 (capture_object, method_name)。
    capture_object 兼容 cv2.VideoCapture 的 isOpened/read/release 接口。
    """
    if not shutil.which("gst-launch-1.0"):
        raise RuntimeError(
            "gst-launch-1.0 未找到，请确认容器中已安装 GStreamer。"
        )

    import time, select, os

    cap = GstPipeCapture(sensor_id, width, height, fps)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if cap._proc.poll() is not None:
            break
        fd = cap._proc.stdout.fileno()
        try:
            ready, _, _ = select.select([fd], [], [], 0.2)
        except (ValueError, OSError):
            ready = False
        if ready:
            ok, frame = cap.read()
            if ok and frame is not None:
                return _PrefetchCapture(cap, frame), "gst-launch-pipe"
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
