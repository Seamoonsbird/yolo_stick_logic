"""
Jetson CSI 摄像头打开工具。
================================================================================

适用场景：
    NVIDIA Jetson 系列开发板（Nano/TX2/Xavier/Orin 等）上的 MIPI CSI 摄像头。

背景知识（为什么不用 cv2.VideoCapture 直接打开）：
    - OpenCV 的 GStreamer 后端需要编译时开启对应选项，预编译的 opencv-python
      通常不包含 GStreamer 支持，因此在 Jetson 上 cv2.VideoCapture 无法直接
      打开 CSI 摄像头。
    - 本模块绕开 OpenCV 的 GStreamer 依赖：通过外部子进程运行 GStreamer 管线，
      将解码后的 BGR 原始数据写入标准输出（stdout），Python 端从管道中读取
      字节流并用 numpy 重构为图像数组。

核心思路（三步走）：
    1. 子进程 —— gst-launch-1.0 启动一条 GStreamer pipeline，
       从 CSI 摄像头取流 → 硬解码 → 色彩转换 → BGR 裸数据 → stdout（fdsink）
    2. 读取 —— Python 通过 subprocess.PIPE 从子进程 stdout 逐帧读取
       width × height × 3 字节，用 np.frombuffer 转为 numpy 数组
    3. 封装 —— 接口刻意模仿 cv2.VideoCapture（isOpened / read / release），
       方便在现有 OpenCV 代码中替换使用
"""
import shutil
import subprocess

import cv2
import numpy as np


class GstPipeCapture:
    """
    通过 gst-launch-1.0 子进程读取 CSI 摄像头的 BGR 原始帧。

    ----------------------------------------------------------------------------
    工作原理（分步详解）:
    1. __init__ 中拼接一条 GStreamer 管线命令，并用 subprocess.Popen 启动它。
       管线各环节说明见 __init__ 的注释。
    2. read() 从子进程的 stdout 中读取固定字节数（= 一帧 BGR 数据的大小），
       然后用 numpy 重构为 H×W×3 的 uint8 数组，返回给调用方。
    3. isOpened() 通过检查子进程是否仍在运行来判断摄像头是否可用。
    4. release() 向子进程发送 SIGTERM，若超时未退出则 SIGKILL 强杀。

    ----------------------------------------------------------------------------
    接口兼容性:
        为了无缝替换 cv2.VideoCapture，本类实现了：
        - isOpened()  →  等同于 cv2.VideoCapture.isOpened()
        - read()      →  等同于 cv2.VideoCapture.read()，返回 (ret, frame)
        - release()   →  等同于 cv2.VideoCapture.release()
        - get(prop)   →  等同于 cv2.VideoCapture.get()，仅支持宽高属性
        - set(prop, v)→  空实现，不支持运行时修改参数

    ----------------------------------------------------------------------------
    注意事项:
        - 网络断开或摄像头故障时，read() 会读到空字节，此时返回 (False, None)
        - 本类不处理帧率控制，调用方需自行控制读取频率
        - bufsize 设为 2 帧大小，防止管道缓冲区溢出导致子进程阻塞
    """

    def __init__(self, sensor_id=0, width=640, height=480, fps=30):
        # ---- 基本参数 ----
        self.width = width
        self.height = height
        # 一帧 BGR 数据的总字节数：每个像素占 3 个通道（B、G、R），每通道 1 字节
        self.frame_bytes = width * height * 3

        # ---- 拼接 GStreamer 管线命令 ----
        # 下面这条命令等价于在终端执行：
        #   gst-launch-1.0 --quiet -e \
        #       nvarguscamerasrc sensor-id=0 ! \
        #       'video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1' ! \
        #       nvvidconv ! \
        #       'video/x-raw,width=640,height=480,format=BGRx' ! \
        #       videoconvert ! \
        #       'video/x-raw,format=BGR' ! \
        #       fdsink
        #
        # 管线各环节说明（从左到右，! 是 GStreamer 的管道符号）：
        #
        #   [1] nvarguscamerasrc
        #       NVIDIA 的 CSI 摄像头源插件。
        #       sensor-id: 摄像头编号，Jetson 上第一个 CSI 口通常是 0
        #
        #   [2] 'video/x-raw(memory:NVMM),width=...,height=...,framerate=...'
        #       Caps 过滤器（capabilities filter）。
        #       memory:NVMM 表示数据存储在 NVIDIA 统一内存中，
        #       这是后续 nvvidconv 硬件加速转换的前提要求。
        #
        #   [3] nvvidconv
        #       NVIDIA 硬件视频转换器。
        #       将 NVMM 内存中的原始帧转换为 CPU 可访问的常规内存格式。
        #
        #   [4] 'video/x-raw,width=...,height=...,format=BGRx'
        #       指定 nvvidconv 输出格式为 BGRx。
        #       BGRx = B-G-R-填充字节，每像素 4 字节（32 位对齐），
        #       硬件转换通常要求对齐宽度，所以用 BGRx 而非直接 BGR。
        #
        #   [5] videoconvert
        #       GStreamer 软件色彩空间转换插件。
        #       将 BGRx（4 字节/像素）转为 BGR（3 字节/像素），
        #       去掉多余的填充字节，减小数据量。
        #
        #   [6] 'video/x-raw,format=BGR'
        #       指定 videoconvert 输出为纯 BGR 格式。
        #       这是 OpenCV numpy 数组直接使用的格式（不包含 alpha 通道）。
        #
        #   [7] fdsink
        #       将管线最终输出写入文件描述符。
        #       默认写入 stdout（标准输出，fd=1），
        #       Python 端通过 subprocess.PIPE 读取。
        #
        # 命令行参数说明:
        #   --quiet : 禁止 gst-launch 打印 INFO/WARNING 到 stderr，避免干扰
        #   -e      : 收到 EOS（End Of Stream）信号后强制关闭管线，确保子进程退出
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

        # ---- 启动子进程 ----
        # shell=True:
        #   让系统 shell 来解释执行整条命令字符串。
        #   好处是 GStreamer 的 ! 管道符号可以直接写在字符串中，无需拆分为列表。
        #   （Windows 上可能有安全风险，但 Jetson 只跑 Linux，这里安全可控）
        #
        # stdout=PIPE:
        #   子进程的 stdout 被重定向到一个管道（pipe），
        #   父进程（Python）通过 self._proc.stdout 读取。
        #
        # stderr=DEVNULL:
        #   子进程的 stderr 丢弃。配合 --quiet 参数，避免 GStreamer 日志干扰。
        #   调试时可以临时改为 stderr=subprocess.PIPE 来查看管线报错信息。
        #
        # bufsize=frame_bytes * 2:
        #   管道内核缓冲区大小设为 2 帧数据量。
        #   如果 Python 读取速度跟不上摄像头帧率，缓冲区能暂存最多 2 帧，
        #   超出后子进程的 write 会阻塞，间接实现背压（back pressure）。
        self._proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_bytes * 2,
        )

    def isOpened(self):
        """
        判断摄像头是否仍然可用。

        通过 poll() 检查子进程是否还在运行：
        - poll() 返回 None → 子进程仍在运行 → 摄像头可用
        - poll() 返回退出码 → 子进程已退出 → 摄像头断开或发生错误
        """
        return self._proc.poll() is None

    def read(self):
        """
        从子进程 stdout 读取一帧 BGR 图像。

        读取流程：
        1. 循环读取管道数据，直到凑满 frame_bytes 字节。
           为什么要循环？因为管道是流式的，一次 read 不一定能返回完整一帧，
           可能被 OS 拆成多个 chunk 返回。
        2. 用 np.frombuffer 将原始字节解释为 uint8 一维数组。
        3. 用 reshape 将一维数组重组为 (height, width, 3) 的三维数组。
           注意：每个像素 3 个通道的顺序是 B-G-R（不是 RGB），
           这是 OpenCV 的默认色彩顺序。
        4. 调用 .copy() 创建副本返回，避免后续覆盖缓冲区数据。

        返回值：
            (True, frame)  —— 成功读取一帧，frame 是 numpy.ndarray, shape=(H,W,3)
            (False, None)  —— 管道关闭、EOF 或摄像头断开

        注意：
            本方法不会超时阻塞无限等待；如果管道断开，read 会立刻返回空 bytes，
            此时返回 (False, None)。
        """
        raw = b""
        # 循环读取直到凑满一帧的字节数
        while len(raw) < self.frame_bytes:
            # 每次读取还差多少字节就请求多少字节
            chunk = self._proc.stdout.read(self.frame_bytes - len(raw))
            if not chunk:
                # 管道返回空字节 → 子进程已关闭 stdout 或崩溃
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

    def release(self):
        """
        释放摄像头资源、终止子进程。

        先尝试优雅退出（SIGTERM），给子进程 5 秒时间自行清理；
        超时未退出则强制杀死（SIGKILL）。
        捕获所有异常（比如子进程已自行退出导致的 wait 异常），
        确保 release 不会抛出异常。
        """
        try:
            # terminate() = 发送 SIGTERM（信号 15），子进程可以捕获并做清理
            self._proc.terminate()
            # 等待最多 5 秒，让子进程完成清理
            self._proc.wait(timeout=5)
        except Exception:
            # 超时或进程已不存在 → 发送 SIGKILL（信号 9），操作系统强制终止
            self._proc.kill()

    def get(self, prop_id):
        """
        获取摄像头属性（兼容 cv2.VideoCapture.get）。

        当前只支持查询帧宽度和帧高度，其他属性一律返回 0.0。
        这是因为 GStreamer 管线创建时已固定分辨率，无法动态获取其他属性。

        参数:
            prop_id: cv2 属性常量，例如 cv2.CAP_PROP_FRAME_WIDTH (=3)

        返回值:
            float 类型的属性值
        """
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def set(self, prop_id, value):
        """
        设置摄像头属性（空实现，兼容 cv2.VideoCapture.set）。

        GStreamer 管线创建后无法在运行时修改分辨率等参数，
        所以此方法为空操作。如需不同参数，请创建新的 GstPipeCapture 实例。
        """
        pass


class _PrefetchCapture:
    """
    预读帧包装器（内部类，下划线前缀表示模块私有）。

    ----------------------------------------------------------------------------
    为什么需要这个类？
        在 open_jetson_csi_capture() 中，为了验证摄像头能正常工作，
        会预先读取一帧（预读/探针帧）。如果直接丢弃这一帧太浪费，
        所以用本类把它缓存起来，后续第一次 read() 调用时优先返回它。

    ----------------------------------------------------------------------------
    工作流程：
        1. 创建时接收一个 inner（真正的 GstPipeCapture）和 first_frame（预读帧）
        2. 第一次调用 read() 时返回缓存的 first_frame，然后清空缓存
        3. 后续 read() 调用直接转发给 inner（GstPipeCapture）

    ----------------------------------------------------------------------------
    类比：
        类似 Python 迭代器中的 peek（窥视）操作，
        先看一眼下一个元素但不消费，确认可用后再真正取出。
    """

    def __init__(self, inner, first_frame):
        """
        参数:
            inner:       底层的视频捕获对象（如 GstPipeCapture 实例）
            first_frame: 预先读取的第一帧 numpy 数组
        """
        self._inner = inner
        self._first = first_frame  # 缓存预读帧，非 None 表示还未被消费

    def isOpened(self):
        """
        判断摄像头是否可用。
        如果预读帧还在（未被消费），说明摄像头至少成功捕获过一次，返回 True。
        如果 pre-read frame 已被消费，则委托给底层 inner 判断。
        """
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


def open_jetson_csi_capture(sensor_id=0, width=640, height=480, fps=30):
    """
    打开 Jetson CSI 摄像头的顶层入口函数（推荐使用此函数而非直接实例化类）。

    ----------------------------------------------------------------------------
    返回值:
        (capture_object, method_name)
        - capture_object: 兼容 cv2.VideoCapture 接口的对象
          （实际类型是 _PrefetchCapture，包装了 GstPipeCapture）
        - method_name:   字符串 "gst-launch-pipe"，
          用于标识当前使用的摄像头打开方式，方便上层按来源走不同分支

    ----------------------------------------------------------------------------
    打开流程（带超时保护）：
        1. 检查 gst-launch-1.0 是否可用
        2. 创建 GstPipeCapture 并启动 GStreamer 子进程
        3. 使用 select 监听子进程 stdout，等待首帧数据到达（最长 5 秒）
        4. 若在超时前成功读到一帧 → 用 _PrefetchCapture 包装后返回
        5. 若超时或子进程退出 → 释放资源，抛出 RuntimeError

    ----------------------------------------------------------------------------
    超时机制说明（为什么用 select 而不是直接 read）：
        直接 read() 是阻塞调用，如果 GStreamer 管线启动失败（比如摄像头
        未连接），read 会永远阻塞。select 是操作系统级别的 I/O 多路复用，
        可以「等管道有数据可读再动手」，配合超时参数实现轮询等待。

    ----------------------------------------------------------------------------
    参数:
        sensor_id: CSI 摄像头编号，默认 0（第一个 CSI 接口）
        width:     期望的图像宽度（像素）
        height:    期望的图像高度（像素）
        fps:       期望的帧率（每秒帧数）

    返回值:
        (capture, "gst-launch-pipe")  —— 成功
        抛出 RuntimeError             —— 失败（设备不可用或超时）

    ----------------------------------------------------------------------------
    使用示例:
        cap, method = open_jetson_csi_capture(sensor_id=0, width=1280, height=720, fps=30)
        print(f"使用 {method} 方式打开了摄像头")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    ----------------------------------------------------------------------------
    常见故障排查（对应错误信息中的提示）:
        1. "gst-launch-1.0 未找到"
           → 容器/系统中没有安装 GStreamer，请 apt install gstreamer1.0-tools

        2. "CSI 摄像头已连接且在容器内 gst-launch 能正常工作"
           → 检查摄像头排线是否松动，在终端手动运行调试命令验证

        3. "Docker 启动时挂载了 /tmp（含 argus_socket）"
           → NVIDIA Argus 守护进程通过 /tmp/argus_socket 提供摄像头服务，
              Docker 容器需要挂载宿主机 /tmp 才能访问该 socket

        4. "宿主机 nvargus-daemon 正在运行"
           → 在 Jetson 宿主机上执行 systemctl status nvargus-daemon 确认
    """
    # ---- 步骤 1：检查 gst-launch-1.0 是否可执行 ----
    # shutil.which() 等价于 Linux 的 which 命令，
    # 在 PATH 环境变量中查找可执行文件，找不到返回 None
    if not shutil.which("gst-launch-1.0"):
        raise RuntimeError(
            "gst-launch-1.0 未找到，请确认容器中已安装 GStreamer。"
        )

    # 延迟导入：这些模块只在打开摄像头时需要，放在文件顶部导入也不浪费，
    # 但这里保留函数内导入是原作者的习惯
    import time, select, os

    # ---- 步骤 2：创建 GstPipeCapture，启动 GStreamer 子进程 ----
    cap = GstPipeCapture(sensor_id, width, height, fps)

    # ---- 步骤 3：轮询等待首帧数据（最长等 5 秒） ----
    # time.monotonic() 使用系统单调时钟，不受系统时间调整影响。
    # 区别于 time.time()（可能因 NTP 校时跳变），monotonic 只增不减，
    # 适合计算超时时间。
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        # 如果子进程已退出（poll() 返回非 None），说明 GStreamer 管线启动失败
        if cap._proc.poll() is not None:
            break

        # ---- 步骤 3a：用 select 检查管道是否有数据可读 ----
        # fileno() 获取管道读端的文件描述符（一个整数 fd），
        # select 可以监听这个 fd 是否「可读」（有数据到达）
        fd = cap._proc.stdout.fileno()
        try:
            # select.select(read_fds, write_fds, error_fds, timeout)
            # 这里只关心可读，所以后两个列表为空
            # timeout=0.2 秒：每 200ms 检查一次，避免空转 CPU
            ready, _, _ = select.select([fd], [], [], 0.2)
        except (ValueError, OSError):
            # fd 可能已关闭（子进程崩溃），select 会抛异常
            ready = False

        # ---- 步骤 3b：如果有数据可读，尝试读取一帧 ----
        if ready:
            ok, frame = cap.read()
            if ok and frame is not None:
                # 成功读到首帧！用 _PrefetchCapture 包装，
                # 把这一帧缓存起来，后续调用方第一次 read() 时直接返回
                return _PrefetchCapture(cap, frame), "gst-launch-pipe"
            # 读到空数据或 EOF，退出循环，走失败处理
            break

        # 还没数据，睡 100ms 再试（配合 select 的 200ms 超时，
        # 实际上外层循环每次迭代最多等 200ms）
        time.sleep(0.1)

    # ---- 步骤 4：超时或启动失败，释放资源并报错 ----
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
        # 调试命令末尾用的 fakesink（而非 fdsink）：
        # fakesink 会吞掉所有数据，仅用于测试管线能否正常启动。
        # 如果这条命令能跑通说明摄像头和 GStreamer 都没问题，
        # 问题可能出在 Python 端的管道读取逻辑。
    )
