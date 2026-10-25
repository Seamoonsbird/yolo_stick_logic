"""
录制进程。

从 recorder_q 接收待录制的帧，从 state_q 接收状态信息（叠加到画面），
写入视频文件。

变更:
  - PIL 渲染中文（替换 OpenCV putText，解决中文问号问题）
  - 文件名带时间戳，不覆盖旧视频
  - 显示最近告警文案
  - VideoWriter FPS 根据头两帧到达间隔动态计算
  - 提示音按事件时间戳混入视频（nodes/video_audio.py + ffmpeg），音画同源不漂移
"""

from __future__ import annotations

import queue
import time
import datetime
import os
import signal

import cv2
import numpy as np

from config import (
    RECORDER_CODEC,
    RECORDER_WIDTH,
    RECORDER_HEIGHT,
    MUX_ALERT_AUDIO,
    AUDIO_MIX_RATE,
    get_output_dir,
)

from nodes import video_audio

# ==========================================
# 中文字体检测
# ==========================================

# 按优先级排列的 CJK 字体路径（Ubuntu / Jetson 常见位置）
_CJK_FONT_PATHS = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
]

# 固定路径都没命中时，再在这些目录里按文件名关键字扫一遍。
# 好处：容器里 `docker cp` 任意中文 ttf/ttc 进去（哪怕放错目录）也能被自动找到。
_CJK_FONT_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/root/.fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
]

# 文件名里出现这些关键字即认为是中文字体（小写匹配）
_CJK_FONT_KEYWORDS = (
    "wqy", "zenhei", "microhei", "cjk", "notosanssc", "notoserifsc",
    "sourcehansans", "sourcehanserif", "droidsansfallback",
    "msyh", "msjh", "simhei", "simsun", "msung", "uming", "ukai", "arphic",
)

_CJK_FONT_EXTS = (".ttf", ".ttc", ".otf", ".otc")

# PIL 字体缓存
_pil_fonts: dict[int, object] = {}  # size -> ImageFont
_pil_font_path: str | None = None


def _scan_cjk_font() -> str | None:
    """在字体目录里按关键字搜索中文字体（固定路径未命中时使用）。"""
    for root_dir in _CJK_FONT_DIRS:
        if not os.path.isdir(root_dir):
            continue
        for cur, _dirs, files in os.walk(root_dir):
            for name in sorted(files):
                lower = name.lower()
                if not lower.endswith(_CJK_FONT_EXTS):
                    continue
                if any(k in lower for k in _CJK_FONT_KEYWORDS):
                    return os.path.join(cur, name)
    return None


def _find_cjk_font() -> str | None:
    """查找第一个可用的 CJK 字体：先查固定路径，再扫目录。"""
    for path in _CJK_FONT_PATHS:
        if os.path.isfile(path):
            return path
    return _scan_cjk_font()


def _get_font(size: int):
    """获取指定大小的 PIL 字体（带缓存）。"""
    global _pil_font_path
    if size in _pil_fonts:
        return _pil_fonts[size]

    if _pil_font_path is None:
        _pil_font_path = _find_cjk_font()

    try:
        from PIL import ImageFont
        if _pil_font_path:
            font = ImageFont.truetype(_pil_font_path, size)
        else:
            font = ImageFont.load_default()
        _pil_fonts[size] = font
        return font
    except Exception:
        from PIL import ImageFont
        font = ImageFont.load_default()
        _pil_fonts[size] = font
        return font


# ==========================================
# PIL 中文叠加
# ==========================================

def _put_text_pil(frame, text: str, xy: tuple[int, int], *,
                  font_size: int = 20, color: tuple[int, int, int] = (0, 255, 0),
                  anchor: str = "lt"):
    """
    用 PIL 在 OpenCV BGR 帧上绘制文字（支持中文）。
    anchor: "lt"=左上, "rb"=右下, "rt"=右上, "lb"=左下
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # PIL 不可用 → 降级为 OpenCV putText（只能 ASCII）
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
        return

    font = _get_font(font_size)

    # OpenCV BGR → PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 获取文字尺寸用于定位
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x, y = xy
    if anchor == "rb":
        x, y = x - tw, y - th
    elif anchor == "rt":
        x, y = x - tw, y
    elif anchor == "lb":
        x, y = x, y - th

    draw.text((x, y), text, font=font, fill=color[::-1])  # BGR → RGB

    # 写回 OpenCV BGR
    rgb = np.array(pil_img)
    frame[:] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ==========================================
# VideoWriter
# ==========================================

def _build_video_writer(fps: float):
    """
    创建 cv2.VideoWriter，文件名带时间戳避免覆盖。

    直接写正式文件名 record_时间戳.mp4（不再有 .silent 临时名）：
      进程被强杀时，留下的就是和「无声方案」完全一样的 record_*.mp4；
      混音成功后才生成 *_mux.mp4 并原子替换，失败不影响这个文件。
    """
    out_dir = get_output_dir()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"{out_dir}/record_{timestamp}.mp4"

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


# ==========================================
# 提示音混入录像
# ==========================================

def _finalize_audio(video_path: str, events: list,
                    t_start: float, frame_count: int, real_fps: float):
    """
    录制结束后把播报事件铺成音轨、混进视频。

    events: [{"t": 绝对时间戳, "wav": 路径, "text": 文案}, ...]

    安全策略：视频已经在最终文件名上、并且已经 release（有 moov、可正常播放）。
    只有混流成功才用 os.replace 原子替换；任何一步失败都保留无声版，画面绝不丢。
    """
    if not MUX_ALERT_AUDIO:
        return

    duration = frame_count / real_fps if real_fps > 0 else 0.0

    # 绝对时间戳 → 相对视频首帧的偏移（首帧之前的告警夹到 0，即开场就响）
    timed = []
    for ev in events:
        try:
            offset = max(0.0, float(ev.get("t", 0.0)) - t_start)
        except Exception:
            continue
        if duration > 0 and offset > duration:
            continue        # 视频已经结束之后的提醒，放不进画面
        timed.append((offset, ev.get("wav")))

    if not timed:
        print("[Recorder] 本次录制没有语音提示，视频保持无声")
        return

    base = os.path.splitext(video_path)[0]
    mix_wav = base + ".mix.wav"
    mux_out = base + "_mux.mp4"

    if not video_audio.build_wav(mix_wav, timed, duration, AUDIO_MIX_RATE):
        print("[Recorder] 音轨生成失败，视频保持无声")
        return

    if video_audio.mux_audio(video_path, mix_wav, mux_out):
        try:
            os.replace(mux_out, video_path)      # 成功才覆盖无声版
        except OSError as e:
            print(f"[Recorder] ⚠ 替换视频失败，保留无声版；带声音的文件在: {mux_out} ({e})")
            return
        try:
            os.remove(mix_wav)
        except OSError:
            pass
        print(f"[Recorder] 已混入 {len(timed)} 条语音提示 → {video_path}")
        return

    # 混流失败（没装 ffmpeg / 编码器不支持…）：无声视频照旧可用，音轨留着手动合成
    print("[Recorder] ⚠ 混流失败，视频保持无声；音频已单独保存：", mix_wav)
    print(f"[Recorder]   手动合成: ffmpeg -i {os.path.basename(video_path)} "
          f"-i {os.path.basename(mix_wav)} -c:v copy -c:a aac out.mp4")


def _install_worker_signals(stop_event):
    """
    录制子进程的信号策略，专门保护 mp4 收尾：

      - 忽略 SIGINT：终端的 Ctrl+C 会发给整个进程组，若在 writer.release()
        （写 moov 索引）时被打断，就会留下「有画面数据但没有索引」的 mp4，
        任何播放器都渲染失败。Ctrl+C 统一交给主进程处理。
      - SIGTERM → 置停止事件，走正常收尾（被 kill/容器停止时也能落盘）。
    """
    def _on_term(signum, frame):
        stop_event.set()

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, _on_term)
    except (ValueError, OSError, AttributeError):
        pass


# ==========================================
# 画面叠加
# ==========================================

def _draw_overlay(frame, state: dict, frame_count: int, real_fps: float):
    """
    在帧画面上叠加信息（PIL 渲染，全中文支持）。

    左上角: 时间戳 + 帧序号 + 实时 FPS
    右上角: 录制指示灯
    左下角: 最近告警文案（高亮）
    右下角: 检测状态
    """
    h, w = frame.shape[:2]

    # ---- 左上角：时间戳 + 帧序号 + FPS ----
    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    _put_text_pil(frame, f"{ts}  |  frame #{frame_count}  |  {real_fps:.1f} fps",
                  (10, 5), font_size=18, color=(0, 255, 0))

    # ---- 右上角：录制指示灯 ----
    _put_text_pil(frame, "● REC", (w - 10, 5),
                  font_size=22, color=(0, 0, 255), anchor="rt")

    # ---- 右下角：检测状态 ----
    y_bottom = h - 5
    # 先画检测状态（从下往上排列）
    status_keys = [k for k in state.keys() if k not in ("最近提醒",)]
    # 按固定顺序排列：帧、盲道、楼梯、水坑、井盖、草地
    priority_order = ["帧", "盲道", "上楼梯", "下楼梯", "水坑", "井盖", "草地", "斑马线", "木板"]
    ordered = [k for k in priority_order if k in state] + \
              [k for k in status_keys if k not in priority_order]

    for key in reversed(ordered):
        value = state[key]
        text = f"{key}: {value}" if value else key
        _put_text_pil(frame, text, (w - 10, y_bottom),
                      font_size=18, color=(0, 255, 0), anchor="rb")
        y_bottom -= 22

    # ---- 左下角：最近提醒（黄色醒目） ----
    alerts = state.get("最近提醒", [])
    if alerts:
        y_alert = h - 5
        for alert_text in reversed(alerts[-3:]):  # 最多显示最近 3 条
            _put_text_pil(frame, f"⚠ {alert_text}", (10, y_alert),
                          font_size=20, color=(0, 255, 255), anchor="lb")
            y_alert -= 26

    return frame


# ==========================================
# 主循环
# ==========================================

def recorder_worker(recorder_q, stop_event, state_q, audio_q=None):
    """
    录制主循环。

    工作流程:
        1. 等待第一帧到达
        2. 等待第二帧到达 → 根据时间间隔计算真实 FPS
        3. 以真实 FPS 创建 VideoWriter，写入缓冲的 2 帧
        4. 循环取帧 → 叠加信息 → 写入视频；同时收集播报事件（audio_q）
        5. stop_event 触发后清空残帧、释放资源，最后把提示音混进视频

    注意：本进程忽略 SIGINT，Ctrl+C 只由主进程接管。否则在写 mp4 索引
    （writer.release）时被打断，会留下有画面数据但没有 moov 的残缺文件。
    """
    _install_worker_signals(stop_event)

    print("[Recorder] 正在等待首帧以测量真实帧率...")

    if MUX_ALERT_AUDIO:
        if video_audio.FFMPEG_BIN:
            print(f"[Recorder] 提示音混入录像: 开 (ffmpeg: {video_audio.FFMPEG_BIN})")
        else:
            print("[Recorder] ⚠ 提示音混入录像: 开，但未找到 ffmpeg，"
                  "视频先按无声保存，音轨另存 .wav")
            print("[Recorder]    安装: apt-get install -y ffmpeg")

    # 检测中文字体
    font_path = _find_cjk_font()
    if font_path:
        print(f"[Recorder] 中文字体: {font_path}")
    else:
        print("[Recorder] ⚠ 未找到中文字体，中文将显示成方块。容器内执行：")
        print("[Recorder]     apt-get update && apt-get install -y fonts-wqy-zenhei")
        print("[Recorder]   或者从宿主机拷一个中文字体进去（二选一）：")
        print("[Recorder]     docker cp /path/to/msyh.ttc <容器名>:/usr/share/fonts/truetype/")
        print("[Recorder]     docker exec <容器名> fc-cache -fv")

    current_state: dict = {}
    frame_count = 0
    audio_events: list = []      # [{"t": 绝对时间戳, "wav": 路径, "text": 文案}, ...]

    def _drain_audio_events():
        """把逻辑进程发来的播报事件收进本地列表（不阻塞）。"""
        if audio_q is None:
            return
        while True:
            try:
                audio_events.append(audio_q.get_nowait())
            except queue.Empty:
                break
            except Exception:
                break

    # ---- 阶段 1：缓冲开头两帧，测量真实 FPS ----
    buffered_frames = []
    t_first = 0.0
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
        delta = 0.2
    real_fps = 1.0 / delta
    real_fps = max(1.0, min(real_fps, 60.0))
    print(f"[Recorder] 实测帧率: {real_fps:.1f} fps (帧间隔: {delta*1000:.0f}ms)")

    # ---- 阶段 2：创建 VideoWriter（文件名带时间戳） ----
    writer, filepath = _build_video_writer(real_fps)
    print(f"[Recorder] 视频文件已创建: {filepath}")

    # 写入缓冲的两帧
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

            # 顺手收走这期间发生的播报事件（录像音轨用）
            _drain_audio_events()

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
        _drain_audio_events()   # 收尾阶段最后几条提醒也要进音轨

    finally:
        # release() 会写入 mp4 的 moov 索引：没有它文件根本打不开。
        # 用 try 包住，避免 release 抛错导致后面的混音步骤被跳过。
        try:
            writer.release()
        except Exception as e:
            print(f"[Recorder] ⚠ 关闭视频文件时出错（视频可能不完整）: {e}")
        print(f"[Recorder] 视频已保存: {filepath}  "
              f"(共 {frame_count} 帧, {real_fps:.1f} fps)")

        # ---- 阶段 5：把提示音混进视频（失败不影响上面这个无声视频） ----
        try:
            _finalize_audio(filepath, audio_events,
                            t_first, frame_count, real_fps)
        except Exception as e:
            print(f"[Recorder] ⚠ 混入提示音失败（视频仍是无声版）: {e}")
