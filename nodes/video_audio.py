"""
把告警提示音混入录制视频。

链路:
  1. logic.speak() 决定播哪条 WAV 时，把 {"t": 时间戳, "wav": 路径, "text": 文案}
     塞进 audio_q（见 nodes/logic.py 的 set_audio_event_queue / _emit_audio_event）。
  2. recorder 进程记下视频首帧时刻 t_start，录制结束时算出每条提示相对视频的偏移。
  3. build_wav() 按偏移把 WAV 叠成一条与视频等长的单声道音轨（重叠部分相加）。
  4. mux_audio() 用 ffmpeg 复制视频流、把音轨编码成 AAC，得到带声音的 mp4。

为什么不去录系统声卡（parec / PulseAudio monitor）:
  paplay 走的是宿主机默认输出，容器里录 monitor 源要额外权限，而且音频、视频两条
  独立时间线会漂移；按事件时间重建音轨，音画对齐由同一套 time.time() 保证。

依赖: numpy（已有）、ffmpeg（缺失时降级：视频照录，音轨单独存成 .wav）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave

import numpy as np

__all__ = ["FFMPEG_BIN", "build_wav", "mux_audio"]

# 启动时探测一次，避免每条告警都 which
FFMPEG_BIN = shutil.which("ffmpeg")

# 生成音轨时的分块长度（秒）。内存只跟它有关，与录像时长无关。
_BLOCK_SEC = 5.0


# ==========================================
# WAV 读取 / 重采样
# ==========================================

def _decode_pcm(raw: bytes, sw: int, ch: int):
    """
    PCM 字节 → float32 单声道 [-1, 1]。支持 8/16/24/32-bit，多声道取平均。
    不支持时返回 None。
    """
    if sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= (1 << 23), v - (1 << 24), v)
        data = v.astype(np.float32) / float(1 << 23)
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return None

    if ch > 1 and data.size % ch == 0:
        data = data.reshape(-1, ch).mean(axis=1)

    return np.ascontiguousarray(data, dtype=np.float32)


def _read_mono(path: str):
    """
    读整个 WAV → (float32 单声道数据, 采样率)。失败返回 (None, 0)。
    """
    try:
        with wave.open(path, "rb") as w:
            ch = w.getnchannels()
            sw = w.getsampwidth()
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
    except Exception as e:
        print(f"[video_audio] ⚠️ 读取 WAV 失败: {path} ({e})")
        return None, 0

    data = _decode_pcm(raw, sw, ch)
    if data is None:
        print(f"[video_audio] ⚠️ 不支持的位深 {sw} 字节: {path}")
        return None, 0

    return data, sr


def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """线性插值重采样（语音足够用，不需要高质量重采样器）。"""
    if sr_in == sr_out or data.size == 0:
        return data
    n_out = max(1, int(round(data.size * sr_out / float(sr_in))))
    x_old = np.linspace(0.0, 1.0, data.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)


# ==========================================
# 混音
# ==========================================

def build_wav(out_path: str, events, duration: float, rate: int = 22050) -> bool:
    """
    按事件把提示音铺成一条与视频等长的音轨。

    参数:
      out_path : 输出 WAV 路径
      events   : [(offset_sec, wav_path), ...]  offset 相对视频首帧
      duration : 视频时长（秒）
      rate     : 输出采样率
    返回:
      True  → 至少混入一条，文件已写出
      False → 没有可用音频（文件不存在 / 全被裁掉），未写出文件

    实现要点（内存）:
      按 _BLOCK_SEC 秒一块流式生成并写盘，内存占用只与块大小有关，与视频
      时长无关。老写法一次性 np.zeros(duration*rate)，录 8 小时要 2.5GB，
      Jetson 上必 OOM。
    """
    total = max(1, int(round(max(0.0, duration) * rate)))

    # 先只读 WAV 头，算出每条在输出时间轴上的区间（不加载音频数据）
    entries = []
    for off, wav in events:
        if not wav or not os.path.isfile(wav):
            print(f"[video_audio] ⚠️ 提示音文件已不存在，跳过: {wav}")
            continue
        try:
            with wave.open(wav, "rb") as w:
                ch, sw, sr, nframes = (w.getnchannels(), w.getsampwidth(),
                                       w.getframerate(), w.getnframes())
        except Exception as e:
            print(f"[video_audio] ⚠️ 读取 WAV 失败: {wav} ({e})")
            continue
        if nframes <= 0 or sr <= 0 or sw not in (1, 2, 3, 4):
            print(f"[video_audio] ⚠️ 跳过不可用 WAV: {wav}")
            continue

        start = int(round(max(0.0, float(off)) * rate))
        if start >= total:
            continue                      # 提示发生在视频结束之后，丢弃
        span = int(round(nframes * rate / float(sr)))
        entries.append((start, span, wav, ch, sw, sr))

    if not entries:
        return False

    entries.sort(key=lambda e: e[0])

    block = max(1, int(rate * _BLOCK_SEC))
    placed = 0

    try:
        with wave.open(out_path, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)

            pos = 0
            while pos < total:
                n = min(block, total - pos)
                mix = np.zeros(n, dtype=np.float32)

                for start, span, wav, ch, sw, sr in entries:
                    end = start + span
                    if end <= pos:
                        continue
                    if start >= pos + n:
                        break                 # entries 按 start 排序，后面的更靠后

                    lo = max(start, pos)
                    hi = min(end, pos + n)

                    # 输出时间轴上的交集 → 源 WAV 的采样区间
                    src_from = int((lo - start) * sr / float(rate))
                    src_count = max(1, int(np.ceil((hi - lo) * sr / float(rate))))
                    try:
                        with wave.open(wav, "rb") as w:
                            nf = w.getnframes()
                            if nf <= 0:
                                continue
                            w.setpos(max(0, min(src_from, nf - 1)))
                            raw = w.readframes(src_count)
                    except Exception as e:
                        print(f"[video_audio] ⚠️ 读取音频片段失败: {wav} ({e})")
                        continue

                    data = _decode_pcm(raw, sw, ch)
                    if data is None or data.size == 0:
                        continue

                    data = _resample(data, sr, rate)
                    seg = data[: hi - lo]
                    mix[lo - pos: lo - pos + seg.size] += seg
                    placed += 1

                np.clip(mix, -1.0, 1.0, out=mix)      # 叠加可能超过 1.0，削顶保护
                out.writeframes((mix * 32767.0).astype("<i2").tobytes())
                pos += n
    except Exception as e:
        print(f"[video_audio] ⚠️ 写音轨失败: {out_path} ({e})")
        return False

    if placed == 0:
        try:
            os.remove(out_path)      # 内容全是静音，别留个空音轨
        except OSError:
            pass
        return False

    print(f"[video_audio] 音轨已生成: {len(entries)} 条提示, {duration:.1f}s, {rate}Hz")
    return True


# ==========================================
# 混流
# ==========================================

def mux_audio(video_path: str, audio_path: str, out_path: str, timeout: int = 600) -> bool:
    """
    把音轨混进视频：视频流直接复制（不重新编码，不损失画质），音频编码为 AAC。
    成功返回 True。ffmpeg 缺失或失败返回 False（不抛出）。
    """
    if not FFMPEG_BIN:
        return False

    cmd = [
        FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"[video_audio] ⚠️ ffmpeg 混流超时（>{timeout}s）")
        return False
    except Exception as e:
        print(f"[video_audio] ⚠️ ffmpeg 调用异常: {e}")
        return False

    if proc.returncode != 0:
        err = (proc.stderr or b"").decode(errors="ignore").strip()
        print(f"[video_audio] ⚠️ ffmpeg 混流失败 ({proc.returncode}): {err[-500:]}")
        return False

    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
