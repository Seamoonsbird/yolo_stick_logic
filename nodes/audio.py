"""
音频播报后端 — 盲杖智能辅助系统。

设计约束（来自部署要求）:
  - 不依赖任何需要编译的 Python 音频库（sounddevice / pyaudio / simpleaudio 等）。
  - 仅依赖系统 CLI: paplay / aplay（pulseaudio-utils / alsa-utils）+ espeak-ng。
  - 走系统默认音频输出（PulseAudio socket 或 ALSA 默认设备），不硬编码声卡号。
  - 任何失败（文件缺失 / 播放超时 / 命令不存在 / 其它异常）只打印警告，绝不抛出，避免主程序崩溃。

对外接口:
  - play_alert(wav_path):           同步播放，内部用锁串行化，避免多进程/线程抢设备。
  - play_alert_async(wav_path):     非阻塞版本，后台线程播放。
  - espeak_ng_synthesize(text):     用 espeak-ng 把文本合成 WAV 文件，返回路径或 None。
  - prepare_audio(text, ...):       只准备 WAV、不播放，返回路径（录像混音用它拿文件）。
  - speak_audio(text, ...):         高层封装：准备 WAV → 异步播放；失败回退占位提示音。
"""

from __future__ import annotations

import os
import sys
import threading
import subprocess
import tempfile

# ----------------------------------------------------------
# 二进制探测（启动时缓存，避免每次调用都 which）
# ----------------------------------------------------------
def _find(bin_name: str):
    from shutil import which
    return which(bin_name)

PAPLAY_BIN = _find("paplay")
APLAY_BIN = _find("aplay")
ESPEAK_BIN = _find("espeak-ng") or _find("espeak")

# 串行化锁：多个告警同时触发时排队播放，避免抢占音频设备导致混乱/报错
_play_lock = threading.Lock()


def _resolve(path: str) -> str:
    """把相对路径转绝对路径，便于在容器内任意 cwd 下定位音频文件。"""
    return path if os.path.isabs(path) else os.path.abspath(path)


def play_alert(wav_path, timeout: int = 10, volume: float = None) -> bool:
    """
    同步播放一个本地 WAV 文件。

    策略: 优先 paplay（PulseAudio），aplay（ALSA）作为兜底。
    鲁棒性: 文件不存在 / 播放超时 / 命令缺失 / 其它异常 → 只打印警告，返回 False，不抛出。
    并发: 使用模块级锁串行化，多个播放请求依次进行。

    参数:
      wav_path: WAV 文件路径
      timeout:  单次播放超时（秒），防止 paplay 卡死
      volume:   0.0~1.0 的相对音量（仅 paplay 支持 --volume），None 表示不调整
    返回:
      True 表示已尝试播放（不代表用户一定听到），False 表示无法播放
    """
    wav_path = _resolve(wav_path)
    if not wav_path or not os.path.isfile(wav_path):
        print(f"[audio] ⚠️ 音频文件不存在，跳过播放: {wav_path}")
        return False

    with _play_lock:
        # 1) 优先 paplay（PulseAudio 默认输出，已切到 USB 声卡）
        if PAPLAY_BIN:
            cmd = [PAPLAY_BIN]
            if volume is not None:
                # paplay --volume 取值 0~65536
                cmd += ["--volume", str(int(max(0.0, min(1.0, volume)) * 65536))]
            cmd.append(wav_path)
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
                if proc.returncode == 0:
                    return True
                err = proc.stderr.decode(errors="ignore").strip()
                print(f"[audio] ⚠️ paplay 返回非零 ({proc.returncode}): {err}")
            except subprocess.TimeoutExpired:
                print(f"[audio] ⚠️ paplay 播放超时（>{timeout}s）: {wav_path}")
            except Exception as e:
                print(f"[audio] ⚠️ paplay 播放异常: {e}")
        else:
            print("[audio] ⚠️ 未找到 paplay（pulseaudio-utils 未安装？）")

        # 2) 兜底 aplay（ALSA）
        if APLAY_BIN:
            try:
                proc = subprocess.run(
                    [APLAY_BIN, wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
                if proc.returncode == 0:
                    return True
                err = proc.stderr.decode(errors="ignore").strip()
                print(f"[audio] ⚠️ aplay 返回非零 ({proc.returncode}): {err}")
            except subprocess.TimeoutExpired:
                print(f"[audio] ⚠️ aplay 播放超时（>{timeout}s）: {wav_path}")
            except Exception as e:
                print(f"[audio] ⚠️ aplay 播放异常: {e}")
        else:
            print("[audio] ⚠️ 未找到 aplay（alsa-utils 未安装？）")

    print("[audio] ⚠️ paplay/aplay 均不可用，无法播放音频")
    return False


def play_alert_async(wav_path, timeout: int = 10, volume: float = None):
    """
    非阻塞播放：在后台守护线程中调用 play_alert。
    用于告警高频触发场景，避免阻塞推理/逻辑主循环。
    返回线程对象（通常无需等待）。
    """
    t = threading.Thread(
        target=play_alert,
        args=(wav_path,),
        kwargs={"timeout": timeout, "volume": volume},
        daemon=True,
    )
    t.start()
    return t


def espeak_ng_synthesize(text, out_path=None, voice="zh", speed=160, volume=100):
    """
    用 espeak-ng 命令行把文本合成 WAV 文件。

    参数:
      text:     要合成的文本（中文）
      out_path: 输出 WAV 路径；为 None 时自动生成临时文件
      voice/speed/volume: espeak-ng 参数
    返回:
      成功 → WAV 文件路径；失败 → None（不抛出）
    """
    if not ESPEAK_BIN:
        print("[audio] ⚠️ 未找到 espeak-ng，无法合成语音（请 apt install espeak-ng）")
        return None
    if not text:
        return None

    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="blindstick_")
        os.close(fd)

    try:
        cmd = [
            ESPEAK_BIN,
            "-v", str(voice),
            "-s", str(speed),
            "-a", str(volume),
            "-w", out_path,
            "--",
            text,
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="ignore").strip()
            print(f"[audio] ⚠️ espeak-ng 合成失败 ({proc.returncode}): {err}")
            return None
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 44:  # WAV 头至少 44 字节
            return out_path
        print(f"[audio] ⚠️ espeak-ng 未生成有效 WAV: {out_path}")
        return None
    except subprocess.TimeoutExpired:
        print("[audio] ⚠️ espeak-ng 合成超时")
        return None
    except Exception as e:
        print(f"[audio] ⚠️ espeak-ng 合成异常: {e}")
        return None


def prepare_audio(text, voice="zh", speed=160, volume=100, fallback_wav=None):
    """
    只准备音频、不播放：返回实际要播放的 WAV 路径（供录像混音复用同一份文件）。

    优先级:
      1. espeak-ng 把文案合成 WAV
      2. 合成失败 → 返回 fallback_wav（占位提示音）
      3. 都不可用 → None
    """
    wav = espeak_ng_synthesize(text, voice=voice, speed=speed, volume=volume)
    if wav:
        return wav

    if fallback_wav:
        fb = _resolve(fallback_wav)
        if os.path.isfile(fb):
            return fb
        print(f"[audio] ⚠️ 占位提示音也不存在: {fb}")
    return None


def speak_audio(text, voice="zh", speed=160, volume=100, fallback_wav=None) -> bool:
    """
    高层语音播报：先用 espeak-ng 把文案合成 WAV，再异步播放。
    若 espeak-ng 不可用，则尝试播放 fallback_wav（占位提示音）。
    任何情况都不抛出，且后台播放不阻塞调用方。

    返回:
      True  → 已发起播放（合成成功或回退成功）
      False → 连占位音都无法播放（彻底静默，仅打印）
    """
    wav = prepare_audio(text, voice=voice, speed=speed,
                        volume=volume, fallback_wav=fallback_wav)
    if not wav:
        return False

    play_alert_async(wav)
    return True


def diag():
    """打印音频后端探测结果，便于自检/调试。"""
    print("=== audio 后端探测 ===")
    print(f"paplay : {PAPLAY_BIN or '未找到'}")
    print(f"aplay  : {APLAY_BIN or '未找到'}")
    print(f"espeak : {ESPEAK_BIN or '未找到'}")
    print(f"Python : {sys.version.split()[0]}")


if __name__ == "__main__":
    diag()
