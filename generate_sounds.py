#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
盲杖语音播报 —— 预录音频生成脚本（一次性工具）。

用 edge-tts（微软中文神经语音，离线不可用，仅用于「生成」阶段）
把 nodes/logic.py 中 SOUND_PRESETS 的 17 句文案合成 WAV，
写成 16-bit PCM 单声道文件，落进 assets/sounds/。

生成的 WAV 是静态文件，之后设备端离线播放，不依赖网络。

依赖（仅需在本机生成时安装）:
    pip install edge-tts miniaudio

用法:
    python generate_sounds.py
"""

import asyncio
import os
import sys
import wave

import edge_tts
import miniaudio

# 让本脚本能 import 同目录下的 config / nodes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nodes.logic import SOUND_PRESETS, SOUNDS_DIR  # noqa: E402

# 中文神经语音（可换成 zh-CN-YunxiNeural 男声 / zh-CN-XiaoyiNeural 等）
VOICE = "zh-CN-XiaoxiaoNeural"
# 语速微调：负数变慢更清晰（盲杖场景建议稍慢）。默认 +0%。
RATE = "-5%"


async def _synth_to_mp3(text: str, mp3_path: str):
    """用 edge-tts 把文本合成 MP3 写入文件。"""
    comm = edge_tts.Communicate(text, VOICE, rate=RATE)
    with open(mp3_path, "wb") as f:
        async for chunk in comm.stream():
            if chunk.get("type") == "audio":
                f.write(chunk["data"])


def _mp3_to_wav(mp3_path: str, wav_path: str):
    """把 MP3 解码为 16-bit PCM 单声道 WAV（miniaudio 解码 + 手动降混，无需 ffmpeg）。"""
    import array

    with open(mp3_path, "rb") as f:
        mp3_bytes = f.read()
    # 默认解码为 16-bit PCM 立体声（强制单声道会触发 miniaudio 内部报错，故先解后降混）
    decoded = miniaudio.decode(mp3_bytes, output_format=miniaudio.SampleFormat.SIGNED16)

    raw = decoded.samples  # 已是 array.array('h')（int16）
    if decoded.nchannels == 2:
        mono = array.array("h")
        for i in range(0, len(raw), 2):
            mono.append((raw[i] + raw[i + 1]) // 2)
        raw = mono

    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)                  # 单声道（盲杖单喇叭）
        w.setsampwidth(2)                  # 16-bit
        w.setframerate(decoded.sample_rate)
        w.writeframes(raw.tobytes())


async def main():
    os.makedirs(SOUNDS_DIR, exist_ok=True)
    total = len(SOUND_PRESETS)
    print(f"共 {total} 条预录语音，输出目录: {SOUNDS_DIR}")
    print(f"语音: {VOICE}  语速: {RATE}\n")
    for i, (key, (fname, text)) in enumerate(SOUND_PRESETS.items(), 1):
        wav_path = os.path.join(SOUNDS_DIR, fname)
        mp3_path = os.path.join(SOUNDS_DIR, key + ".mp3")
        try:
            await _synth_to_mp3(text, mp3_path)
            _mp3_to_wav(mp3_path, wav_path)
            os.remove(mp3_path)
            size = os.path.getsize(wav_path)
            print(f"[{i:2d}/{total}] ✅ {fname}  「{text}」  ({size} bytes)")
        except Exception as e:
            print(f"[{i:2d}/{total}] ❌ {fname}  「{text}」  失败: {e}")
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
    print("\n生成完成。")


if __name__ == "__main__":
    asyncio.run(main())
