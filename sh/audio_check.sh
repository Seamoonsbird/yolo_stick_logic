#!/bin/bash
# ============================================================
# 容器内音频自检脚本（盲杖语音播报）
# 用法: 在容器内执行  bash sh/audio_check.sh
# 作用: 验证 PulseAudio 环境、播放工具、espeak-ng 是否就绪，
#       并实际合成一句中文做播放测试。
# ============================================================

echo "==== 环境变量 ===="
echo "PULSE_SERVER    = ${PULSE_SERVER:-<未设置>}"
echo "XDG_RUNTIME_DIR = ${XDG_RUNTIME_DIR:-<未设置>}"
echo "XDG_CONFIG_HOME = ${XDG_CONFIG_HOME:-<未设置>}"

echo
echo "==== PulseAudio native socket ===="
if [ -n "$XDG_RUNTIME_DIR" ] && [ -S "$XDG_RUNTIME_DIR/pulse/native" ]; then
  echo "✓ 找到 PulseAudio socket: $XDG_RUNTIME_DIR/pulse/native"
else
  echo "✗ 未找到 PulseAudio native socket，声音不可用"
  echo "  检查宿主机: ps aux | grep -i pulseaudio ; ls -l /run/user/\$(id -u)/pulse/native"
fi

echo
echo "==== 音频 / 合成工具 ===="
for b in paplay aplay espeak-ng espeak; do
  if command -v "$b" >/dev/null 2>&1; then
    echo "✓ $b: $(command -v "$b")"
  else
    echo "✗ $b 未安装（apt-get install pulseaudio-utils alsa-utils espeak-ng）"
  fi
done

echo
echo "==== 测试播放（中文合成 → paplay）===="
TMP=$(mktemp --suffix=.wav)
if command -v espeak-ng >/dev/null 2>&1; then
  if espeak-ng -v zh -s 160 -a 100 -w "$TMP" "语音播报测试成功" 2>/dev/null; then
    if paplay "$TMP" 2>/dev/null; then
      echo "✓ 合成 + 播放成功，请确认是否听到『语音播报测试成功』"
    else
      echo "✗ paplay 播放失败（设备/权限问题？）"
    fi
  else
    echo "✗ espeak-ng 合成失败"
  fi
else
  echo "✗ 无 espeak-ng，尝试直接播放占位音（如有）"
  python - <<'PY'
import os, glob
cands = glob.glob(os.path.join(os.path.dirname(__file__), "..", "assets", "sounds", "*.wav"))
if cands:
    from nodes.audio import play_alert
    print("播放:", cands[0])
    play_alert(cands[0])
else:
    print("未找到任何 WAV 占位音")
PY
fi
rm -f "$TMP"
