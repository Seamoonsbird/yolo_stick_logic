#!/bin/bash
# ============================================================
# YOLO Docker 启动脚本（NVIDIA GPU + 摄像头 + 语音播报）
# 镜像: yahboomtechnology/ultralytics:1.0.4
#
# 音频方案（盲杖语音播报）:
#   不再挂载 /dev/snd，也不使用 --privileged。
#   改为映射宿主机 PulseAudio native socket（+ cookie），
#   由容器内的 paplay 走系统默认输出（宿主机已切到 USB 声卡）。
#   容器仍保持 root 身份运行（与原有脚本一致）。
# ============================================================

set -e  # 遇到任何错误立即退出，避免后续命令在异常状态下执行

# --- X11 显示配置 ---
export DISPLAY=:0
xhost +local:root >/dev/null 2>&1 || true
xhost +local: >/dev/null 2>&1 || true

# --- 摄像头设备检测 ---
VIDEO_DEVICES=""
for dev in /dev/video*; do
  [ -e "$dev" ] && VIDEO_DEVICES="$VIDEO_DEVICES --device=$dev"
done

# --- PulseAudio socket 映射（替代 /dev/snd） ---
# 宿主机以当前用户身份运行桌面 / PulseAudio，native socket 位于 XDG_RUNTIME_DIR 下。
HOST_UID=$(id -u)
export XDG_RUNTIME_DIR="/run/user/${HOST_UID}"
PULSE_SOCKET="${XDG_RUNTIME_DIR}/pulse/native"

# 若上面的路径不是 socket（例如某些 root 桌面环境），退回 /run/pulse
if [ ! -S "$PULSE_SOCKET" ]; then
  XDG_RUNTIME_DIR="/run/pulse"
  PULSE_SOCKET="${XDG_RUNTIME_DIR}/native"
fi

# PulseAudio cookie（客户端认证用），位于宿主机用户家目录
PULSE_COOKIE_SRC="$HOME/.config/pulse"
PULSE_COOKIE_DST="/root/.config/pulse"

AUDIO_MOUNTS=""
AUDIO_ENVS="-e XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR} -e PULSE_SERVER=unix:${PULSE_SOCKET} -e XDG_CONFIG_HOME=/root/.config"
if [ -S "$PULSE_SOCKET" ]; then
  # 把宿主机 runtime 目录挂进容器（同路径），并挂入 cookie 供认证
  AUDIO_MOUNTS="-v ${XDG_RUNTIME_DIR}:${XDG_RUNTIME_DIR} "
  if [ -d "$PULSE_COOKIE_SRC" ]; then
    AUDIO_MOUNTS="$AUDIO_MOUNTS -v ${PULSE_COOKIE_SRC}:${PULSE_COOKIE_DST}:ro"
  fi
else
  echo "⚠️ 未找到 PulseAudio native socket ($PULSE_SOCKET)，容器将无声音。"
  echo "   请确认宿主机 PulseAudio 已启动：systemctl --user status pulseaudio"
fi

# --- 启动 Docker 容器 ---
# 关键变更（相较旧版）:
#   - 移除 --privileged（改用显式设备 / 挂载，符合最小权限）
#   - 移除 /dev/snd 挂载（改用 PulseAudio socket）
#   - 新增 PulseAudio socket + cookie 挂载与 PULSE_SERVER 等环境变量
# 保留: GPU 透传、摄像头设备、X11 显示、IPC/网络 host。
docker run -it \
  --net=host \
  --ipc=host \
  --runtime=nvidia \
  --name yolo26_csi_speak \
  --gpus all \
  -e DISPLAY=:0 \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=/root/.Xauthority \
  -e TZ=Asia/Shanghai \
  -v ~/yolo26_data:/root/yolo26_data \
  -v ~/yahboom_demo:/ultralytics/yahboom_demo \
  -v ~/yolo_stick_logic:/ultralytics/yolo_stick_logic \
  -v /tmp:/tmp \
  -v $HOME/.Xauthority:/root/.Xauthority:ro \
  -v /etc/nv_tegra_release:/etc/nv_tegra_release:ro \
  -v /etc/localtime:/etc/localtime:ro \
  $VIDEO_DEVICES \
  $AUDIO_MOUNTS \
  $AUDIO_ENVS \
  yahboomtechnology/ultralytics:1.0.4 /bin/bash
