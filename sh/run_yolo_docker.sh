#!/bin/bash
# ============================================================
# YOLO Docker 启动脚本（NVIDIA GPU + 摄像头）
# 镜像: yahboomtechnology/ultralytics:1.0.4
# 用途: 在 Jetson 等 NVIDIA 边缘设备上以特权模式运行 YOLO 容器，
#       支持 GPU 加速、X11 图形显示、摄像头直通。
# ============================================================

set -e  # 遇到任何错误立即退出，避免后续命令在异常状态下执行

# --- X11 显示配置 ---
# 设置 DISPLAY 环境变量，指向本地显示器（:0 通常是物理主屏幕）
export DISPLAY=:0

# 允许本地用户和 root 用户通过 xhost 连接到 X Server
# 错误输出重定向到 /dev/null，失败也不终止脚本（某些无桌面的服务器没有 X Server）
xhost +local:root >/dev/null 2>&1 || true
xhost +local: >/dev/null 2>&1 || true

# --- 摄像头设备检测 ---
# 动态检测 /dev/video* 设备（USB 摄像头、CSI 摄像头等）
# 遍历所有匹配项，存在则拼接到 --device 参数中
# 没有摄像头设备也不会报错，容器仍可正常启动
VIDEO_DEVICES=""
for dev in /dev/video*; do
  [ -e "$dev" ] && VIDEO_DEVICES="$VIDEO_DEVICES --device=$dev"
done

# --- 启动 Docker 容器 ---
docker run -it \
  --net=host \                          # 使用宿主机网络栈（性能更好，但端口直接暴露）
  --ipc=host \                          # 共享宿主机 IPC 命名空间（多进程通信需要）
  --privileged \                        # 特权模式，给予容器几乎所有宿主机能力（硬件访问需要）
  --runtime=nvidia \                    # 使用 NVIDIA 容器运行时，支持 GPU 透传
  --name yolo26_csi \                   # 将容器命名，方便下次重新打开
  --gpus all \                          # 将所有可用 GPU 分配给容器
  -e DISPLAY=:0 \                       # 容器内 DISPLAY 环境变量
  -e QT_X11_NO_MITSHM=1 \               # 禁用 MIT-SHM 共享内存扩展（避免 Qt/X11 显示异常）
  -e XAUTHORITY=/root/.Xauthority \     # X11 认证文件路径
  -v ~/yolo26_data:/root/yolo26_data \  # 挂载 YOLO 数据集目录
  -v ~/yahboom_demo:/ultralytics/yahboom_demo \  # 挂载 yahboom 演示程序目录
  -v ~/yolo_stick_logic:/root/yolo_stick_logic\  #挂载决策程序目录
  -v /tmp:/tmp \                        # 共享临时文件目录（X11 socket 通信需要）
  -v $HOME/.Xauthority:/root/.Xauthority:ro \     # 挂载 X11 认证文件（只读）
  -v /etc/nv_tegra_release:/etc/nv_tegra_release:ro \  # 挂载 NVIDIA Tegra 版本信息（只读）
  $VIDEO_DEVICES \                      # 动态注入摄像头设备
  yahboomtechnology/ultralytics:1.0.4 /bin/bash    # 镜像名:标签 + 启动 shell