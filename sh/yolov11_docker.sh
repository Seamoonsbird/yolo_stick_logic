#!/bin/bash
# ============================================================
# YOLOv11 Docker 启动脚本（YoloSmartStick 项目）
# 镜像: yahboomtechnology/ultralytics:1.0.3
# 用途: 在 Jetson 设备上启动 YOLO 推理容器，挂载 YoloSmartStick
#       项目代码、摄像头设备和 X11 显示，用于智能棒/YOLO 检测开发。
# ============================================================

# 允许所有用户通过 TCP 连接到 X Server（图形界面显示）
xhost +

# --- 启动 Docker 容器 ---
docker run -it \
  --net=host \                                          # 使用宿主机网络栈
  --env="DISPLAY" \                                     # 将宿主机的 DISPLAY 环境变量传入容器
  --env="QT_X11_NO_MITSHM=1" \                          # 禁用 MIT-SHM，避免 Qt 应用显示异常
  -v /tmp/.X11-unix:/tmp/.X11-unix \                    # 挂载 X11 Unix socket，实现图形界面透传
  -v /home/jetson/temp:/ultralytics/ultralytics/temp \  # 挂载临时文件目录到 ultralytics 工作目录
  -v /home/jetson/YoloSmartStick:/YoloSmartStick \      # 挂载 YoloSmartStick 项目源码目录
  --device=/dev/video0 \                                # 直通 USB 摄像头设备
  -p 9090:9090 \                                        # 映射 9090 端口
  -p 8888:8888 \                                        # 映射 8888 端口
  yahboomtechnology/ultralytics:1.0.3 /bin/bash         # 镜像名:标签 + 启动交互式 shell