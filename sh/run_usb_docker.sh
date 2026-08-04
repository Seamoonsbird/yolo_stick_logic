#!/bin/bash
# ============================================================
# ROS2 USB 摄像头 Docker 启动脚本
# 镜像: yahboomtechnology/ros2-base:2.0.2
# 用途: 在 Jetson 设备上启动 ROS2 基础容器，挂载 USB 摄像头
#       和 X11 显示，用于 yahboomcar 机器人开发。
# ============================================================

# 允许所有用户通过 TCP 连接到 X Server（图形界面显示）
xhost +

# --- 启动 Docker 容器 ---
docker run -it \
  --net=host \                                  # 使用宿主机网络栈，ROS2 节点通信需要
  --env="DISPLAY" \                             # 将宿主机的 DISPLAY 环境变量传入容器
  --env="QT_X11_NO_MITSHM=1" \                  # 禁用 MIT-SHM，避免 Qt 应用显示黑屏/闪烁
  -v /tmp/.X11-unix:/tmp/.X11-unix \            # 挂载 X11 Unix socket，实现图形界面透传
  -v /home/jetson/temp:/root/yahboomcar_ros2_ws/temp \  # 挂载临时文件目录到 ROS2 工作空间
  --device=/dev/video0 \                        # 直通 USB 摄像头设备（video0 通常是第一个摄像头）
  -p 9090:9090 \                                # 映射 9090 端口（常用于 Web 服务或调试）
  -p 8888:8888 \                                # 映射 8888 端口（常用于 Jupyter 或 Web 面板）
  yahboomtechnology/ros2-base:2.0.2 /bin/bash   # 镜像名:标签 + 启动交互式 shell