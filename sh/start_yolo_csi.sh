#!/bin/bash
# ============================================================
# 重新进入已存在的 YOLO 容器 (yolo26_csi)
# 如果容器正在运行 -> 使用 docker exec 进入
# 如果容器已停止   -> 使用 docker start -ai 启动并附着
# 如果容器不存在   -> 提示错误
# ============================================================

CONTAINER_NAME="yolo26_csi"

# 检查容器是否存在
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "错误：容器 '${CONTAINER_NAME}' 不存在！"
    echo "请先运行创建脚本创建该容器。"
    exit 1
fi

# 检查容器当前状态
STATUS=$(docker inspect --format='{{.State.Status}}' "$CONTAINER_NAME")

if [ "$STATUS" == "running" ]; then
    echo "容器 '${CONTAINER_NAME}' 正在运行，直接进入交互 shell..."
    docker exec -it "$CONTAINER_NAME" /bin/bash
elif [ "$STATUS" == "exited" ] || [ "$STATUS" == "created" ] || [ "$STATUS" == "stopped" ]; then
    echo "容器 '${CONTAINER_NAME}' 当前状态为 ${STATUS}，正在启动并附着..."
    docker start -ai "$CONTAINER_NAME"
else
    echo "未知状态：${STATUS}，尝试使用 start -ai..."
    docker start -ai "$CONTAINER_NAME"
fi