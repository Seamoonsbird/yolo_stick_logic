import cv2
import os

video_file = os.path.expanduser('output/orange.mkv')
output_folder = os.path.expanduser('/root/yolo26_data/data/orange_pic')

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print(f"无法打开视频文件: {video_file}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"视频总帧数: {total_frames}, FPS: {fps}")

frame_number = 0
frame_interval = 15
saved_frame_number = 1

while True:
    ret, frame = cap.read()

    if not ret:
        print("视频读取结束或读取失败")
        break

    if frame_number % frame_interval == 0:
        image_name = f"{saved_frame_number}.png"
        image_path = os.path.join(output_folder, image_name)

        success = cv2.imwrite(image_path, frame)
        if success:
            print(f"Saving frame {frame_number}/{total_frames} as {image_name}")
            saved_frame_number += 1
        else:
            print(f"保存失败: {image_path}")

    frame_number += 1

cap.release()
print("Video processing complete! Every 15th frame saved as an image!")

