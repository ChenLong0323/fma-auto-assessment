import cv2
import numpy as np
import time
import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
from collections import deque

indices = [27, 28,  # 01 "left_wrist"
           30, 31,  # 23 "right_wrist",30
           0, 1,  # 45 "nose"
           36, 37,  # 67 "right_hip",36
           # action2 判断点
           3, 4,  # 89 "left_eye",3
           6, 7,  # 10,11 "right_eye",6
           # 添加左右手肘
           21, 22,  # 12,13 "left_elbow",21
           24, 25,  # 14,15 "right_elbow",24
           15, 16,  # 16,17 "left_shoulder",15
           12, 13  # 18,19 "right_shoulder",18
           ]

# 基本宽度(r)
shoulder_w = 52
hip_w = 37
window_w = 115

# 检验参数
win_h = 30
win_w = 20
check_h = 140


def main():
    # 初始化
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500

    # FPS 计算
    frame_count = 0
    start_time = time.time()
    fps = 0

    # 初始化 trigger，长度为100，初始空白
    trigger = deque(maxlen=100)

    # 加载模型
    model = YOLO('models/yolov8s-pose.pt')

    # 判断左右手状态
    flag = 0
    # 在l循环中使用的flag
    flag_l = 0
    # 在r循环中使用的flag
    flag_r = 0

    # 计算左右手循环次数
    count_sides = 0

    # 计算完成项目次数
    count = 0

    # 左右侧移动时间
    move_time_l = 0
    min_list_l = []
    move_time_r = 0
    min_list_r = []

    # 循环获取图像主体
    while True:
        capture = k4a.get_capture()
        # 读取frame,YOLO输入是三维图片
        frame = capture.color[:, :, :3]
        frame = cv2.flip(frame, 1)
        # 调用yolo检测
        results = model(frame, stream=True, conf=0.7, verbose=False)
        results1 = model(frame)

        for result in results:
            keypoints = np.ravel(result.keypoints.data[0].tolist())
            frame = result.plot()

            if len(keypoints) > max(indices):
                ext = keypoints[indices]

                # 循环内容


        # 绘制FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("rgb", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1
        # Calculate and print FPS every second
        if frame_count % 30 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 30 / elapsed_time
            print(f"FPS: {fps}")
            # Reset variables for the next second
            frame_count = 0
            start_time = time.time()
    k4a.stop()


if __name__ == "__main__":
    main()
