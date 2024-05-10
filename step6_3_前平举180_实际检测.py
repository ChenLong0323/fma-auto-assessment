import cv2
import numpy as np
import time
import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO

indices = [27, 28,  # 01 "left_wrist"
           30, 31,  # 23 "right_wrist",30
           33, 34,  # 45 "left_hip",33
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

    # 加载模型
    model = YOLO('models/yolov8s-pose.pt')

    flag = 0

    # 循环获取图像主体
    while True:
        capture = k4a.get_capture()
        # 读取frame,YOLO输入是三维图片
        frame = capture.color[:, :, :3]
        frame = cv2.flip(frame, 1)
        # 调用yolo检测
        results = model(frame, stream=True, conf=0.7, verbose=False)

        for result in results:
            keypoints = np.ravel(result.keypoints.data[0].tolist())
            frame = result.plot()

            if len(keypoints) > max(indices):
                ext = keypoints[indices]

                # 真正的循环内容

                if flag == 0:
                    # 绘制：上位开始
                    cv2.putText(frame, 'not begin', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    # 绘制左侧box
                    cv2.rectangle(frame,
                                  (int(ext[4] + win_w), int(ext[5])),
                                  (int(ext[4] + 2 * win_w), int(ext[5] + win_h)),
                                  (0, 255, 255), 2)
                    # 绘制右侧box
                    cv2.rectangle(frame,
                                  (int(ext[6] - win_w), int(ext[7])),
                                  (int(ext[6] - 2 * win_w), int(ext[7] + win_h)),
                                  (0, 255, 255), 2)
                    # 判断左手、左髋
                    if (abs((ext[0] - ext[4] - win_w)) < win_w) and (abs((ext[1] - ext[5])) < win_h):
                        flag = 1
                    # 判断右手、右髋
                    elif (abs((ext[2] - ext[6] + win_w)) < win_w) and (abs((ext[3] - ext[7])) < win_h):
                        flag = 2

                # Flag = 1 , 左手运动检测
                elif flag == 1:
                    # 绘制文字
                    cv2.putText(frame, 'score:0', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    # 绘制左侧目标line
                    cv2.line(frame, (640 + shoulder_w, check_h), (640 + 2 * shoulder_w, check_h), (0, 255, 255), 2)

                    # 判断左肘的y坐标都超过左眼
                    if ext[13] < ext[9]:
                        flag = 3

                # Flag = 2 , 右手运动检测
                elif flag == 2:
                    # 绘制文字
                    cv2.putText(frame, 'score:0', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    # 绘制右侧目标line
                    cv2.line(frame, (640 - shoulder_w, check_h), (640 - 2 * shoulder_w, check_h), (0, 255, 255), 2)
                    # 判断右肘的y坐标都超过右眼
                    if ext[15] < ext[11]:
                        flag = 3

                else:
                    cv2.putText(frame, 'score:2', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # 绘制box，绘制四个点
        cv2.rectangle(frame, (640 - window_w, 180), (640 + window_w, 710), (0, 255, 0), 2)
        # 绘制两肩连线
        cv2.line(frame, (640 + shoulder_w, 295), (640 - shoulder_w, 295), (0, 255, 0), 2)
        # 绘制两臀连线
        cv2.line(frame, (640 + hip_w, 422), (640 - hip_w, 422), (0, 255, 0), 2)

        fps_text = f"FPS: {fps}"  # 绘制FPS到Frame上
        cv2.putText(frame, fps_text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # frame = frame[0:720, 340:940] # 修剪绘制frame, 不修剪更快

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
