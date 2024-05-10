import cv2
import numpy as np
import time
import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO


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

    indices = [27, 28,  # "left_wrist"
               30, 31,  # "right_wrist",30
               33, 34,  # "left_hip",33
               36, 37,  # "right_hip",36
               9, 10,   # "left_ear", 9
               12, 13]  # "right_ear",12

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
                if flag == 0:

                    distance1 = (ext[0] - ext[6]) ** 2 + (ext[1] - ext[7]) ** 2
                    distance2 = (ext[2] - ext[4]) ** 2 + (ext[3] - ext[5]) ** 2
                    if distance1 < 450:  # 肩膀连线长度为130，后退
                        flag = 1
                    elif distance2 < 450:  # 肩膀连线长度为130,前进
                        flag = 2
                elif flag == 1:
                    distance3 = (ext[0] - ext[8]) ** 2 + (ext[1] - ext[9]) ** 2
                    if distance3 < 3700:
                        flag = 3
                elif flag == 2:
                    distance4 = (ext[2] - ext[10]) ** 2 + (ext[3] - ext[11]) ** 2
                    if distance4 < 3700:
                        flag = 3

        if flag == 0:
            cv2.putText(frame, 'not begin', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        elif flag == 1 or flag == 2:
            cv2.putText(frame, 'begin', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'finish', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (520, 70), (760, 700), (0, 255, 0), 2)  # 绘制box，绘制四个点
        cv2.line(frame, (675, 150), (605, 150), (255, 0, 0), 2)  # 绘制两耳连线
        cv2.line(frame, (705, 230), (575, 230), (0, 0, 255), 2)  # 绘制两肩连线

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
