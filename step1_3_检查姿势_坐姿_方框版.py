import cv2
import numpy as np
import time
import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO

indices = [15, 16,  # 01 "left_shoulder",15
           18, 19,  # 23 "right_shoulder",18
           33, 34,  # 45 "left_hip",33
           36, 37,  # 67 "right_hip",36
           ]

# 基本宽度（需要是偶数）
shoulder_w = 52
hip_w = 37
window_w = 115

# 余量
shoulder_ext = 5
hip_ext = 5
sit_ext = 10


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
    model = YOLO('models/yolov8n-pose.pt')

    # 坐姿检测部分

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

                if (ext[0] - ext[2]) > 2 * shoulder_w + shoulder_ext:  # 肩膀连线长度，后退
                    cv2.putText(frame, 'go back', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                elif (ext[0] - ext[2]) < 2 * shoulder_w - shoulder_ext:  # 肩膀连线长度,前进
                    cv2.putText(frame, 'go front', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                elif (ext[0] + ext[2]) / 2 > 640 + sit_ext:  # 坐姿中心超过阈值，左移
                    cv2.putText(frame, 'go left', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                elif (ext[0] + ext[2]) / 2 < 640 - sit_ext:  # 坐姿中心超过阈值，右移
                    cv2.putText(frame, 'go right', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

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
