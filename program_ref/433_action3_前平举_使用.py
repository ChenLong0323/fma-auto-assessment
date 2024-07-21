import cv2
import numpy as np
import time
import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
from typing import Optional, Tuple


def colorize(
        image: np.ndarray,
        clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
        colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # img = cv2.applyColorMap(img, colormap)
    return img


# keypoint
# array([684.1 , 145.42, 614.79, 149.23, 718.22, 230.52, 588.13, 233.17])


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
    model = YOLO('models/yolov8m-pose.pt')

    indices = [27, 28,  # 01 "left_wrist"
               30, 31,  # 23 "right_wrist",30
               33, 34,  # 45 "left_hip",33
               36, 37,  # 67 "right_hip",36
               15, 16,  # 89"left_shoulder", 15
               18, 19,  # 10"right_shoulder",18
               21, 22,  # 12    "left_elbow",21
               24, 25]  # 14    "right_elbow",24

    flag = 0

    win_h = 30
    win_w = 20
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

            # 检查索引最大值，以防空引用报错
            if len(keypoints) > max(indices):
                ext = keypoints[indices]
                if flag == 0:
                    # 绘制：上位开始
                    cv2.putText(frame, 'not begin', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    # 绘制左侧box
                    cv2.rectangle(frame, (int(ext[4] + win_w), int(ext[5])),
                                  (int(ext[4] + 2 * win_w), int(ext[5] + win_h)), (0, 255, 255), 2)
                    # 绘制右侧box
                    cv2.rectangle(frame, (int(ext[6] - win_w), int(ext[7])),
                                  (int(ext[6] - 2 * win_w), int(ext[7] + win_h)), (0, 255, 255), 2)
                    # 判断开始位置：左
                    if (abs((ext[0] - ext[4] - win_w)) < win_w) and (abs((ext[1] - ext[5])) < win_h):
                        flag = 1
                    # 判断开始位置：右
                    elif (abs((ext[2] - ext[6] + win_w)) < win_w) and (abs((ext[3] - ext[7])) < win_h):
                        flag = 2

                # 左侧检测
                elif flag == 1:
                    # 绘制文字
                    cv2.putText(frame, 'score:0', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    # 绘制左侧box
                    cv2.rectangle(frame, (int(ext[8]), int(ext[9])),
                                  (int(ext[8] + win_w), int(ext[9] + win_h)), (0, 255, 255), 2)
                    # 判断左腕到左肩的距离
                    if (abs((ext[0] - ext[8])) < win_w) and (abs((ext[1] - ext[9])) < win_h) and (abs((ext[12] - ext[8])) < win_w) and (
                            abs((ext[13] - ext[9])) < win_h):
                        flag = 3

                # 右侧检测
                elif flag == 2:
                    # 绘制文字
                    cv2.putText(frame, 'score:0', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    # 绘制右侧box
                    cv2.rectangle(frame, (int(ext[10]), int(ext[11])),
                                  (int(ext[10] - win_w), int(ext[11] + win_h)), (0, 255, 255), 2)
                    # 判断右腕到右肩的距离
                    if (abs((ext[2] - ext[10])) < win_w) and (abs((ext[3] - ext[11])) < win_h) and (abs((ext[14] - ext[10])) < win_w) and (
                            abs((ext[15] - ext[11])) < win_h):
                        flag = 3
                else:
                    cv2.putText(frame, 'score:2', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (520, 70), (760, 700), (0, 255, 255), 2)  # 绘制box，绘制四个点
        cv2.line(frame, (675, 150), (605, 150), (255, 0, 0), 2)  # 绘制两耳连线
        cv2.line(frame, (705, 230), (575, 230), (0, 0, 255), 2)  # 绘制两肩连线

        fps_text = f"FPS: {fps}"  # 绘制FPS到Frame上
        cv2.putText(frame, fps_text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # frame = frame[0:720, 340:940] # 修剪绘制frame, 不修剪更快

        cv2.imshow("rgb", frame)
        cv2.imshow("depth", colorize(capture.transformed_depth, (None, 5000)))
        cv2.imshow("ir", colorize(capture.transformed_ir, (None, 500)))
        # 按Q退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # 计算FPS
        frame_count += 1
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
