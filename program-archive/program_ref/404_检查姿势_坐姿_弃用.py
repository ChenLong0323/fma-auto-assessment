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

    # mask设置
    mask = np.load("sit_mask.npy")
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 指定透明度
    alpha = 0.3

    model = YOLO('models/yolov8s-pose.pt')

    # 循环获取图像主体
    while True:
        capture = k4a.get_capture()
        # 读取frame,YOLO输入是三维图片
        frame = capture.color[:, :, :3]

        # 调用yolo检测
        results = model(frame, stream=True, conf=0.7)
        for result in results:
            keypoints = result.keypoints.data[0].tolist()
            frame = result.plot()
        print(frame.shape)

        # 将希望叠加的图片与背景图片进行叠加
        # frame = cv2.addWeighted(frame, 1 - alpha, mask, alpha, 0)

        # 绘制FPS到Frame上
        fps_text = f"FPS: {fps}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 现实Frame
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
