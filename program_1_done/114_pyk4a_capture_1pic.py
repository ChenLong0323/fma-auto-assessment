import cv2
import numpy as np
import time
import pyk4a
from pyk4a import Config, PyK4A
from typing import Optional, Tuple
import os


def colorize(
    # 本函数读入numpy格式图片，和均衡化的上下阈值，返回numpy格式图像
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img

def fps_calculate(start_time):
    # Calculate and print FPS every second
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 30 / elapsed_time
    print(f"FPS: {fps}")
    start_time = time.time()
    return start_time

def main(name, save_path):
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
    # 循环获取图像主体
    capture = k4a.get_capture()
    if np.any(capture.color):
        # 数据获取，此处获取的数据为numpy格式
        rgb_frame = capture.color
        print(type(rgb_frame))
        print(rgb_frame.shape)
        # 图像显示
        cv2.imshow("rgb", rgb_frame)
        cv2.waitKey(0)  # 等待用户按下键盘上的任意键
        cv2.destroyAllWindows()  # 关闭窗口

    k4a.stop()


if __name__ == "__main__":
    save_path = 'C:/000_data'
    patient_name = 'YSL'
    main(patient_name, save_path)
