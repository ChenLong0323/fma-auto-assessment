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


def main(path, name):
    save_path = os.path.join(path + patient_name)
    path_rgb = save_path + '/rgb'
    path_depth = save_path + '/depth'
    path_ir = save_path + '/ir'
    # 检测是否存在路径，不存在创建
    if not os.path.exists(save_path):
        os.makedirs(path_rgb)
        os.makedirs(path_depth)
        os.makedirs(path_ir)
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
    while True:
        capture = k4a.get_capture()
        if np.any(capture.color):
            # Increase frame count
            frame_count += 1

            # 数据获取，此处获取的数据为numpy格式
            rgb_frame = capture.color
            depth_frame = colorize(capture.transformed_depth, (None, 5000))
            ir_frame = colorize(capture.transformed_ir, (None, 200))
            # print(type(depth_frame))

            # 图像显示
            cv2.imshow("rgb", rgb_frame)
            cv2.imshow("depth", depth_frame)
            cv2.imshow("ir", ir_frame)

            # 数据保存
            np.save(os.path.join(path_rgb, f"{name}_rgb_{frame_count}.npy"), rgb_frame)
            np.save(os.path.join(path_depth, f"{name}_depth_{frame_count}.npy"), depth_frame)
            np.save(os.path.join(path_ir, f"{name}_ir_{frame_count}.npy"), ir_frame)

            # [check] 输出保存的路径进行检查
            # print(os.path.join(save_path + '/rgb', f"{name}_rgb_{frame_count}.png"))

            # 循环退出模块
            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break

            # FPS计算
            if frame_count % 30 == 0:
                start_time = fps_calculate(start_time)

    k4a.stop()


if __name__ == "__main__":
    data_path = os.getcwd()
    patient_name = 'hand_test'
    main(data_path, patient_name)
