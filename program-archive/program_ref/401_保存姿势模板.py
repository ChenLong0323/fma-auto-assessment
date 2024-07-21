import cv2
import numpy as np
import time
import pyk4a
from pyk4a import Config, PyK4A
from typing import Optional, Tuple
import os


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        #print('1')
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()
        #print('2')
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # img = cv2.applyColorMap(img, colormap)
    return img


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
    key = cv2.waitKey(3000)
    # FPS 计算
    frame_count = 0
    start_time = time.time()
    # 循环获取图像主体
    while frame_count == 0:
        capture = k4a.get_capture()
        if np.any(capture.color):
            # Increase frame count
            frame_count += 2

            rgb_frame = capture.color
            cv2.imshow("rgb", rgb_frame)
            depth_frame = colorize(capture.transformed_depth, (None, 5000))
            ir_frame = colorize(capture.transformed_ir, (None, 500))

            cv2.imwrite(f"{name}_rgb_{frame_count}.png", rgb_frame, )
            cv2.imwrite(f"{name}_depth_{frame_count}.png", depth_frame)
            cv2.imwrite(f"{name}_ir_{frame_count}.png", ir_frame)
            np.save(f"{name}_rgb_{frame_count}.npy", rgb_frame, )
            np.save(f"{name}_depth_{frame_count}.npy", depth_frame)
            np.save(f"{name}_ir_{frame_count}.npy", ir_frame)
            print(os.path.join(save_path + '/rgb', f"{name}_rgb_{frame_count}.png"))
            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break

        # Calculate and print FPS every second
        if frame_count % 30 ==0:
            end_time = time.time()
            elapsed_time = end_time - start_time

            fps = 30 / elapsed_time
            print(f"FPS: {fps}")

            start_time = time.time()

    k4a.stop()


if __name__ == "__main__":
    save_path = 'D:/02_Codes/07_Pyk4a+Yolo/'
    patient_name = 'step8_2'
    main(patient_name, save_path)
