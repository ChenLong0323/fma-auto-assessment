import cv2
import numpy as np
import time
import pyk4a
from pyk4a import Config, PyK4A
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


def main():
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
        start_time_frame = time.time()
        capture = k4a.get_capture()
        if np.any(capture.depth):
            frame = capture.color
            cv2.imshow("rgb", frame)
            cv2.imshow("depth", colorize(capture.transformed_depth, (None, 5000)))
            cv2.imshow("ir", colorize(capture.transformed_ir, (None, 500)))
            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break
            # Update FPS
        # Increase frame count
        frame_count += 1

        # Calculate elapsed time for the current frame
        end_time_frame = time.time()
        elapsed_time_frame = end_time_frame - start_time_frame

        # Calculate elapsed time for the overall loop
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Calculate and print FPS every second
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps}")

            # Reset variables for the next second
            frame_count = 0
            start_time = time.time()

        # Print FPS for the current frame
        frame_fps = 1 / elapsed_time_frame
        #print(f"Frame FPS: {frame_fps}")
    k4a.stop()


if __name__ == "__main__":
    main()
