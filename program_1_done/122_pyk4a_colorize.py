import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
import time
from typing import Optional, Tuple


def colorize(
        image: np.ndarray,
        clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
        colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #img = cv2.applyColorMap(img, colormap)
    return img


def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.OFF,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=False,
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()

    while True:
        # Start time before capturing
        start_time_frame = time.time()

        capture = k4a.get_capture()
        if np.any(capture.depth):
            # 输出伪彩色深度图
            cv2.imshow("ir500", colorize(capture.ir, (None, 500)))
            cv2.imshow("ir1000", colorize(capture.ir, (None, 1000)))
            cv2.imshow("ir3000", colorize(capture.ir, (None, 3000)))
            # cv2.imshow("ir5000", colorize(capture.transformed_ir, (None, 5000)))
            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break

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
        print(f"Frame FPS: {frame_fps}")

    k4a.stop()


if __name__ == "__main__":
    main()
