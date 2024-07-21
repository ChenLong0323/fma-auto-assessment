import cv2
import numpy as np

import pyk4a
from helpers import convert_to_bgra_if_required
from pyk4a import Config, PyK4A


def get_color_image_size(config, imshow=True):
    if imshow:
        cv2.namedWindow("k4a")
    k4a = PyK4A(config)
    k4a.start()
    count = 0
    while True:
        capture = k4a.get_capture()
        if np.any(capture.color):
            count += 1
            if imshow:
                cv2.imshow("k4a", convert_to_bgra_if_required(config.color_format, capture.color))
                key = cv2.waitKey(10)
                if key == ord('q'):  # 检测按下 'q' 键
                    break
    cv2.destroyAllWindows()
    k4a.stop()
    return capture.color.nbytes


if __name__ == "__main__":
    imshow = True
    config_BGRA32 = Config(color_format=pyk4a.ImageFormat.COLOR_BGRA32)
    nbytes_BGRA32 = get_color_image_size(config_BGRA32, imshow=imshow)

    # output:
    # nbytes_BGRA32=3686400 nbytes_MJPG=229693
    # COLOR_BGRA32 is 16.04924834452944 larger
