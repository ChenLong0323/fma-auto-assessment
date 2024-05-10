import cv2
import numpy as np
import time  # 导入 time 模块

import pyk4a
from helpers import convert_to_bgra_if_required
from pyk4a import Config, PyK4A


def get_color_image_size(config, imshow=True):
    if imshow:
        cv2.namedWindow("k4a")
    k4a = PyK4A(config)
    k4a.start()

    count = 0
    start_time = time.time()  # 记录开始时间

    while True:
        capture = k4a.get_capture()
        if np.any(capture.color):
            count += 1
            if imshow:
                print(config.color_format)
                frame = convert_to_bgra_if_required(config.color_format, capture.color)

                # 计算帧率
                end_time = time.time()  # 记录结束时间
                elapsed_time = end_time - start_time
                fps = count / elapsed_time

                # 在图像左上角添加帧率信息
                #cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(fps)
                cv2.imshow("k4a", frame)
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
    # nbytes_BGRA32=3686400
