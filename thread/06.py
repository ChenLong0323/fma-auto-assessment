"""
问题：去除ext过程，直接从keypoints获取点作为交互点
"""
from func_initial7 import *
from func_data_class import *
import cv2
import time
from collections import deque
import threading
import queue
import logging

# 配置日志记录
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log", mode='w'),  # 使用 mode='w' 来覆盖之前的日志文件
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)


def main():
    k4a, model, mp_hands = initialize(model_name='yolov8s')

    data_cons = Constants()
    data_var = HandStatus()
    data_result = ResultData()

    queue_keypoints = queue.Queue(maxsize=10)  # 增加队列的大小
    queue_image = queue.Queue(maxsize=10)  # 增加队列的大小

    queue_fps_process = deque(maxlen=30)
    queue_fps_show = deque(maxlen=30)
    queue_fps_interface = deque(maxlen=30)

    trigger_queue = deque(maxlen=60)

    flag_run = [1, 0]

    indices = [
        0, 1,  # 鼻子 x 坐标、y 坐标
        15, 16,  # 左肩膀 x 坐标、y 坐标
        18, 19,  # 右肩膀 x 坐标、y 坐标
    ]

    thread_frame = threading.Thread(target=func_frame_cap, args=(k4a, model, mp_hands,
                                                                 queue_image, queue_keypoints, queue_fps_process,
                                                                 data_cons, flag_run))
    thread_show = threading.Thread(target=func_frame_show, args=(queue_image, queue_fps_show,
                                                                 flag_run))
    thread_interface = threading.Thread(target=interface, args=(data_var, data_result, data_cons,
                                                                queue_keypoints, trigger_queue, queue_fps_interface,
                                                                flag_run, indices))

    thread_frame.start()
    thread_show.start()
    thread_interface.start()

    try:
        while flag_run[0]:
            print('main is running')
            time.sleep(0.1)
    except KeyboardInterrupt:
        flag_run[0] = 0

    thread_frame.join()
    thread_show.join()
    thread_interface.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
