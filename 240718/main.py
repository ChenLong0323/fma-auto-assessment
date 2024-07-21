"""
问题：去除ext过程，直接从keypoints获取点作为交互点
"""
import threading
from collections import deque

from func_ActionDetector import *
from func_data_class import *
from func_initial import *


def main():
    k4a, model, mp_hands = initialize(model_name='yolov8s')

    data_cons = Constants()
    data_var = HandStatus()
    data_result = ResultData()

    queue_keypoints = queue.Queue(maxsize=10)  # 增加队列的大小
    queue_frame_image = queue.Queue(maxsize=10)  # 增加队列的大小

    queue_fps_process = deque(maxlen=30)
    queue_fps_show = deque(maxlen=30)
    queue_fps_interface = deque(maxlen=30)

    trigger_queue = deque(maxlen=60)
    trigger_queue_overtime = deque(maxlen=1200)
    # 程序是否运行
    flag_running = threading.Event()
    flag_running.set()  # Initially set the event to True

    # 动作序号，目前还没有0 姿势矫正，只从1开始
    action_idx = 0

    action_manager = ActionManager()  # 初始化动作管理器

    thread_frame = threading.Thread(target=func_frame_cap, args=(k4a, model, mp_hands,
                                                                 queue_frame_image, queue_keypoints, queue_fps_process,
                                                                 data_cons, flag_running, action_idx))

    thread_show = threading.Thread(target=func_frame_show, args=(queue_frame_image, queue_fps_show,
                                                                 flag_running))

    thread_interface = threading.Thread(target=func_inter, args=(flag_running, action_idx, action_manager,
                                                                 data_cons, data_var, data_result,
                                                                 queue_keypoints, queue_fps_interface,queue_frame_image,
                                                                 trigger_queue, trigger_queue_overtime))

    thread_frame.start()
    thread_show.start()
    thread_interface.start()

    try:
        while flag_running.is_set():
            print('main is running')
            time.sleep(0.1)
    except KeyboardInterrupt:
        flag_running.clear()  # Set the event to False to signal all threads to stop

    thread_frame.join()
    thread_show.join()
    thread_interface.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
