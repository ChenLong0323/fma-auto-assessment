"""
想通过判断Queue是否满进行
方案2:if queue_image.full():
"""
from func_initial import initialize, draw_circles, extract_landmark_coordinates
from func_data_class import Constants, HandStatus, ResultData
import cv2
import time
from collections import deque
import threading
import queue
import numpy as np
import logging

# 配置日志记录
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log", mode='w'),  # 使用 mode='w' 来覆盖之前的日志文件
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)


def process_frame(k4a, model, mp_hands,
                  queue_image, queue_keypoints, queue_fps_process,
                  data_cons, running_flag):
    while running_flag[0]:
        if queue_image.full():
            print("Queue is full in main")
            continue
        else:
            try:
                capture = k4a.get_capture()
                image = capture.color[:, 640 - data_cons.WIDTH_DET_POSE:640 + data_cons.WIDTH_DET_POSE, :3]
                image_hand = cv2.cvtColor(
                    image[:, data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND:data_cons.WIDTH_DET_POSE + data_cons.WIDTH_DET_HAND, :3],
                    cv2.COLOR_BGR2RGB)
                height, width, _ = image_hand.shape

                results_pose = model(image, stream=True, conf=0.7, verbose=False)
                results_hand = mp_hands.process(image_hand).multi_hand_landmarks

                x_coordinates, y_coordinates = extract_landmark_coordinates(results_hand, height, width,
                                                                            data_cons.WIDTH_DET_POSE,
                                                                            data_cons.WIDTH_DET_HAND)

                for result in results_pose:
                    keypoints = np.ravel(result.keypoints.data[0].tolist())
                    image_show = result.plot()
                    draw_circles(image_show, x_coordinates, y_coordinates)
                    image_show = cv2.flip(image_show, 1)

                    queue_keypoints.put((keypoints, x_coordinates, y_coordinates), timeout=0.1, block=False)
                    queue_image.put(image_show, timeout=0.1)

                    queue_fps_process.append(time.time())
                    if len(queue_fps_process) > 30:
                        queue_fps_process.popleft()

                    if len(queue_fps_process) > 1:
                        fps = len(queue_fps_process) / (queue_fps_process[-1] - queue_fps_process[0])
                        logging.info(f"FPS_process: {fps:.2f}")

            except Exception as e:
                logging.error(f"Error in process_frame: {e}")
                continue


def show_frame(queue_image, running_flag, queue_fps_show):
    while running_flag[0]:
        try:
            queue_fps_show.append(time.time())
            if len(queue_fps_show) > 30:
                queue_fps_show.popleft()

            if len(queue_fps_show) > 1:
                fps = len(queue_fps_show) / (queue_fps_show[-1] - queue_fps_show[0])
                logging.info(f"FPS_show: {fps:.2f}")

            image_show = queue_image.get()
            cv2.imshow("rgb", image_show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running_flag[0] = False
                break

        except queue.Full:
            logging.warning("Queue is full in show_frame")
            continue
        except Exception as e:
            logging.error(f"Error in show_frame: {e}")
            break


def interface(data_var, data_result, data_cons,
              queue_keypoints, trigger_queue, queue_fps_interface,
              running_flag, indices):
    while running_flag[0]:
        try:
            queue_fps_interface.append(time.time())
            if len(queue_fps_interface) > 30:
                queue_fps_interface.popleft()

            if len(queue_fps_interface) > 1:
                fps = len(queue_fps_interface) / (queue_fps_interface[-1] - queue_fps_interface[0])
                logging.info(f"FPS_interface: {fps:.2f}")

            keypoints, x_coordinates, y_coordinates = queue_keypoints.get()
            if len(keypoints) > max(indices) and None not in x_coordinates and None not in y_coordinates:
                ext = keypoints[indices]
                logging.info(f"Handedness: {data_var.flag_handedness}, Left: {data_var.flag_left}, Right: {data_var.flag_right}")

                if data_var.flag_handedness == 0:
                    if ext[5] > y_coordinates[0]:
                        trigger_queue.append(1)
                        if sum(trigger_queue) > data_cons.THRESHOLD:
                            data_var.flag_handedness = 2
                            trigger_queue.clear()
                    elif ext[3] > y_coordinates[3]:
                        trigger_queue.append(1)
                        if sum(trigger_queue) > data_cons.THRESHOLD:
                            trigger_queue.clear()
                            data_var.flag_handedness = 1
                    else:
                        trigger_queue.append(0)

                elif data_var.flag_handedness == 1:
                    handle_left_hand(data_var, data_result, data_cons, x_coordinates, y_coordinates, ext, trigger_queue)

                elif data_var.flag_handedness == 2:
                    handle_right_hand(data_var, data_result, data_cons, x_coordinates, y_coordinates, ext, trigger_queue)

        except queue.Full:
            logging.warning("Queue is full in interface")
            continue
        except Exception as e:
            logging.error(f"Error in interface: {e}")
            break


def handle_left_hand(data_var, data_result, data_cons, x_coordinates, y_coordinates, ext, trigger_queue):
    if data_var.flag_left == 0:
        logging.info("Handling left hand")
        if abs((ext[0] - x_coordinates[3])) > 6 * data_cons.WIN_W:
            trigger_queue.append(1)
            if sum(trigger_queue) > data_cons.THRESHOLD:
                trigger_queue.clear()
                data_var.flag_left = 1
                if data_var.count == 0:
                    data_var.move_start_time = time.time()

                if data_var.count != 0:
                    data_result.min_list_l.append(data_var.dis)
                data_var.dis = 1000

                if data_var.count > 4:
                    data_var.flag_handedness = 2
                    data_var.count = 0
                    data_var.count_sides += 1
                    data_result.time_left = time.time() - data_var.move_start_time
                    return

        else:
            trigger_queue.append(0)

    elif data_var.flag_left == 1:
        if ext[2] > x_coordinates[3]:
            data_var.dis1 = abs(ext[0] - x_coordinates[3]) + abs(ext[1] - y_coordinates[3])
            if data_var.dis > data_var.dis1:
                data_var.dis = data_var.dis1

            trigger_queue.append(1)
            if sum(trigger_queue) > data_cons.THRESHOLD:
                trigger_queue.clear()
                data_var.count += 1
                data_var.flag_left = 0

        else:
            trigger_queue.append(0)


def handle_right_hand(data_var, data_result, data_cons, x_coordinates, y_coordinates, ext, trigger_queue):
    if data_var.flag_right == 0:
        logging.info("Handling right hand")
        if abs((ext[0] - x_coordinates[0])) > 6 * data_cons.WIN_W:
            trigger_queue.append(1)
            if sum(trigger_queue) > data_cons.THRESHOLD:
                trigger_queue.clear()
                data_var.flag_right = 1
                if data_var.count == 0:
                    data_var.move_start_time = time.time()

                if data_var.count != 0:
                    data_result.min_list_r.append(data_var.dis)
                data_var.dis = 1000
        else:
            trigger_queue.append(0)

    elif data_var.flag_right == 1:
        if ext[4] < x_coordinates[0]:
            data_var.dis1 = abs(ext[0] - x_coordinates[0]) + abs(ext[1] - y_coordinates[0])
            if data_var.dis > data_var.dis1:
                data_var.dis = data_var.dis1

            trigger_queue.append(1)
            if sum(trigger_queue) > data_cons.THRESHOLD:
                trigger_queue.clear()
                data_var.count += 1
                data_var.flag_right = 0

                if data_var.count > 4:
                    data_var.flag_handedness = 1
                    data_var.count = 0
                    data_var.count_sides += 1
                    data_result.time_right = time.time() - data_var.move_start_time
                    data_result.min_list_r.append(data_var.dis)
                    data_var.dis = 1000


def main():
    k4a, model, mp_hands = initialize(model_name='yolov8s')

    data_cons = Constants()
    data_var = HandStatus()
    data_result = ResultData()

    queue_keypoints = queue.Queue(maxsize=1)
    queue_image = queue.Queue(maxsize=1)

    queue_fps_process = deque(maxlen=30)
    queue_fps_show = deque(maxlen=30)
    queue_fps_interface = deque(maxlen=30)

    trigger_queue = deque(maxlen=60)
    running_flag = [True]

    indices = [
        0, 1,  # 鼻子 x 坐标、y 坐标
        15, 16,  # 左肩膀 x 坐标、y 坐标
        18, 19,  # 右肩膀 x 坐标、y 坐标
    ]

    thread_frame = threading.Thread(target=process_frame, args=(k4a, model, mp_hands,
                                                                queue_image, queue_keypoints, queue_fps_process,
                                                                data_cons, running_flag))
    thread_show = threading.Thread(target=show_frame, args=(queue_image,
                                                            running_flag,
                                                            queue_fps_show))
    thread_interface = threading.Thread(target=interface, args=(data_var, data_result, data_cons,
                                                                queue_keypoints, trigger_queue, queue_fps_interface,
                                                                running_flag, indices))

    thread_frame.start()
    thread_show.start()
    thread_interface.start()

    try:
        while running_flag[0]:
            print('main is running')
            time.sleep(1)
    except KeyboardInterrupt:
        running_flag[0] = False

    thread_frame.join()
    thread_show.join()
    thread_interface.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
