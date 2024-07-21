"""
线程异常处理：在主线程中捕获子线程异常，这样可以更方便地调试和管理线程状态。
日志记录：使用 Python 的 logging 模块替代 print 语句，这样可以更好地管理日志级别和输出格式。
线程关闭：确保在 running_flag 被设置为 False 后，所有子线程都能优雅地退出。
代码结构优化：分离函数逻辑，提升代码的可读性和复用性。
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
                  constants, running_flag):
    while running_flag[0]:
        try:
            capture = k4a.get_capture()
            image = capture.color[:, 640 - constants.WIDTH_DET_POSE:640 + constants.WIDTH_DET_POSE, :3]
            image_hand = cv2.cvtColor(
                image[:, constants.WIDTH_DET_POSE - constants.WIDTH_DET_HAND:constants.WIDTH_DET_POSE + constants.WIDTH_DET_HAND, :3],
                cv2.COLOR_BGR2RGB)
            height, width, _ = image_hand.shape

            results_pose = model(image, stream=True, conf=0.7, verbose=False)
            results_hand = mp_hands.process(image_hand).multi_hand_landmarks

            x_coordinates, y_coordinates = extract_landmark_coordinates(results_hand, height, width,
                                                                        constants.WIDTH_DET_POSE,
                                                                        constants.WIDTH_DET_HAND)

            for result in results_pose:
                keypoints = np.ravel(result.keypoints.data[0].tolist())
                image_show = result.plot()
                draw_circles(image_show, x_coordinates, y_coordinates)
                image_show = cv2.flip(image_show, 1)

                queue_keypoints.put((keypoints, x_coordinates, y_coordinates), timeout=0.1)
                queue_image.put(image_show, timeout=0.1)

                queue_fps_process.append(time.time())
                if len(queue_fps_process) > 30:
                    queue_fps_process.popleft()

                if len(queue_fps_process) > 1:
                    fps = len(queue_fps_process) / (queue_fps_process[-1] - queue_fps_process[0])
                    logging.info(f"FPS_process: {fps:.2f}")

        except queue.Full:
            logging.warning("Queue is full in process_frame")
            continue
        except Exception as e:
            logging.error(f"Error in process_frame: {e}")
            break


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


def interface(hand_status, result_data, constants, running_flag, queue_keypoints, trigger_queue, queue_fps_interface, indices):
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
                logging.info(f"Handedness: {hand_status.flag_handedness}, Left: {hand_status.flag_left}, Right: {hand_status.flag_right}")

                if hand_status.flag_handedness == 0:
                    if ext[5] > y_coordinates[0]:
                        trigger_queue.append(1)
                        if sum(trigger_queue) > constants.THRESHOLD:
                            hand_status.flag_handedness = 2
                            trigger_queue.clear()
                    elif ext[3] > y_coordinates[3]:
                        trigger_queue.append(1)
                        if sum(trigger_queue) > constants.THRESHOLD:
                            trigger_queue.clear()
                            hand_status.flag_handedness = 1
                    else:
                        trigger_queue.append(0)

                elif hand_status.flag_handedness == 1:
                    handle_left_hand(hand_status, result_data, constants, x_coordinates, y_coordinates, ext, trigger_queue)

                elif hand_status.flag_handedness == 2:
                    handle_right_hand(hand_status, result_data, constants, x_coordinates, y_coordinates, ext, trigger_queue)

        except queue.Full:
            logging.warning("Queue is full in interface")
            continue
        except Exception as e:
            logging.error(f"Error in interface: {e}")
            break


def handle_left_hand(hand_status, result_data, constants, x_coordinates, y_coordinates, ext, trigger_queue):
    if hand_status.flag_left == 0:
        logging.info("Handling left hand")
        if abs((ext[0] - x_coordinates[3])) > 6 * constants.WIN_W:
            trigger_queue.append(1)
            if sum(trigger_queue) > constants.THRESHOLD:
                trigger_queue.clear()
                hand_status.flag_left = 1
                if hand_status.count == 0:
                    hand_status.move_start_time = time.time()

                if hand_status.count != 0:
                    result_data.min_list_l.append(hand_status.dis)
                hand_status.dis = 1000

                if hand_status.count > 4:
                    hand_status.flag_handedness = 2
                    hand_status.count = 0
                    hand_status.count_sides += 1
                    result_data.time_left = time.time() - hand_status.move_start_time
                    return

        else:
            trigger_queue.append(0)

    elif hand_status.flag_left == 1:
        if ext[2] > x_coordinates[3]:
            hand_status.dis1 = abs(ext[0] - x_coordinates[3]) + abs(ext[1] - y_coordinates[3])
            if hand_status.dis > hand_status.dis1:
                hand_status.dis = hand_status.dis1

            trigger_queue.append(1)
            if sum(trigger_queue) > constants.THRESHOLD:
                trigger_queue.clear()
                hand_status.count += 1
                hand_status.flag_left = 0

        else:
            trigger_queue.append(0)


def handle_right_hand(hand_status, result_data, constants, x_coordinates, y_coordinates, ext, trigger_queue):
    if hand_status.flag_right == 0:
        logging.info("Handling right hand")
        if abs((ext[0] - x_coordinates[0])) > 6 * constants.WIN_W:
            trigger_queue.append(1)
            if sum(trigger_queue) > constants.THRESHOLD:
                trigger_queue.clear()
                hand_status.flag_right = 1
                if hand_status.count == 0:
                    hand_status.move_start_time = time.time()

                if hand_status.count != 0:
                    result_data.min_list_r.append(hand_status.dis)
                hand_status.dis = 1000
        else:
            trigger_queue.append(0)

    elif hand_status.flag_right == 1:
        if ext[4] < x_coordinates[0]:
            hand_status.dis1 = abs(ext[0] - x_coordinates[0]) + abs(ext[1] - y_coordinates[0])
            if hand_status.dis > hand_status.dis1:
                hand_status.dis = hand_status.dis1

            trigger_queue.append(1)
            if sum(trigger_queue) > constants.THRESHOLD:
                trigger_queue.clear()
                hand_status.count += 1
                hand_status.flag_right = 0

                if hand_status.count > 4:
                    hand_status.flag_handedness = 1
                    hand_status.count = 0
                    hand_status.count_sides += 1
                    result_data.time_right = time.time() - hand_status.move_start_time
                    result_data.min_list_r.append(hand_status.dis)
                    hand_status.dis = 1000


def main():
    k4a, model, mp_hands = initialize()
    constants = Constants()
    hand_status = HandStatus()
    result_data = ResultData()

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
                                                                constants, running_flag))
    thread_show = threading.Thread(target=show_frame, args=(queue_image,
                                                            running_flag,
                                                            queue_fps_show))
    thread_interface = threading.Thread(target=interface, args=(hand_status, result_data, constants, running_flag,
                                                                queue_keypoints, trigger_queue, queue_fps_interface, indices))

    thread_frame.start()
    thread_show.start()
    thread_interface.start()

    try:
        while running_flag[0]:
            print('123')
            time.sleep(0.1)
    except KeyboardInterrupt:
        running_flag[0] = False

    thread_frame.join()
    thread_show.join()
    thread_interface.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
