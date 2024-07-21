import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
import mediapipe as mp
import cv2
import time
import queue
import numpy as np
import logging


def initialize(model_name='yolov8s'):
    """
    初始化设备和模型

    参数:
    model_name : str
        要加载的YOLO模型名称（默认为'yolov8s'）
    """
    # 初始化Kinect设备
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # 构建模型路径
    model_path = f'../models/{model_name}-pose.pt'
    print(f"Loading model from {model_path}")

    # 初始化YOLO模型
    model = YOLO(model_path)

    # 初始化MediaPipe Hands
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return k4a, model, mp_hands


def fps_cal(start_time, frame_count):
    """
    计算 FPS

    参数:
    start_time : float
        开始计算FPS的时间戳
    frame_count : int
        已处理的帧数

    返回:
    fps : float
        当前的FPS值
    start_time : float
        更新后的时间戳
    frame_count : int
        更新后的帧数
    """
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 2:
        fps = frame_count / elapsed_time
        start_time = time.time()
        frame_count = 0
    else:
        fps = frame_count / elapsed_time
    return fps, start_time, frame_count


def extract_landmark_coordinates(results_hand, height, width, width_det_pose, width_det_hand):
    """
    提取手部关键点的坐标

    参数:
    results_hand : list
        检测到的手部关键点列表
    height : int
        图像的高度
    width : int
        图像的宽度
    width_det_pose : int
        显示窗口的宽度
    width_det_hand : int
        手部检测窗口的宽度

    返回:
    x_coordinates : list
        手部关键点的x坐标列表
    y_coordinates : list
        手部关键点的y坐标列表
    """
    # 初始化手部坐标
    right_hand_x = [None, None, None]
    right_hand_y = [None, None, None]
    left_hand_x = [None, None, None]
    left_hand_y = [None, None, None]

    if not results_hand:
        return right_hand_x + left_hand_x, right_hand_y + left_hand_y

    # 处理单手检测
    if len(results_hand) == 1:
        hand_landmarks = results_hand[0].landmark
        x0, x5, x17 = hand_landmarks[0].x, hand_landmarks[5].x, hand_landmarks[17].x
        y0, y5, y17 = hand_landmarks[0].y, hand_landmarks[5].y, hand_landmarks[17].y

        if x5 < 0.5:  # 假设 x5 < 0.5 为右手
            right_hand_x = [x0, x5, x17]
            right_hand_y = [y0, y5, y17]
        else:  # 否则为左手
            left_hand_x = [x0, x5, x17]
            left_hand_y = [y0, y5, y17]

    # 处理双手检测
    elif len(results_hand) == 2:
        hand_landmarks_l = results_hand[0].landmark
        hand_landmarks_r = results_hand[1].landmark

        x0_l, x5_l, x17_l = hand_landmarks_l[0].x, hand_landmarks_l[5].x, hand_landmarks_l[17].x
        y0_l, y5_l, y17_l = hand_landmarks_l[0].y, hand_landmarks_l[5].y, hand_landmarks_l[17].y

        x0_r, x5_r, x17_r = hand_landmarks_r[0].x, hand_landmarks_r[5].x, hand_landmarks_r[17].x
        y0_r, y5_r, y17_r = hand_landmarks_r[0].y, hand_landmarks_r[5].y, hand_landmarks_r[17].y

        # 比较 x5_l 和 x5_r，确定左右手
        if x5_l < x5_r:
            right_hand_x = [x0_l, x5_l, x17_l]
            right_hand_y = [y0_l, y5_l, y17_l]
            left_hand_x = [x0_r, x5_r, x17_r]
            left_hand_y = [y0_r, y5_r, y17_r]
        else:
            right_hand_x = [x0_r, x5_r, x17_r]
            right_hand_y = [y0_r, y5_r, y17_r]
            left_hand_x = [x0_l, x5_l, x17_l]
            left_hand_y = [y0_l, y5_l, y17_l]

    x_coordinates = [(x * width + width_det_pose - width_det_hand) if x is not None else None for x in (right_hand_x + left_hand_x)]
    y_coordinates = [(y * height) if y is not None else None for y in (right_hand_y + left_hand_y)]

    return x_coordinates, y_coordinates


def draw_circles(image, x_coordinates, y_coordinates):
    """
    在图像上绘制圆圈

    参数:
    image : numpy.ndarray
        要绘制的图像
    x_coordinates : list
        手部关键点的x坐标列表
    y_coordinates : list
        手部关键点的y坐标列表
    """
    colors = [(0, 0, 255), (0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 0), (0, 255, 0)]
    radius = [10, 5, 5, 10, 5, 5]
    for (x, y), color, radiu in zip(zip(x_coordinates, y_coordinates), colors, radius):
        if x is not None and y is not None:
            point = (int(x), int(y))
            cv2.circle(image, point, radiu, color, -1)


def func_frame_cap(k4a, model, mp_hands,
                   queue_image, queue_keypoints, queue_fps_process,
                   data_cons, flag_run):
    # 先判断是否运行
    while flag_run[0]:
        try:
            # 判断需要手  并且  线程2初始化成功
            # 这里是 0 0 不需要
            if not flag_run[1] and not flag_run[2]:
                capture = k4a.get_capture()
                image = capture.color[:, 640 - data_cons.WIDTH_DET_POSE:640 + data_cons.WIDTH_DET_POSE, :3]
                results_pose = model(image, stream=True, conf=0.7, verbose=False)

                for result in results_pose:
                    keypoints = np.ravel(result.keypoints.data[0].tolist())
                    image_show = result.plot()
                    image_show = cv2.flip(image_show, 1)
                    time.sleep(0.1)
                    # queue_keypoints.put(keypoints, timeout=0.1)
                    queue_image.put(image_show, timeout=0.1)

            # 这里是 1 1 需要
            elif flag_run[1] and flag_run[2]:
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

                    # queue_keypoints.put((keypoints, x_coordinates, y_coordinates), timeout=0.1)
                    queue_image.put(image_show, timeout=0.1)
            # -------------------------------------------------------------
            queue_fps_process.append(time.time())
            if len(queue_fps_process) > 30:
                queue_fps_process.popleft()
            if len(queue_fps_process) > 1:
                fps = len(queue_fps_process) / (queue_fps_process[-1] - queue_fps_process[0])
                logging.info(f"FPS_cap: {fps:.2f}")

        except queue.Full:
            logging.warning("Queue is full in func_frame_process")
            continue
        except Exception as e:
            logging.error(f"Error in func_frame_process: {e}")
            break



def func_frame_show(queue_image, queue_fps_show, flag_run):
    while flag_run[0]:
        try:
            image_show = queue_image.get()
            cv2.imshow("rgb", image_show)

            print(f"There are {queue_image.qsize()} elements in the queue.")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                flag_run[0] = 0
                break
            # -------------------------------------------------------------
            # 计算FPS
            queue_fps_show.append(time.time())
            if len(queue_fps_show) > 30:
                queue_fps_show.popleft()

            if len(queue_fps_show) > 1:
                fps = len(queue_fps_show) / (queue_fps_show[-1] - queue_fps_show[0])
                logging.info(f"FPS_show: {fps:.2f}")
        # thread内容
        except queue.Full:
            logging.warning("Queue is full in show_frame")
            continue
        except Exception as e:
            logging.error(f"Error in show_frame: {e}")
            break


def func_frame_process():
    pass


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
