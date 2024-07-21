import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
import mediapipe as mp
import cv2
import time
import queue
import numpy as np
import logging
import func_ActionDetector


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
    elif len(results_hand) == 1:
        hand_landmarks = results_hand[0].landmark
        x0, x5, x17 = hand_landmarks[0].x, hand_landmarks[5].x, hand_landmarks[17].x
        y0, y5, y17 = hand_landmarks[0].y, hand_landmarks[5].y, hand_landmarks[17].y
        # FIXME: 这里的判断条件可能需要调整，比如引入nose坐标，根据nose坐标判断左右手
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


def capture_frame(k4a, data_cons):
    capture = k4a.get_capture()
    image = capture.color[:, 640 - data_cons.WIDTH_DET_POSE:640 + data_cons.WIDTH_DET_POSE, :3]
    return image


def process_pose(image, model, data_cons):
    results_pose = model(image, stream=True, conf=0.7, verbose=False)
    keypoints, image_show, boxes = None, None, None
    for result in results_pose:
        keypoints = result.keypoints.data[0].tolist()
        boxes = result.boxes.data[0].tolist()
        image_show = result.plot()
    return keypoints, image_show, boxes


def process_hand(image, mp_hands, data_cons):
    image_hand = cv2.cvtColor(
        image[:, data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND:data_cons.WIDTH_DET_POSE + data_cons.WIDTH_DET_HAND, :3],
        cv2.COLOR_BGR2RGB)
    height, width, _ = image_hand.shape
    results_hand = mp_hands.process(image_hand).multi_hand_landmarks
    x_coordinates, y_coordinates = extract_landmark_coordinates(results_hand, height, width,
                                                                data_cons.WIDTH_DET_POSE,
                                                                data_cons.WIDTH_DET_HAND)
    return x_coordinates, y_coordinates


def func_frame_cap(k4a, model, mp_hands,
                   queue_frame_image, queue_keypoints, queue_fps_process,
                   data_cons, flag_running, action_idx):
    while flag_running.is_set():
        try:
            image = capture_frame(k4a, data_cons)
            keypoints, image_show, boxes = process_pose(image, model, data_cons)
            x_coordinates, y_coordinates = [], []

            if data_cons.hand_flag[action_idx]:
                x_coordinates, y_coordinates = process_hand(image, mp_hands, data_cons)
                draw_circles(image_show, x_coordinates, y_coordinates)

            image_show = cv2.flip(image_show, 1)

            try:
                queue_frame_image.put_nowait(image_show)
            except queue.Full:
                continue  # 直接继续，不记录日志以减少影响

            if keypoints is not None:
                try:
                    queue_keypoints.put_nowait((keypoints, boxes, x_coordinates, y_coordinates))
                except queue.Full:
                    continue  # 直接继续，不记录日志以减少影响

            queue_fps_process.append(time.time())
            if len(queue_fps_process) > 30:
                queue_fps_process.popleft()
            if len(queue_fps_process) > 1:
                fps = len(queue_fps_process) / (queue_fps_process[-1] - queue_fps_process[0])
                logging.info(f"FPS_cap: {fps:.2f}")
        except Exception as e:
            logging.error(f"Error in func_frame_cap: {str(e)}")
            break


def func_frame_show(queue_frame_image, queue_fps_show, flag_running):
    while flag_running.is_set():
        try:
            image_show = queue_frame_image.get_nowait()
            cv2.imshow("rgb", image_show)

            print(f"There are {queue_frame_image.qsize()} elements in the queue.")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                flag_running.clear()
                break

            queue_fps_show.append(time.time())
            if len(queue_fps_show) > 30:
                queue_fps_show.popleft()

            if len(queue_fps_show) > 1:
                fps = len(queue_fps_show) / (queue_fps_show[-1] - queue_fps_show[0])
                logging.info(f"FPS_show: {fps:.2f}")
        except queue.Empty:
            logging.warning("Queue is empty in show_frame")
            time.sleep(0.05)  # 添加短暂的休眠，避免忙等待
        except Exception as e:
            logging.error(f"Error in show_frame: {str(e)}")
            break


def func_inter(flag_running, action_idx, action_manager,
               data_cons, data_var, data_result,
               queue_keypoints, queue_fps_interface,
               trigger_queue, trigger_queue_overtime):
    while flag_running.is_set():
        try:
            queue_fps_interface.append(time.time())
            if len(queue_fps_interface) > 30:
                queue_fps_interface.popleft()

            if len(queue_fps_interface) > 1:
                fps = len(queue_fps_interface) / (queue_fps_interface[-1] - queue_fps_interface[0])
                logging.info(f"FPS_interface: {fps:.2f}")

            keypoints, boxes, x_coordinates, y_coordinates = queue_keypoints.get()

            # 使用动作管理器根据 action_idx 动态选择动作检测器
            result = action_manager.detect_action(action_idx,
                                                  data_cons, data_var, data_result,
                                                  keypoints, boxes, x_coordinates, y_coordinates,
                                                  trigger_queue, trigger_queue_overtime)
            if isinstance(result, int):
                action_idx = result
            else:
                logging.warning(result)

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in func_inter: {str(e)}")
            break
