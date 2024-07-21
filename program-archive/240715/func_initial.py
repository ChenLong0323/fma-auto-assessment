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


def draw_circles(image, hands_coordinates):
    """
    在图像上绘制圆圈

    参数:
    image : numpy.ndarray
        要绘制的图像
    hands_coordinates : list
        手部关键点的坐标列表，包含左右手的关键点坐标
    """
    colors = [(255, 0, 0), (0, 255, 255), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 0, 255)]
    radius = [10, 5, 5, 10, 5, 5]

    for (coord, color, radiu) in zip(hands_coordinates, colors, radius):
        x, y = coord
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
        boxes = None
        # boxes = result.boxes.data[0].tolist()
        image_show = result.plot()
    return keypoints, image_show, boxes


def process_hand(image, mp_hands, data_cons, keypoints):
    image_hand = cv2.cvtColor(
        image[:, data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND:data_cons.WIDTH_DET_POSE + data_cons.WIDTH_DET_HAND, :3],
        cv2.COLOR_BGR2RGB)
    height, width, _ = image_hand.shape
    results_hand = mp_hands.process(image_hand).multi_hand_landmarks

    # 提取左右手手腕坐标
    lx, ly, rx, ry = None, None, None, None
    hands_coordinates = [[None, None], [None, None], [None, None], [None, None], [None, None], [None, None]]

    if len(keypoints) > 9:  # "left_wrist"
        lx, ly, _ = keypoints[9]
    if len(keypoints) > 10:  # "right_wrist"
        rx, ry, _ = keypoints[10]

    # 如果没有检测到手
    if not results_hand:
        return hands_coordinates
    # 检测到手
    else:
        hand_landmarks = results_hand[0].landmark
        x0, x5, x17 = hand_landmarks[0].x, hand_landmarks[5].x, hand_landmarks[17].x
        y0, y5, y17 = hand_landmarks[0].y, hand_landmarks[5].y, hand_landmarks[17].y
        # 去除归一化
        x0, x5, x17 = (x0 * width + data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND,
                       x5 * width + data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND,
                       x17 * width + data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND)
        y0, y5, y17 = y0 * height, y5 * height, y17 * height
        # 计算手腕与手的距离并分配手
        if lx is not None and ly is not None and rx is not None and ry is not None:
            dis_l = (x0 - lx) ** 2 + (y0 - ly) ** 2
            dis_r = (x0 - rx) ** 2 + (y0 - ry) ** 2

            if dis_l < dis_r:
                hands_coordinates[0] = [x0, y0]
                hands_coordinates[1] = [x5, y5]
                hands_coordinates[2] = [x17, y17]
                if len(results_hand) == 2:
                    x0, x5, x17 = results_hand[1].landmark[0].x, results_hand[1].landmark[5].x, results_hand[1].landmark[17].x
                    y0, y5, y17 = results_hand[1].landmark[0].y, results_hand[1].landmark[5].y, results_hand[1].landmark[17].y
                    x0, x5, x17 = (x0 * width + data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND,
                                   x5 * width + data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND,
                                   x17 * width + data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND)
                    y0, y5, y17 = y0 * height, y5 * height, y17 * height
                    hands_coordinates[3] = [x0, y0]
                    hands_coordinates[4] = [x5, y5]
                    hands_coordinates[5] = [x17, y17]
            else:
                hands_coordinates[3] = [x0, y0]
                hands_coordinates[4] = [x5, y5]
                hands_coordinates[5] = [x17, y17]
                if len(results_hand) == 2:
                    x0, x5, x17 = results_hand[1].landmark[0].x, results_hand[1].landmark[5].x, results_hand[1].landmark[17].x
                    y0, y5, y17 = results_hand[1].landmark[0].y, results_hand[1].landmark[5].y, results_hand[1].landmark[17].y
                    x0, x5, x17 = (x0 * width + data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND,
                                   x5 * width + data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND,
                                   x17 * width + data_cons.WIDTH_DET_POSE - data_cons.WIDTH_DET_HAND)
                    y0, y5, y17 = y0 * height, y5 * height, y17 * height
                    hands_coordinates[0] = [x0, y0]
                    hands_coordinates[1] = [x5, y5]
                    hands_coordinates[2] = [x17, y17]
        # 只有左手腕
        elif lx is not None and ly is not None:
            hands_coordinates[0] = [x0, y0]
            hands_coordinates[1] = [x5, y5]
            hands_coordinates[2] = [x17, y17]

        # 只有右手腕
        elif rx is not None and ry is not None:
            hands_coordinates[3] = [x0, y0]
            hands_coordinates[4] = [x5, y5]
            hands_coordinates[5] = [x17, y17]

        # 没有手腕信息，按检测顺序分配
        else:
            if hands_coordinates[3] == [None, None]:
                hands_coordinates[3] = [x0, y0]
                hands_coordinates[4] = [x5, y5]
                hands_coordinates[5] = [x17, y17]
            else:
                hands_coordinates[0] = [x0, y0]
                hands_coordinates[1] = [x5, y5]
                hands_coordinates[2] = [x17, y17]
    return hands_coordinates


def func_frame_cap(k4a, model, mp_hands,
                   queue_frame_image, queue_keypoints, queue_fps_process,
                   data_cons, flag_running, action_idx):
    while flag_running.is_set():
        try:

            image = capture_frame(k4a, data_cons)
            keypoints, image_show, boxes = process_pose(image, model, data_cons)
            hands_coordinates = None

            # 提取手腕坐标
            if data_cons.hand_flag[action_idx]:
                hands_coordinates = process_hand(image, mp_hands, data_cons, keypoints)
                draw_circles(image_show, hands_coordinates)

            if keypoints is not None:
                try:
                    queue_keypoints.put_nowait((keypoints, boxes, hands_coordinates, image_show))
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
               queue_keypoints, queue_fps_interface, queue_frame_image,
               trigger_queue, trigger_queue_overtime):
    while flag_running.is_set():
        try:
            queue_fps_interface.append(time.time())
            if len(queue_fps_interface) > 30:
                queue_fps_interface.popleft()

            if len(queue_fps_interface) > 1:
                fps = len(queue_fps_interface) / (queue_fps_interface[-1] - queue_fps_interface[0])
                logging.info(f"FPS_interface: {fps:.2f}")

            keypoints, boxes, hands_coordinates, image_show1 = queue_keypoints.get()

            # 使用动作管理器根据 action_idx 动态选择动作检测器
            action_idx, image_show = action_manager.detect_action(action_idx,
                                                                  data_cons, data_var, data_result,
                                                                  keypoints, boxes, hands_coordinates, image_show1,
                                                                  trigger_queue, trigger_queue_overtime)
            cv2.line(image_show, (100, 100), (100, 300), (0, 255, 0), 2)
            cv2.line(image_show, (200, 100), (200, 500), (255, 0, 0), 2)
            image_show = cv2.flip(image_show, 1)
            try:
                queue_frame_image.put_nowait(image_show)
            except queue.Full:
                continue  # 直接继续，不记录日志以减少影响

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in func_inter: {str(e)}")
            break
