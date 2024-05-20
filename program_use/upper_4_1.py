"""
本项目基于framework_7
"""
import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
import mediapipe as mp
import cv2
import time
import threading
import queue

def initialize():
    """
    初始化设备和模型
    """
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    model = YOLO('../models/yolov8s-pose.pt')

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

def extract_landmark_coordinates(results_hand, height, width):
    """
    提取手部关键点的坐标

    参数:
    results_hand : list
        检测到的手部关键点列表
    height : int
        图像的高度
    width : int
        图像的宽度

    返回:
    x_coordinates : list
        手部关键点的x坐标列表
    y_coordinates : list
        手部关键点的y坐标列表
    """
    if not results_hand:
        return [None, None, None, None], [None, None, None, None]

    right_hand_x = [None, None]
    right_hand_y = [None, None]
    left_hand_x = [None, None]
    left_hand_y = [None, None]

    for hand_landmarks in results_hand:
        x5, x17 = hand_landmarks.landmark[5].x, hand_landmarks.landmark[17].x
        y5, y17 = hand_landmarks.landmark[5].y, hand_landmarks.landmark[17].y

        if x5 < 0.5:
            right_hand_x = [x5, x17]
            right_hand_y = [y5, y17]
        else:
            left_hand_x = [x5, x17]
            left_hand_y = [y5, y17]

    x_coordinates = [(x * width) if x is not None else None for x in (right_hand_x + left_hand_x)]
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
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]
    radius = [10, 5, 10, 5]
    for (x, y), color, radiu in zip(zip(x_coordinates, y_coordinates), colors, radius):
        if x is not None and y is not None:
            point = (int(x), int(y))
            cv2.circle(image, point, radiu, color, -1)

def process_frame(k4a, model, mp_hands, frame_queue, window_width, running_flag):
    while running_flag[0]:
        try:
            capture = k4a.get_capture()
            image = capture.color[:, 640 - window_width:640 + window_width, :3]
            height, width, _ = image.shape

            results_pose = model(image, stream=True, conf=0.7, verbose=False)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_hand = mp_hands.process(image_rgb).multi_hand_landmarks

            x_coordinates, y_coordinates = extract_landmark_coordinates(results_hand, height, width)

            frame_queue.put((image, results_pose, x_coordinates, y_coordinates), timeout=0.1)
        except queue.Full:
            # 在队列满时忽略，以避免阻塞
            continue
        except Exception as e:
            print(f"Error in thread: {e}")
            break

    print("Thread exiting...")

def main():
    k4a, model, mp_hands = initialize()

    frame_queue = queue.Queue(maxsize=1)
    running_flag = [True]  # 使用列表作为可变标志

    window_width = 200

    thread = threading.Thread(target=process_frame, args=(k4a, model, mp_hands, frame_queue, window_width, running_flag))
    thread.start()

    frame_count = 0
    start_time = time.time()
    fps = 0

    try:
        while True:
            if not frame_queue.empty():
                image, results_pose, x_coordinates, y_coordinates = frame_queue.get()

                fps, start_time, frame_count = fps_cal(start_time, frame_count)

                for result in results_pose:
                    image = result.plot()
                draw_circles(image, x_coordinates, y_coordinates)

                # 翻转图像
                image = cv2.flip(image, 1)

                cv2.putText(image, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("rgb", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                running_flag[0] = False  # 设置标志为False以停止线程
                break
    finally:
        k4a.stop()
        cv2.destroyAllWindows()
        mp_hands.close()
        print('1')
        thread.join()  # 确保线程结束
        print('2')

if __name__ == "__main__":
    main()
