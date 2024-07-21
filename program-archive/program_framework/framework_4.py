"""
初始化部分的优化：将所有初始化代码放在一起，形成一个专门的初始化函数，并且尽可能减少初始化时的冗余操作
使用上下文管理器管理资源：在main函数中使用上下文管理器来管理资源，以确保在异常情况下也能正确释放资源
优化FPS计算函数：你可以将FPS计算的初始时间放在外部，并且直接返回新的时间戳和帧数。
添加注释和文档：保持你的注释和文档，这样有助于其他开发者理解代码。
"""

import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
import mediapipe as mp
import cv2
import time

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

    model = YOLO('../../models/yolov8l-pose.pt')

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

def main():
    k4a, model, mp_hands = initialize()

    frame_count = 0
    start_time = time.time()
    fps = 0

    window_width = 200

    try:
        while True:

            capture = k4a.get_capture()
            image = capture.color[:, 640 - window_width:640 + window_width, :3]
            height, width, _ = image.shape

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            results_pose = model(image, stream=True, conf=0.7, verbose=False)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_hand = mp_hands.process(image_rgb).multi_hand_landmarks

            x_coordinates, y_coordinates = extract_landmark_coordinates(results_hand, height, width)
            print(x_coordinates, y_coordinates)

            for result in results_pose:
                image = result.plot()
            draw_circles(image, x_coordinates, y_coordinates)

            cv2.putText(image, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("rgb", image)
            fps, start_time, frame_count = fps_cal(start_time, frame_count)
    finally:
        k4a.stop()
        cv2.destroyAllWindows()
        mp_hands.close()

if __name__ == "__main__":
    main()
