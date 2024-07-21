import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
import mediapipe as mp

import cv2
import time


# 初始化 pyk4a
def initial_k4a():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    return k4a


# 初始化 YOLO 模型
def initial_pose():
    model = YOLO('../../models/yolov8s-pose.pt')
    return model


# 计算 FPS
def fps_cal(fps, frame_count, start_time):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 2:
        fps = frame_count / elapsed_time
        start_time = time.time()
        frame_count = 0
    return fps, start_time, frame_count


# 提取手部关键点的坐标
def extract_landmark_coordinates(results_hand, image):
    if not results_hand or len(results_hand) < 2:
        return [], []

    landmarks_l = results_hand[0].landmark
    landmarks_r = results_hand[1].landmark

    # 判断哪个手部 landmark 更靠左
    if landmarks_r[5].x < landmarks_l[5].x:
        left_index = 1
    else:
        left_index = 0

    # 根据左右手的顺序调整坐标
    x_coordinates = [
        landmarks_r[5].x, landmarks_r[17].x,
        landmarks_l[5].x, landmarks_l[17].x
    ]
    y_coordinates = [
        landmarks_r[5].y, landmarks_r[17].y,
        landmarks_l[5].y, landmarks_l[17].y
    ]

    # 根据图像宽度和高度调整坐标
    height, width, _ = image.shape
    x_coordinates = [x * width for x in x_coordinates]
    y_coordinates = [y * height for y in y_coordinates]

    # 如果右手在左手之前，则交换左右手的坐标
    if left_index == 0:
        x_coordinates = [x_coordinates[2], x_coordinates[3], x_coordinates[0], x_coordinates[1]]
        y_coordinates = [y_coordinates[2], y_coordinates[3], y_coordinates[0], y_coordinates[1]]

    return x_coordinates, y_coordinates


# 在图像上绘制圆圈
def draw_circles(image, x_coordinates, y_coordinates):
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)]  # 红, 黄, 蓝, 绿
    radius = [5, 10, 5, 10]
    for (x, y), color, radiu in zip(zip(x_coordinates, y_coordinates), colors, radius):
        point = (int(x), int(y))
        cv2.circle(image, point, radiu, color, -1)


def main():
    k4a = initial_k4a()
    k4a.start()

    model = initial_pose()

    mp_hands = mp.solutions.hands

    # FPS 计算
    frame_count = 0
    start_time = time.time()
    fps = 0

    window_width = 200

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while True:
            # 计算 FPS
            fps, start_time, frame_count = fps_cal(fps, frame_count, start_time)

            # 获取图像
            capture = k4a.get_capture()
            image = capture.color[:, 640 - window_width:640 + window_width, :3]

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # 检测姿态
            results_pose = model(image, stream=True, conf=0.7, verbose=False)

            # 检测手部
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_hand = hands.process(image_rgb).multi_hand_landmarks

            x_coordinates, y_coordinates = extract_landmark_coordinates(results_hand, image)

            # 绘制结果
            for result in results_pose:
                image = result.plot()  # Assuming results_pose returns a list
            draw_circles(image, x_coordinates, y_coordinates)

            # 显示 FPS
            # image = cv2.flip(image, 1)
            cv2.putText(image, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("rgb", image)

    k4a.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
