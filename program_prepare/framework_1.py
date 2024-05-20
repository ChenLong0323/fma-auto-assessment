import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
import mediapipe as mp

import cv2
import time
from collections import deque

# For static images:
cap = cv2.VideoCapture(0)



## 初始化
# pyk4a初始化
def initial_k4a():
    # pyk4a初始化
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    return k4a


def initial_pose():
    model = YOLO('../models/yolov8s-pose.pt')
    return model


def fps_cal(fps, frame_count, start_time):
    frame_count += 1
    els_time = time.time() - start_time
    if els_time > 2:
        # 计算FPS
        fps = frame_count / els_time
        start_time = time.time()
        frame_count = 0
    return fps, start_time, frame_count


def main():
    k4a = initial_k4a()
    k4a.start()

    model = initial_pose()

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands


    # FPS 计算
    frame_count = 0
    start_time = time.time()
    fps = 0

    window_width = 150

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as hands:
        while True:
            # 计算FPS
            fps, start_time, frame_count = fps_cal(fps,frame_count, start_time)
            # 获取图像
            capture = k4a.get_capture()
            image = capture.color[:, 640-window_width:640+window_width, :3]  #

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # 检测1：pose
            results_pose = model(image, stream=True, conf=0.7, verbose=False)

            # 检测2：hands
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_hand = hands.process(image)

            for result in results_pose:
                # 绘制1：pose
                image = result.plot()
                print(image.flags.writeable)
                # 绘制2：hands
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results_hand.multi_hand_landmarks:
                    for hand_landmarks in results_hand.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

            image = cv2.flip(image, 1)
            cv2.putText(image, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("rgb", image)

    k4a.stop()


if __name__ == "__main__":
    main()
    cap.release()
