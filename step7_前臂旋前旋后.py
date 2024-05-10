import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
import mediapipe as mp

import cv2
import time
from collections import deque

# For static images:
cap = cv2.VideoCapture(0)

# FPS 计算
frame_count = 0
start_time = time.time()
fps = 0


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
    model = YOLO('models/yolov8s-pose.pt')
    return model


def main():
    k4a = initial_k4a()
    k4a.start()

    model = initial_pose()

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # while True:
    #     capture = k4a.get_capture() #  这里获取的RGB已经是RGB格式了
    #     frame = capture.color[:, 430:850, :3]  #
    #     cv2.imshow("rgb", frame)
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as hands:
        while True:
            capture = k4a.get_capture()
            frame = capture.color[:, 430:850, :3]  #
            cv2.imshow("rgb", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
