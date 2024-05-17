"""
手部显示为完整关节点数据
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

    model = YOLO('models/yolov8s-pose.pt')

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


def process_frame(k4a, model, mp_hands, frame_queue, window_width, running_flag):
    while running_flag[0]:
        capture = k4a.get_capture()
        image = capture.color[:, 640 - window_width:640 + window_width, :3]
        height, width, _ = image.shape

        results_pose = model(image, stream=True, conf=0.7, verbose=False)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hand = mp_hands.process(image_rgb)

        if not frame_queue.full():
            frame_queue.put((image, results_pose, results_hand))


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

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    try:
        while True:
            if not frame_queue.empty():
                image, results_pose, results_hand = frame_queue.get()

                fps, start_time, frame_count = fps_cal(start_time, frame_count)

                # 绘制姿态
                for result in results_pose:
                    image = result.plot()

                # 绘制手部关键点
                if results_hand.multi_hand_landmarks:
                    for hand_landmarks in results_hand.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                # 翻转图像
                image = cv2.flip(image, 1)

                cv2.putText(image, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("rgb", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                running_flag[0] = False  # 设置标志为False以停止线程
                break
    finally:
        print('1')
        thread.join()  # 确保线程结束
        print('2')
        k4a.stop()
        cv2.destroyAllWindows()
        mp_hands.close()


if __name__ == "__main__":
    main()
