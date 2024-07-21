"""
本项目基于framework_9
在上一个项目的基础上，增加了交互部分
"""
import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
import mediapipe as mp
import cv2
import time
import threading
import queue
import numpy as np
from collections import deque


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

    model = YOLO('../models/yolov8m-pose.pt')

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


def process_frame(k4a, model, mp_hands, frame_queue,fps_queue, width_det_pose, width_det_hand, running_flag):
    while running_flag[0]:
        try:
            capture = k4a.get_capture()
            image = capture.color[:, 640 - width_det_pose:640 + width_det_pose, :3]
            image_hand = cv2.cvtColor(image[:, width_det_pose - width_det_hand:width_det_pose + width_det_hand, :3], cv2.COLOR_BGR2RGB)
            height, width, _ = image_hand.shape

            results_pose = model(image, stream=True, conf=0.7, verbose=False)

            results_hand = mp_hands.process(image_hand).multi_hand_landmarks

            x_coordinates, y_coordinates = extract_landmark_coordinates(results_hand, height, width, width_det_pose, width_det_hand)

            frame_queue.put((image, results_pose, x_coordinates, y_coordinates), timeout=0.1)
            fps_queue.append(time.time())
            if len(fps_queue) > 30:
                fps_queue.popleft()

            # 确保队列长度大于1以避免除零错误
            if len(fps_queue) > 1:
                fps = len(fps_queue) / (fps_queue[-1] - fps_queue[0])
                print(f"FPS_thread: {fps:.2f}")
        except queue.Full:
            # 在队列满时忽略，以避免阻塞
            print("full")
            continue
        except Exception as e:
            print(f"Error in thread: {e}")
            break

    print("Thread exiting...")


def main():
    k4a, model, mp_hands = initialize()

    frame_queue = queue.Queue(maxsize=1)
    running_flag = [True]  # 使用列表作为可变标志

    width_det_pose = 300  # 用于姿势检测的图像宽度，同时也是实际画布的绘制宽度
    width_det_hand = 300  # 用于手部检测的图像宽度

    # 基本参数
    width_draw_shoulder = 52  # 两肩宽度，用于绘制定位线
    width_draw_hip = 37  # 两髋宽度，用于绘制定位线
    width_draw_sit = 115  # 用于绘制定位线

    # 检验参数
    win_h = 30
    win_w = 20
    threshold = 30
    fps_queue = deque(maxlen=30)  # 用于计算FPS的时间戳队列
    thread = threading.Thread(target=process_frame,
                              args=(k4a, model, mp_hands, frame_queue,fps_queue,
                                    width_det_pose, width_det_hand, running_flag))
    thread.start()

    frame_count = 0
    start_time = time.time()
    fps = 0

    # 初始化 trigger，长度为100，初始空白
    trigger = deque(maxlen=60)
    '''定制部分'''
    # 判断左右手状态
    flag_handedness = 0
    # 在l循环中使用的flag
    flag_l = 0
    # 在r循环中使用的flag
    flag_r = 0

    # 计算左右手循环次数
    count_sides = 0

    # 计算完成项目次数
    count = 0

    # 左右侧移动时间
    move_time_l = 0
    min_list_l = []
    move_time_r = 0
    min_list_r = []
    # 从keypoints中提取需要的点
    indices = [
        0, 1,  # 鼻子 x 坐标、y 坐标
        15, 16,  # 左肩膀 x 坐标、y 坐标
        18, 19,  # 右肩膀 x 坐标、y 坐标
    ]
    try:
        while True:
            if not frame_queue.empty():
                image, results_pose, x_coordinates, y_coordinates = frame_queue.get()

                fps, start_time, frame_count = fps_cal(start_time, frame_count)

                for result in results_pose:
                    keypoints = np.ravel(result.keypoints.data[0].tolist())
                    image = result.plot()

                '''定制部分'''
                if len(keypoints) > max(indices) and None not in x_coordinates and None not in y_coordinates:
                    ext = keypoints[indices]

                    # 寻找惯用手，先检测惯用手
                    if flag_handedness == 0:
                        # 右侧
                        cv2.rectangle(image,
                                      (width_det_pose - width_draw_shoulder, int(ext[5])),
                                      (width_det_pose - width_draw_shoulder - 40, int(ext[5]) - 60),
                                      (0, 255, 255), 2)

                        # 左侧
                        cv2.rectangle(image,
                                      (width_det_pose + width_draw_shoulder, int(ext[3])),
                                      (width_det_pose + width_draw_shoulder + 40, int(ext[3]) - 60),
                                      (0, 255, 255), 2)

                        # 惯用手检测：右侧
                        if ext[5] > y_coordinates[0]:
                            trigger.append(1)
                            if sum(trigger) > threshold:
                                flag_handedness = 2
                                # 触发flag_handedness后，清空trigger
                                trigger.clear()
                        #
                        elif ext[3] > y_coordinates[3]:
                            trigger.append(1)
                            if sum(trigger) > threshold:
                                # 触发flag_handedness后，清空trigger
                                trigger.clear()
                                # flag_handedness值改变
                                flag_handedness = 1
                        else:
                            trigger.append(0)
                        print(sum(trigger))
                        print('flag_handedness =', flag_handedness)


                    # 左侧检测
                    elif flag_handedness == 1:
                        if flag_l == 0:
                            # 交互绘制:左侧box
                            cv2.line(image,
                                     (width_det_pose + 4 * width_draw_shoulder, 240),
                                     (width_det_pose + 4 * width_draw_shoulder, 380),
                                     (0, 255, 255), 2)
                            # 左手远离鼻子
                            if abs((ext[0] - x_coordinates[3])) > 6 * win_w:
                                trigger.append(1)
                                if sum(trigger) > threshold:
                                    # 触发flag后，清空trigger
                                    trigger.clear()
                                    # flag值改变
                                    flag_l = 1
                                    if count == 0:
                                        move_start_time = time.time()

                                    # 第二轮循环记录上一轮最小值
                                    if count != 0:
                                        min_list_l.append(dis)
                                    dis = 1000

                                    # 指鼻5次后，换边
                                    if count > 4:
                                        flag_handedness = 2
                                        count = 0  # 重置count，单侧完成次数清零
                                        count_sides += 1
                                        move_time_l = time.time() - move_start_time
                                        continue

                            else:
                                trigger.append(0)

                        # 开始交互
                        elif flag_l == 1:
                            # 绘制左侧目标line
                            # 如果左肩x坐标大于左手x坐标（左手在左肩内侧）
                            if ext[2] > x_coordinates[3]:

                                # 整合运动过程中，记录最小值
                                dis1 = (abs(ext[0] - x_coordinates[3]) +
                                        abs(ext[1] - y_coordinates[3]))  # 可以展示到屏幕上
                                if dis > dis1:
                                    dis = dis1

                                trigger.append(1)
                                if sum(trigger) > threshold:
                                    # 触发flag后，清空trigger
                                    trigger.clear()
                                    # flag值改变
                                    count += 1
                                    flag_l = 0


                            else:
                                trigger.append(0)

                    # 右侧检测
                    elif flag_handedness == 2:
                        if flag_r == 0:
                            # 交互绘制:左侧box
                            cv2.line(image,
                                     (width_det_pose - 4 * width_draw_shoulder, 240),
                                     (width_det_pose - 4 * width_draw_shoulder, 380),
                                     (0, 255, 255), 2)
                            # 如果右手手腕x坐标远离鼻子x坐标
                            if abs((ext[0] - x_coordinates[0])) > 6 * win_w:
                                trigger.append(1)
                                if sum(trigger) > threshold:
                                    # 触发flag后，清空trigger
                                    trigger.clear()
                                    # flag值改变
                                    flag_r = 1
                                    if count == 0:
                                        move_start_time = time.time()

                                    # 第二轮循环记录上一轮最小值
                                    if count != 0:
                                        min_list_r.append(dis)
                                    dis = 1000
                            else:
                                trigger.append(0)

                        # 开始交互
                        elif flag_r == 1:
                            # 如果右肩x坐标小于右手x坐标（右手在右肩内侧）
                            if ext[4] < x_coordinates[0]:
                                # 本动作特定：记录最小值
                                dis1 = (abs(ext[0] - x_coordinates[0]) +
                                        abs(ext[1] - y_coordinates[0]))  # 可以展示到屏幕上
                                if dis > dis1:
                                    dis = dis1

                                trigger.append(1)
                                if sum(trigger) > threshold:
                                    # 触发flag后，清空trigger
                                    trigger.clear()
                                    # flag值改变
                                    count += 1
                                    flag_r = 0

                                    # 指鼻5次后，换边
                                    if count > 4:
                                        flag_handedness = 1
                                        count = 0  # 重置count，单侧完成次数清零
                                        count_sides += 1
                                        move_time_r = time.time() - move_start_time
                                        min_list_r.append(dis)
                                        dis = 1000


                            else:
                                trigger.append(0)
                '''定制部分结束'''

                draw_circles(image, x_coordinates, y_coordinates)

                '''绘制部分'''
                image = cv2.flip(image, 1)  # 翻转图像
                print(f"FPS_while: {fps:.2f}")
                # 绘制文字
                # cv2.putText(image, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                # 绘制box，绘制四个点
                cv2.rectangle(image, (width_det_pose - width_draw_sit, 180), (width_det_pose + width_draw_sit, 710), (0, 255, 0), 2)
                # 绘制两肩连线
                cv2.line(image, (width_det_pose + width_draw_shoulder, 295), (width_det_pose - width_draw_shoulder, 295), (0, 255, 0), 2)
                # 绘制两臀连线
                cv2.line(image, (width_det_pose + width_draw_hip, 422), (width_det_pose - width_draw_hip, 422), (0, 255, 0), 2)

                # 绘制FPS
                cv2.putText(image, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                # 绘制trigger长度
                cv2.putText(image, f"trigger:{sum(trigger)}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"count:{count}", (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                # 将左右侧时间绘制
                cv2.putText(image, f"left_time:{move_time_l:.2f}", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"right_time:{move_time_r:.2f}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                # 将左右侧最小值绘制
                cv2.putText(image, f"left_min:{(min_list_l)}", (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"right_min:{(min_list_r)}", (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("rgb", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                running_flag[0] = False  # 设置标志为False以停止线程
                break

    finally:
        k4a.stop()
        cv2.destroyAllWindows()
        mp_hands.close()
        thread.join()  # 确保线程结束


if __name__ == "__main__":
    main()
