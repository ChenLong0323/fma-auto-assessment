"""
本文件将imshow作为第三个线程独立出去
稳定运行结果为
FPS_interface: 24.62
FPS_show: 24.72
FPS_process: 24.681
"""
from func_initial import initialize, draw_circles, extract_landmark_coordinates
from func_data_class import Constants, HandStatus, ResultData
import cv2
import time
from collections import deque
import threading
import queue
import numpy as np


def process_frame(k4a, model, mp_hands,
                  image_queue, frame_queue, fps_queue_process,
                  constants,running_flag
                  ):
    while running_flag[0]:
        try:
            capture = k4a.get_capture()
            image = capture.color[:, 640 - constants.WIDTH_DET_POSE:640 + constants.WIDTH_DET_POSE, :3]
            image_hand = cv2.cvtColor(
                image[:, constants.WIDTH_DET_POSE - constants.WIDTH_DET_HAND:constants.WIDTH_DET_POSE + constants.WIDTH_DET_HAND, :3],
                cv2.COLOR_BGR2RGB)
            height, width, _ = image_hand.shape

            results_pose = model(image, stream=True, conf=0.7, verbose=False)

            results_hand = mp_hands.process(image_hand).multi_hand_landmarks

            x_coordinates, y_coordinates = extract_landmark_coordinates(results_hand, height, width, constants.WIDTH_DET_POSE,
                                                                        constants.WIDTH_DET_HAND)

            for result in results_pose:
                keypoints = np.ravel(result.keypoints.data[0].tolist())
                image_show = result.plot()
                draw_circles(image_show, x_coordinates, y_coordinates)
                image_show = cv2.flip(image_show, 1)  # 翻转图像

                # 输出内容
                frame_queue.put((keypoints, x_coordinates, y_coordinates), timeout=0.1)
                image_queue.put(image_show, timeout=0.1)

                # 计算并显示FPS
                fps_queue_process.append(time.time())
                if len(fps_queue_process) > 30:
                    fps_queue_process.popleft()

                # 确保队列长度大于1以避免除零错误
                if len(fps_queue_process) > 1:
                    fps = len(fps_queue_process) / (fps_queue_process[-1] - fps_queue_process[0])
                    print(f"FPS_process: {fps:.2f}")

        except queue.Full:
            # 在队列满时忽略，以避免阻塞
            print("full")
            continue
        except Exception as e:
            print(f"Error in thread: {e}")
            break


def show_frame(image_queue, running_flag, fps_queue_show):
    while running_flag[0]:
        try:
            # 计算并显示FPS
            fps_queue_show.append(time.time())
            if len(fps_queue_show) > 30:
                fps_queue_show.popleft()

            # 确保队列长度大于1以避免除零错误
            if len(fps_queue_show) > 1:
                fps = len(fps_queue_show) / (fps_queue_show[-1] - fps_queue_show[0])
                print(f"FPS_show: {fps:.2f}")

            # 图片显示
            image_show = image_queue.get()
            cv2.imshow("rgb", image_show)
            # 退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running_flag[0] = False  # 设置标志为False以停止线程
                break

        except queue.Full:
            # 在队列满时忽略，以避免阻塞
            print("full")
            continue
        except Exception as e:
            print(f"Error in thread: {e}")
            break


def interface(hand_status, result_data, constants, running_flag,
              frame_queue, trigger_queue, fps_queue_interface,
              indices):
    while running_flag[0]:
        try:
            # 计算并显示FPS
            fps_queue_interface.append(time.time())
            if len(fps_queue_interface) > 30:
                fps_queue_interface.popleft()

            # 确保队列长度大于1以避免除零错误
            if len(fps_queue_interface) > 1:
                fps = len(fps_queue_interface) / (fps_queue_interface[-1] - fps_queue_interface[0])
                print(f"FPS_interface: {fps:.2f}")

            keypoints, x_coordinates, y_coordinates = frame_queue.get()
            if len(keypoints) > max(indices) and None not in x_coordinates and None not in y_coordinates:
                ext = keypoints[indices]
                print(hand_status.flag_handedness, hand_status.flag_left, hand_status.flag_right)
                # 寻找惯用手，先检测惯用手
                if hand_status.flag_handedness == 0:
                    # 惯用手检测：右侧
                    if ext[5] > y_coordinates[0]:
                        trigger_queue.append(1)
                        if sum(trigger_queue) > constants.THRESHOLD:
                            hand_status.flag_handedness = 2
                            trigger_queue.clear()
                    #
                    elif ext[3] > y_coordinates[3]:
                        trigger_queue.append(1)
                        if sum(trigger_queue) > constants.THRESHOLD:
                            trigger_queue.clear()
                            hand_status.flag_handedness = 1
                    else:
                        trigger_queue.append(0)

                # 左侧检测
                elif hand_status.flag_handedness == 1:
                    if hand_status.flag_left == 0:
                        print('right')
                        # 左手远离鼻子
                        if abs((ext[0] - x_coordinates[3])) > 6 * constants.WIN_W:
                            trigger_queue.append(1)
                            if sum(trigger_queue) > constants.THRESHOLD:
                                # 触发flag后，清空trigger
                                trigger_queue.clear()
                                # flag值改变
                                hand_status.flag_left = 1
                                if hand_status.count == 0:
                                    hand_status.move_start_time = time.time()

                                # 第二轮循环记录上一轮最小值
                                if hand_status.count != 0:
                                    result_data.min_list_l.append(hand_status.dis)
                                hand_status.dis = 1000

                                # 指鼻5次后，换边
                                if hand_status.count > 4:
                                    hand_status.flag_handedness = 2
                                    hand_status.count = 0  # 重置hand_status.count，单侧完成次数清零
                                    hand_status.count_sides += 1
                                    result_data.time_left = time.time() - hand_status.move_start_time
                                    continue

                        else:
                            trigger_queue.append(0)

                    # 开始交互
                    elif hand_status.flag_left == 1:
                        # 绘制左侧目标line
                        # 如果左肩x坐标大于左手x坐标（左手在左肩内侧）
                        if ext[2] > x_coordinates[3]:

                            # 整合运动过程中，记录最小值
                            hand_status.dis1 = (abs(ext[0] - x_coordinates[3]) +
                                                abs(ext[1] - y_coordinates[3]))  # 可以展示到屏幕上
                            if hand_status.dis > hand_status.dis1:
                                hand_status.dis = hand_status.dis1

                            trigger_queue.append(1)
                            if sum(trigger_queue) > constants.THRESHOLD:
                                # 触发flag后，清空trigger
                                trigger_queue.clear()
                                # flag值改变
                                hand_status.count += 1
                                hand_status.flag_left = 0

                        else:
                            trigger_queue.append(0)

                # 右侧检测
                elif hand_status.flag_handedness == 2:
                    if hand_status.flag_right == 0:
                        # 交互绘制:左侧box
                        print('left')
                        # 如果右手手腕x坐标远离鼻子x坐标
                        if abs((ext[0] - x_coordinates[0])) > 6 * constants.WIN_W:
                            trigger_queue.append(1)
                            if sum(trigger_queue) > constants.THRESHOLD:
                                # 触发flag后，清空trigger_queue
                                trigger_queue.clear()
                                # flag值改变
                                hand_status.flag_right = 1
                                if hand_status.count == 0:
                                    hand_status.move_start_time = time.time()

                                # 第二轮循环记录上一轮最小值
                                if hand_status.count != 0:
                                    result_data.min_list_r.append(hand_status.dis)
                                hand_status.dis = 1000
                        else:
                            trigger_queue.append(0)

                    # 开始交互
                    elif hand_status.flag_right == 1:
                        # 如果右肩x坐标小于右手x坐标（右手在右肩内侧）
                        if ext[4] < x_coordinates[0]:
                            # 本动作特定：记录最小值
                            hand_status.dis1 = (abs(ext[0] - x_coordinates[0]) +
                                                abs(ext[1] - y_coordinates[0]))  # 可以展示到屏幕上
                            if hand_status.dis > hand_status.dis1:
                                hand_status.dis = hand_status.dis1

                            trigger_queue.append(1)
                            if sum(trigger_queue) > constants.THRESHOLD:
                                # 触发flag后，清空trigger_queue
                                trigger_queue.clear()
                                # flag值改变
                                hand_status.count += 1
                                hand_status.flag_right = 0

                                # 指鼻5次后，换边
                                if hand_status.count > 4:
                                    hand_status.flag_handedness = 1
                                    hand_status.count = 0  # 重置hand_status.count，单侧完成次数清零
                                    hand_status.count_sides += 1
                                    result_data.time_right = time.time() - hand_status.move_start_time
                                    result_data.min_list_r.append(hand_status.dis)
                                    hand_status.dis = 1000

        except queue.Full:
            # 在队列满时忽略，以避免阻塞
            print("full_interface")
            continue
        except Exception as e:
            print(f"Error in thread interface: {e}")
            break


def main():
    # 初始化模型
    k4a, model, mp_hands = initialize()

    # 初始化常量
    constants = Constants()

    # 初始化 HandStatus 和 ResultData 实例
    hand_status = HandStatus()
    result_data = ResultData()

    frame_queue = queue.Queue(maxsize=1)
    image_queue = queue.Queue(maxsize=1)
    running_flag = [True]  # 使用列表作为可变标志
    fps_queue_process = deque(maxlen=30)  # 用于计算FPS的时间戳队列
    fps_queue_show = deque(maxlen=30)
    fps_queue_interface = deque(maxlen=30)

    trigger_queue = deque(maxlen=60)
    indices = [
        0, 1,  # 鼻子 x 坐标、y 坐标
        15, 16,  # 左肩膀 x 坐标、y 坐标
        18, 19,  # 右肩膀 x 坐标、y 坐标
    ]

    thread_frame = threading.Thread(target=process_frame, args=((k4a, model, mp_hands,
                                                                 image_queue, frame_queue, fps_queue_process,
                                                                 constants,running_flag
                                                                 )))
    thread_frame.start()

    thread_show = threading.Thread(target=show_frame, args=(image_queue, running_flag, fps_queue_show))
    thread_show.start()

    thread_interface = threading.Thread(target=interface, args=(hand_status, result_data, constants, running_flag,
                                                                frame_queue, trigger_queue, fps_queue_interface,
                                                                indices))
    thread_interface.start()

    # try:
    #     while running_flag[0]:
    #         # 主线程保持事件循环，防止程序退出
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     running_flag[0] = False
    #
    # # 等待子线程结束
    # thread.join()
    #
    # # 释放OpenCV资源
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
