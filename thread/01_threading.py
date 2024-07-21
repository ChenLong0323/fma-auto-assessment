from func_initial import initialize, draw_circles, extract_landmark_coordinates
from func_data_class import Constants, HandStatus, ResultData
import cv2
import time
from collections import deque
import threading
import queue
import numpy as np


def process_frame(k4a, model, mp_hands,
                  frame_queue, running_flag, constants, fps_queue):
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
                image = cv2.flip(image_show, 1)  # 翻转图像

                # 图片显示
                cv2.imshow("rgb", image)

                # 输出内容
                frame_queue.put((keypoints, x_coordinates, y_coordinates), timeout=0.1)

                # 计算并显示FPS
                fps_queue.append(time.time())
                if len(fps_queue) > 30:
                    fps_queue.popleft()

                # 确保队列长度大于1以避免除零错误
                if len(fps_queue) > 1:
                    fps = len(fps_queue) / (fps_queue[-1] - fps_queue[0])
                    print(f"FPS: {fps:.2f}")

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


def interface(hand_status, result_data, constants,
              frame_queue, trigger_queue,
              indices):
    while True:
        try:
            keypoints, x_coordinates, y_coordinates = frame_queue.get()
            if len(keypoints) > max(indices) and None not in x_coordinates and None not in y_coordinates:
                ext = keypoints[indices]

                # 寻找惯用手，先检测惯用手
                if hand_status.flag_handedness == 0:
                    # 惯用手检测：右侧
                    if ext[5] > y_coordinates[0]:
                        trigger_queue.append(1)
                        if sum(trigger_queue) > constants.threshold:
                            hand_status.flag_handedness = 2
                            trigger_queue.clear()
                    #
                    elif ext[3] > y_coordinates[3]:
                        trigger_queue.append(1)
                        if sum(trigger_queue) > constants.threshold:
                            trigger_queue.clear()
                            hand_status.flag_handedness = 1
                    else:
                        trigger_queue.append(0)

                # 左侧检测
                elif hand_status.flag_handedness == 1:
                    if hand_status.flag_left == 0:
                        print('right')
                        # 左手远离鼻子
                        if abs((ext[0] - x_coordinates[3])) > 6 * constants.win_w:
                            trigger_queue.append(1)
                            if sum(trigger_queue) > constants.threshold:
                                # 触发flag后，清空trigger
                                trigger_queue.clear()
                                # flag值改变
                                hand_status.flag_left = 1
                                if count == 0:
                                    move_start_time = time.time()

                                # 第二轮循环记录上一轮最小值
                                if count != 0:
                                    result_data.min_list_l.append(dis)
                                dis = 1000

                                # 指鼻5次后，换边
                                if count > 4:
                                    hand_status.flag_handedness = 2
                                    count = 0  # 重置count，单侧完成次数清零
                                    constants.count_sides += 1
                                    result_data.time_left = time.time() - move_start_time
                                    continue

                        else:
                            trigger_queue.append(0)

                    # 开始交互
                    elif hand_status.flag_left == 1:
                        # 绘制左侧目标line
                        # 如果左肩x坐标大于左手x坐标（左手在左肩内侧）
                        if ext[2] > x_coordinates[3]:

                            # 整合运动过程中，记录最小值
                            dis1 = (abs(ext[0] - x_coordinates[3]) +
                                    abs(ext[1] - y_coordinates[3]))  # 可以展示到屏幕上
                            if dis > dis1:
                                dis = dis1

                            trigger_queue.append(1)
                            if sum(trigger_queue) > constants.threshold:
                                # 触发flag后，清空trigger
                                trigger_queue.clear()
                                # flag值改变
                                count += 1
                                hand_status.flag_left = 0


                        else:
                            trigger_queue.append(0)

                # 右侧检测
                elif hand_status.flag_handedness == 2:
                    if hand_status.flag_right == 0:
                        # 交互绘制:左侧box
                        print('left')
                        # 如果右手手腕x坐标远离鼻子x坐标
                        if abs((ext[0] - x_coordinates[0])) > 6 * constants.win_w:
                            trigger_queue.append(1)
                            if sum(trigger_queue) > constants.threshold:
                                # 触发flag后，清空trigger_queue
                                trigger_queue.clear()
                                # flag值改变
                                hand_status.flag_right = 1
                                if count == 0:
                                    move_start_time = time.time()

                                # 第二轮循环记录上一轮最小值
                                if count != 0:
                                    result_data.min_list_r.append(dis)
                                dis = 1000
                        else:
                            trigger_queue.append(0)

                    # 开始交互
                    elif hand_status.flag_right == 1:
                        # 如果右肩x坐标小于右手x坐标（右手在右肩内侧）
                        if ext[4] < x_coordinates[0]:
                            # 本动作特定：记录最小值
                            dis1 = (abs(ext[0] - x_coordinates[0]) +
                                    abs(ext[1] - y_coordinates[0]))  # 可以展示到屏幕上
                            if dis > dis1:
                                dis = dis1

                            trigger_queue.append(1)
                            if sum(trigger_queue) > constants.threshold:
                                # 触发flag后，清空trigger_queue
                                trigger_queue.clear()
                                # flag值改变
                                count += 1
                                hand_status.flag_right = 0

                                # 指鼻5次后，换边
                                if count > 4:
                                    hand_status.flag_handedness = 1
                                    count = 0  # 重置count，单侧完成次数清零
                                    constants.count_sides += 1
                                    result_data.time_right = time.time() - move_start_time
                                    result_data.min_list_r.append(dis)
                                    dis = 1000
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
    running_flag = [True]  # 使用列表作为可变标志
    fps_queue = deque(maxlen=30)  # 用于计算FPS的时间戳队列
    trigger_queue = deque(maxlen=30)
    indices = [
        0, 1,  # 鼻子 x 坐标、y 坐标
        15, 16,  # 左肩膀 x 坐标、y 坐标
        18, 19,  # 右肩膀 x 坐标、y 坐标
    ]

    thread_frame = threading.Thread(target=process_frame, args=(k4a, model, mp_hands,
                                                                frame_queue, running_flag, constants,
                                                                fps_queue))
    thread_frame.start()

    thread_interface = threading.Thread(target=interface, args=(hand_status, result_data, constants,
                                                                frame_queue, trigger_queue,
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
