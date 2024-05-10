import cv2
import numpy as np
import time
import pyk4a
from pyk4a import Config, PyK4A
from ultralytics import YOLO
from collections import deque

indices = [27, 28,  # 01 "left_wrist"
           30, 31,  # 23 "right_wrist",30
           0, 1,  # 45 "nose"
           36, 37,  # 67 "right_hip",36
           # action2 判断点
           3, 4,  # 89 "left_eye",3
           6, 7,  # 10,11 "right_eye",6
           # 添加左右手肘
           21, 22,  # 12,13 "left_elbow",21
           24, 25,  # 14,15 "right_elbow",24
           15, 16,  # 16,17 "left_shoulder",15
           12, 13  # 18,19 "right_shoulder",18
           ]

# 基本宽度(r)
shoulder_w = 52
hip_w = 37
window_w = 115

# 检验参数
win_h = 30
win_w = 20
check_h = 140


def main():
    # 初始化
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # FPS 计算
    frame_count = 0
    start_time = time.time()
    fps = 0

    # 初始化 trigger，长度为100，初始空白
    trigger = deque(maxlen=100)

    # 加载模型
    model = YOLO('models/yolov8s-pose.pt')

    # 判断左右手状态
    flag = 0
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

    # 循环获取图像主体
    while True:
        capture = k4a.get_capture()
        # 读取frame,YOLO输入是三维图片
        frame = capture.color[:, 430:850, :3]
        frame = cv2.flip(frame, 1)
        # 调用yolo检测
        results = model(frame, stream=True, conf=0.7, verbose=False)
        results1 = model(frame)

        for result in results:
            keypoints = np.ravel(result.keypoints.data[0].tolist())
            frame = result.plot()

            if len(keypoints) > max(indices):
                ext = keypoints[indices]

                # 真正的循环内容

                if flag == 0:
                    # Step1.1：交互绘制
                    cv2.putText(frame, 'check hand', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (640 - window_w, 245), (640 - 165, 315), (0, 255, 255), 2)
                    cv2.rectangle(frame, (640 + window_w, 245), (640 + 165, 315), (0, 255, 255), 2)

                    # step1.2：交互判断
                    if abs((ext[1] - ext[17])) < win_h:
                        trigger.append(1)
                        if sum(trigger) > 60:
                            flag = 1
                            # 触发flag后，清空trigger
                            trigger.clear()
                    # 判断左手高于左肩-20
                    elif abs((ext[3] - ext[19])) < win_h:
                        trigger.append(1)
                        if sum(trigger) > 60:
                            # 触发flag后，清空trigger
                            trigger.clear()
                            # flag值改变
                            flag = 2
                    else:
                        trigger.append(0)

                # 左侧检测
                elif flag == 1:
                    if flag_l == 0:
                        # Step2.1.1：交互绘制:左侧box
                        cv2.line(frame, (640 + 4 * shoulder_w, 240), (640 + 4 * shoulder_w, 380), (0, 255, 255), 2)
                        cv2.putText(frame, 'not begin', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        # Step2.1.2:交互判断：判断左手、左髋
                        if abs((ext[0] - ext[16])) > 6 * win_w:
                            trigger.append(1)
                            if sum(trigger) > 20:
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
                                    flag = 2
                                    count = 0  # 重置count，单侧完成次数清零
                                    count_sides += 1
                                    move_time_l = time.time() - move_start_time
                                    continue

                        else:
                            trigger.append(0)

                    # 开始交互
                    elif flag_l == 1:
                        # 绘制文字
                        cv2.putText(frame, 'score:0', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        # 绘制左侧目标line
                        # 如果左肩x坐标大于左手x坐标（左手在左肩内侧）
                        if ext[0] - ext[16] < win_w:

                            # 本动作特定：记录最小值
                            dis1 = abs(ext[4] - ext[0]) + abs(ext[5] - ext[1])
                            if dis > dis1:
                                dis = dis1

                            trigger.append(1)
                            if sum(trigger) > 20:
                                # 触发flag后，清空trigger
                                trigger.clear()
                                # flag值改变
                                count += 1
                                flag_l = 0


                        else:
                            trigger.append(0)

                # 右侧检测
                elif flag == 2:
                    if flag_r == 0:
                        # Step2.1.1：交互绘制:左侧box
                        cv2.line(frame, (640 - 4 * shoulder_w, 240), (640 - 4 * shoulder_w, 380), (0, 255, 255), 2)
                        cv2.putText(frame, 'not begin', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        # Step2.1.2:交互判断：判断左手、左髋
                        if abs((ext[2] - ext[18])) > 6 * win_w:
                            trigger.append(1)
                            if sum(trigger) > 20:
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
                        # 绘制文字
                        cv2.putText(frame, 'score:0', (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        # 如果右肩x坐标小于右手x坐标（右手在右肩内侧）
                        if ext[2] - ext[18] > - win_w:
                            # 本动作特定：记录最小值
                            dis1 = abs(ext[4] - ext[2]) + abs(ext[5] - ext[3])
                            if dis > dis1:
                                dis = dis1

                            trigger.append(1)
                            if sum(trigger) > 20:
                                # 触发flag后，清空trigger
                                trigger.clear()
                                # flag值改变
                                count += 1
                                flag_r = 0

                                # 指鼻5次后，换边
                                if count > 4:
                                    flag = 1
                                    count = 0  # 重置count，单侧完成次数清零
                                    count_sides += 1
                                    move_time_r = time.time() - move_start_time
                                    min_list_r.append(dis)
                                    dis = 1000


                        else:
                            trigger.append(0)

        # 绘制box，绘制四个点
        cv2.rectangle(frame, (640 - window_w, 180), (640 + window_w, 710), (0, 255, 0), 2)
        # 绘制两肩连线
        cv2.line(frame, (640 + shoulder_w, 295), (640 - shoulder_w, 295), (0, 255, 0), 2)
        # 绘制两臀连线
        cv2.line(frame, (640 + hip_w, 422), (640 - hip_w, 422), (0, 255, 0), 2)

        # 绘制FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        # 绘制trigger长度
        cv2.putText(frame, f"trigger:{sum(trigger)}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"count:{count}", (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        # 将左右侧时间绘制
        cv2.putText(frame, f"left_time:{move_time_l:.2f}", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"right_time:{move_time_r:.2f}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        # 将左右侧最小值绘制
        cv2.putText(frame, f"left_min:{(min_list_l)}", (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"right_min:{(min_list_r)}", (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("rgb", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1
        # Calculate and print FPS every second
        if frame_count % 30 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 30 / elapsed_time
            print(f"FPS: {fps}")
            # Reset variables for the next second
            frame_count = 0
            start_time = time.time()
    k4a.stop()


if __name__ == "__main__":
    main()
