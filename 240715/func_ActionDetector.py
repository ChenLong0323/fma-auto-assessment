'''
"keypoints": [
    0"nose",
    1"left_eye",
    2"right_eye",
    3"left_ear",
    4"right_ear",
    5"left_shoulder",
    6"right_shoulder",
    7"left_elbow",
    8"right_elbow",
    9"left_wrist",
    10"right_wrist",
    11"left_hip",
    12"right_hip",
    13"left_knee",
    14"right_knee",
    15"left_ankle",
    16"right_ankle"
]
'''
import logging
import math
import time
import cv2

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志记录级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志记录格式
    filename='3_darw_in_inter.log',  # 设置日志文件名
    filemode='w'  # 设置文件模式，'w' 表示覆盖写入，'a' 表示追加写入
)


class ActionDetector:
    def detect(self,
               action_idx,
               data_cons, data_var, data_result,
               keypoints_ad, boxes, hands_coordinates, image_show,
               trigger_queue, trigger_queue_overtime):
        raise NotImplementedError("Subclasses should implement this method")


class Action1Detector(ActionDetector):
    def detect(self,
               action_idx,
               data_cons, data_var, data_result,
               keypoints_ad, boxes, hands_coordinates, image_show,
               trigger_queue, trigger_queue_overtime):
        required_keypoints_indices = [14,  # "right_knee",
                                      13,  # "left_knee",
                                      3,  # "left_ear"
                                      4]  # "right_ear"
        # 判断是否进入检测内容
        logging.info('action1 is running')
        # 判断是否进入检测内容
        if len(keypoints_ad) > max(required_keypoints_indices):
            required_keypoints = [keypoints_ad[i][:2] for i in required_keypoints_indices]
            # 如果左手，data_var.flag_handedness = 0，则是hans_coordinates[0]不为空
            # 如果右手，data_var.flag_handedness = 1，则是hands_coordinates[3]不为空
            if hands_coordinates[3 * data_var.flag_handedness][0] is not None:
                # 获取左右手的位置
                x1, y1 = hands_coordinates[1 + 3 * data_var.flag_handedness]
                distance_threshold = 30
                # TODO：这里是这个动作的修改内容
                # data_var.flag_handedness = 0  # 惯用手，0是左手，1是右手
                if data_result.fma[action_idx][data_var.flag_handedness] == 0:
                    # 获取左右膝盖的位置
                    x2, y2 = required_keypoints[data_var.flag_handedness]
                    # 计算距离
                    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    # 绘制交互
                    cv2.circle(image_show, (int(x2), int(y2)), distance_threshold, (0, 255, 0), 2)
                    # 交互判断
                    if distance < distance_threshold:
                        trigger_queue.append(1)
                        trigger_queue_overtime.append(0)
                        # 如果trigger 超过阈值，触发下一阶段
                        if sum(trigger_queue) > data_cons.THRESHOLD_TRIGGER:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 评分为1
                            data_result.fma[action_idx][data_var.flag_handedness] += 1
                            data_result.move_start_time[action_idx][data_var.flag_handedness] = time.time()
                    # 超时检测
                    else:
                        trigger_queue_overtime.append(1)
                        trigger_queue.append(0)
                        if sum(trigger_queue_overtime) > data_cons.THRESHOLD_OVERTIME:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 超时换手
                            data_var.flag_handedness = 1 - data_var.flag_handedness
                            data_var.num_sides_finished += 1

                # 量表评分为 1
                else:
                    # 获取左右耳坐标
                    x2, y2 = required_keypoints[2 + data_var.flag_handedness]
                    # 计算距离
                    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    # 记录最小距离
                    if distance < data_result.dis[action_idx][data_var.flag_handedness]:
                        data_result.dis[action_idx][data_var.flag_handedness] = distance

                    # 绘制交互
                    cv2.circle(image_show, (int(x2), int(y2)), distance_threshold, (0, 255, 0), 2)
                    # 交互判断
                    if distance < distance_threshold:
                        trigger_queue.append(1)
                        trigger_queue_overtime.append(0)
                        # 如果trigger 超过阈值，触发下一阶段
                        if sum(trigger_queue) > data_cons.THRESHOLD_TRIGGER:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 评分+1,为2
                            data_result.fma[action_idx][data_var.flag_handedness] += 1
                            data_result.move_start_time[action_idx][data_var.flag_handedness] = time.time() - data_result.move_start_time[action_idx][
                                data_var.flag_handedness]
                            # 正常换手
                            data_var.flag_handedness = 1 - data_var.flag_handedness
                            data_var.num_sides_finished += 1
                    # 超时检测
                    else:
                        trigger_queue_overtime.append(1)
                        trigger_queue.append(0)
                        if sum(trigger_queue_overtime) > data_cons.THRESHOLD_OVERTIME:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 超时换手
                            data_var.flag_handedness = 1 - data_var.flag_handedness
                            data_var.num_sides_finished += 1
                # 检测是否更换动作
                if data_var.num_sides_finished == 2:
                    action_idx += 1
                    data_var.num_sides_finished = 0

        logging.info(f"action_idx: {action_idx}, flag_handedness: {data_var.flag_handedness}, "
                     f"fma: {data_result.fma}, num_sides_finished: {data_var.num_sides_finished}, "
                     f"num_side_count: {data_var.num_side_count}, move_start_time: {data_result.move_start_time},"
                     f"dis: {data_result.dis}")
        return action_idx, image_show


class Action2Detector(ActionDetector):
    def detect(self,
               action_idx,
               data_cons, data_var, data_result,
               keypoints_ad, boxes, hands_coordinates, image_show,
               trigger_queue, trigger_queue_overtime):
        required_keypoints_indices = [14,  # "right_knee",
                                      13,  # "left_knee",
                                      3,  # "left_ear"
                                      4]  # "right_ear"
        # 判断是否进入检测内容
        logging.info('action1 is running')
        # 判断是否进入检测内容
        if len(keypoints_ad) > max(required_keypoints_indices):
            required_keypoints = [keypoints_ad[i][:2] for i in required_keypoints_indices]
            # 如果左手，data_var.flag_handedness = 0，则是hans_coordinates[0]不为空
            # 如果右手，data_var.flag_handedness = 1，则是hands_coordinates[3]不为空
            if hands_coordinates[3 * data_var.flag_handedness][0] is not None:
                # 获取左右手的位置
                x1, y1 = hands_coordinates[1 + 3 * data_var.flag_handedness]
                distance_threshold = 30
                # TODO：这里是这个动作的修改内容
                # data_var.flag_handedness = 0  # 惯用手，0是左手，1是右手
                if data_result.fma[action_idx][data_var.flag_handedness] == 0:
                    # 获取左右膝盖的位置
                    x2, y2 = required_keypoints[data_var.flag_handedness]
                    # 计算距离
                    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    # 绘制交互
                    cv2.circle(image_show, (int(x2), int(y2)), distance_threshold, (0, 255, 0), 2)
                    # 交互判断
                    if distance < distance_threshold:
                        trigger_queue.append(1)
                        trigger_queue_overtime.append(0)
                        # 如果trigger 超过阈值，触发下一阶段
                        if sum(trigger_queue) > data_cons.THRESHOLD_TRIGGER:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 评分为1
                            data_result.fma[action_idx][data_var.flag_handedness] += 1
                            data_result.move_start_time[action_idx][data_var.flag_handedness] = time.time()
                    # 超时检测
                    else:
                        trigger_queue_overtime.append(1)
                        trigger_queue.append(0)
                        if sum(trigger_queue_overtime) > data_cons.THRESHOLD_OVERTIME:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 超时换手
                            data_var.flag_handedness = 1 - data_var.flag_handedness
                            data_var.num_sides_finished += 1

                # 量表评分为 1
                else:
                    # 获取左右耳坐标
                    x2, y2 = required_keypoints[2 + data_var.flag_handedness]
                    # 计算距离
                    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    # 记录最小距离
                    if distance < data_result.dis[action_idx][data_var.flag_handedness]:
                        data_result.dis[action_idx][data_var.flag_handedness] = distance

                    # 绘制交互
                    cv2.circle(image_show, (int(x2), int(y2)), distance_threshold, (0, 255, 0), 2)
                    # 交互判断
                    if distance < distance_threshold:
                        trigger_queue.append(1)
                        trigger_queue_overtime.append(0)
                        # 如果trigger 超过阈值，触发下一阶段
                        if sum(trigger_queue) > data_cons.THRESHOLD_TRIGGER:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 评分+1,为2
                            data_result.fma[action_idx][data_var.flag_handedness] += 1
                            data_result.move_start_time[action_idx][data_var.flag_handedness] = time.time() - data_result.move_start_time[action_idx][
                                data_var.flag_handedness]
                            # 正常换手
                            data_var.flag_handedness = 1 - data_var.flag_handedness
                            data_var.num_sides_finished += 1
                    # 超时检测
                    else:
                        trigger_queue_overtime.append(1)
                        trigger_queue.append(0)
                        if sum(trigger_queue_overtime) > data_cons.THRESHOLD_OVERTIME:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 超时换手
                            data_var.flag_handedness = 1 - data_var.flag_handedness
                            data_var.num_sides_finished += 1
                # 检测是否更换动作
                if data_var.num_sides_finished == 2:
                    action_idx += 1
                    data_var.num_sides_finished = 0

        logging.info(f"action_idx: {action_idx}, flag_handedness: {data_var.flag_handedness}, "
                     f"fma: {data_result.fma}, num_sides_finished: {data_var.num_sides_finished}, "
                     f"num_side_count: {data_var.num_side_count}, move_start_time: {data_result.move_start_time},"
                     f"dis: {data_result.dis}")
        return action_idx, image_show


class Action3Detector(ActionDetector):
    def detect(self,
               action_idx,
               data_cons, data_var, data_result,
               keypoints_ad, boxes, hands_coordinates, image_show,
               trigger_queue, trigger_queue_overtime):
        required_keypoints_indices = [13,  # "left_knee",
                                      14,  # "right_knee",
                                      5,  # "left_shoulder"
                                      6,  # "right_shoulder"
                                      7,  # "left_elbow",
                                      8]  # "right_elbow",

        logging.info('action3 is running')
        if len(keypoints_ad) > max(required_keypoints_indices):
            required_keypoints = [keypoints_ad[i][:2] for i in required_keypoints_indices]
            if hands_coordinates[3 * data_var.flag_handedness][0] is not None:
                x1, y1 = hands_coordinates[1 + 3 * data_var.flag_handedness]
                distance_threshold = 40
                # TODO：这里是这个动作的修改内容
                # data_var.flag_handedness = 0  # 惯用手，0是左手，1是右手
                if data_result.fma[action_idx][data_var.flag_handedness] == 0:
                    # 获取左右膝盖的位置
                    x2, y2 = required_keypoints[data_var.flag_handedness]
                    # 计算距离
                    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    # 绘制交互
                    cv2.circle(image_show, (int(x2), int(y2)), distance_threshold, (0, 255, 0), 2)
                    # 交互判断
                    if distance < distance_threshold:
                        trigger_queue.append(1)
                        trigger_queue_overtime.append(0)
                        # 如果trigger 超过阈值，触发下一阶段
                        if sum(trigger_queue) > data_cons.THRESHOLD_TRIGGER:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 评分为1
                            data_result.fma[action_idx][data_var.flag_handedness] += 1
                            data_result.move_start_time[action_idx][data_var.flag_handedness] = time.time()
                    # 超时检测
                    else:
                        trigger_queue_overtime.append(1)
                        trigger_queue.append(0)
                        if sum(trigger_queue_overtime) > data_cons.THRESHOLD_OVERTIME:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 超时换手
                            data_var.flag_handedness = 1 - data_var.flag_handedness
                            data_var.num_sides_finished += 1

                # 量表评分为 1
                else:
                    # 获取肩膀坐标
                    x2, y2 = required_keypoints[2 + data_var.flag_handedness]
                    # 获取手肘坐标
                    x3, y3 = required_keypoints[4 + data_var.flag_handedness]
                    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    distance3 = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
                    # 记录最小距离
                    if distance < data_result.dis[action_idx][data_var.flag_handedness]:
                        data_result.dis[action_idx][data_var.flag_handedness] = distance

                    # 绘制交互
                    cv2.circle(image_show, (int(x2), int(y2)), distance_threshold, (0, 255, 0), 2)
                    # 判断循环
                    if distance < distance_threshold and distance3 < distance_threshold:
                        trigger_queue.append(1)
                        trigger_queue_overtime.append(0)
                        # 如果trigger 超过阈值，触发下一阶段
                        if sum(trigger_queue) > data_cons.THRESHOLD_TRIGGER:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 评分+1,为2
                            data_result.fma[action_idx][data_var.flag_handedness] += 1
                            data_result.move_start_time[action_idx][data_var.flag_handedness] = time.time() - data_result.move_start_time[action_idx][
                                data_var.flag_handedness]
                            # 正常换手
                            data_var.flag_handedness = 1 - data_var.flag_handedness
                            data_var.num_sides_finished += 1
                    # 超时检测
                    else:
                        trigger_queue_overtime.append(1)
                        trigger_queue.append(0)
                        if sum(trigger_queue_overtime) > data_cons.THRESHOLD_OVERTIME:
                            trigger_queue.clear()
                            trigger_queue_overtime.clear()
                            # 超时换手
                            data_var.flag_handedness = 1 - data_var.flag_handedness
                            data_var.num_sides_finished += 1
                # 检测是否更换动作
                if data_var.num_sides_finished == 2:
                    action_idx += 1
                    data_var.num_sides_finished = 0

        logging.info(f"action_idx: {action_idx}, flag_handedness: {data_var.flag_handedness}, "
                     f"fma: {data_result.fma}, num_sides_finished: {data_var.num_sides_finished}, "
                     f"num_side_count: {data_var.num_side_count}, move_start_time: {data_result.move_start_time},"
                     f"dis: {data_result.dis}")
        return action_idx, image_show


class ActionManager:
    def __init__(self):
        self.detectors = {
            1: Action1Detector(),  # Action1 的检测器
            2: Action3Detector(),  # Action3 的检测器
        }

    def detect_action(self,
                      action_idx,
                      data_cons, data_var, data_result,
                      keypoints_ad, boxes, hands_coordinates, image_show,
                      trigger_queue, trigger_queue_overtime):
        if action_idx in self.detectors:
            return self.detectors[action_idx].detect(
                action_idx,
                data_cons, data_var, data_result,
                keypoints_ad, boxes, hands_coordinates, image_show,
                trigger_queue, trigger_queue_overtime)
        else:
            return f"Unknown action index: {action_idx}"
