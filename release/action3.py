import logging
import math
import time

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志记录级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志记录格式
    filename='../240713/1_without_hands.log',  # 设置日志文件名
    filemode='w'  # 设置文件模式，'w' 表示覆盖写入，'a' 表示追加写入
)


def action3(data_cons, data_var, data_result,
            keypoints_ad, boxes, x_coordinates, y_coordinates,
            trigger_queue, trigger_queue_overtime, action_idx):
    required_keypoints_indices = [9,  # "left_wrist"
                                  10,  # "right_wrist"
                                  11,  # "left_hip"
                                  12,  # "right_hip"
                                  5,  # "left_shoulder"
                                  6]  # "right_shoulder"
    logging.info('action3 is running')
    # 判断是否进入检测内容
    if len(keypoints_ad) > max(required_keypoints_indices):
        required_keypoints = [keypoints_ad[i][:2] for i in required_keypoints_indices]
        distance_threshold = 50
        # TODO：这里是这个动作的修改内容
        # data_var.flag_handedness = 0  # 惯用手，0是左手，1是右手
        if data_result.fma[data_var.flag_handedness] == 0:
            # 开始检测：如果左手在同侧臀部附近
            x1, y1 = required_keypoints[0 + data_var.flag_handedness]
            x2, y2 = required_keypoints[2 + data_var.flag_handedness]
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < distance_threshold:
                trigger_queue.append(1)
                trigger_queue_overtime.append(0)
                # 如果trigger 超过阈值，触发下一阶段
                if sum(trigger_queue) > data_cons.THRESHOLD_TRIGGER:
                    trigger_queue.clear()
                    trigger_queue_overtime.clear()
                    # 评分为1
                    data_result.fma[data_var.flag_handedness] += 1
                    data_result.move_start_time[data_var.flag_handedness] = time.time()
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
            # 如果左手在同侧肩部以上
            x1, y1 = required_keypoints[0 + data_var.flag_handedness]
            x2, y2 = required_keypoints[4 + data_var.flag_handedness]
            distance = y2 - y1
            # 记录最小距离
            if distance < data_result.dis[data_var.flag_handedness]:
                data_result.dis[data_var.flag_handedness] = distance
            # 判断循环:手比肩高
            if distance < 0:
                trigger_queue.append(1)
                trigger_queue_overtime.append(0)
                # 如果trigger 超过阈值，触发下一阶段
                if sum(trigger_queue) > data_cons.THRESHOLD_TRIGGER:
                    trigger_queue.clear()
                    trigger_queue_overtime.clear()
                    # 评分+1,为2
                    data_result.fma[data_var.flag_handedness] += 1
                    data_result.move_start_time[data_var.flag_handedness] = time.time() - data_result.move_start_time[data_var.flag_handedness]
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
    return action_idx
