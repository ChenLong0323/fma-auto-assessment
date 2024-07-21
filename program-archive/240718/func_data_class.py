# 类变量（class variable）和实例变量（instance variable）的区别
# 重复调用累变量，累变量不会初始化

class Constants:
    WIDTH_DET_POSE = 300
    WIDTH_DET_HAND = 300
    WIDTH_DRAW_SHOULDER = 52
    WIDTH_DRAW_HIP = 37
    WIDTH_DRAW_SIT = 115
    WIN_H = 30
    WIN_W = 20
    THRESHOLD = 30
    FPS_UPDATE_INTERVAL = 2
    TRIGGER_MAXLEN = 60
    THRESHOLD_TRIGGER = 30
    THRESHOLD_OVERTIME = 600
    hand_flag = [1, 1, 1, 1, 1, 1, 1, 1]


class HandStatus:
    flag_handedness = 0  # 惯用手，0是左手，1是右手
    num_sides_finished = 0  # 单个动作完成次数，判断单边完成后，如果=2，则action_idx+1
    num_side_count = 0  # 记录单侧完成次数（不是所有的都需要，连续指鼻需要）


class ResultData:
    fma = [[0 for _ in range(2)] for _ in range(17)]
    move_start_time = [[0 for _ in range(2)] for _ in range(17)]  # 记录开始移动的时间
    dis = [[10000 for _ in range(2)] for _ in range(17)]  # 记录距离


def main():
    # 检查constants
    constants1 = Constants()
    print('constants1.WIDTH_DET_HAND = (300)', constants1.WIDTH_DET_HAND)  # 输出 300

    # 修改实例变量
    constants1.WIDTH_DET_HAND = 400
    print('constants1.WIDTH_DET_HAND = (400)', constants1.WIDTH_DET_HAND)  # 输出 400

    # 创建另一个实例
    constants2 = Constants()
    print('constants2 = Constants() ', constants2.WIDTH_DET_HAND)  # 输出 300，仍然是类变量的初始值

    # 修改类变量
    Constants.WIDTH_DET_HAND = 500
    print('Constants.WIDTH_DET_HAND = 500时, constants1.WIDTH_DET_HAND = ', constants1.WIDTH_DET_HAND)  # 输出 400，因为它已被实例变量覆盖
    print('Constants.WIDTH_DET_HAND = 500时, constants2.WIDTH_DET_HAND = ', constants2.WIDTH_DET_HAND)  # 输出 500，因为它没有被实例变量覆盖

    result = ResultData()
    print('result.time_right ', result.time_right)
    print('result.min_list_r ', result.min_list_r)
    result.time_right = 2
    result.min_list_r.append(100)
    print('result.time_right ', result.time_right)
    print('result.min_list_r ', result.min_list_r)
    result = ResultData()
    print('result.time_right ', result.time_right)
    print('result.min_list_r ', result.min_list_r)


if __name__ == "__main__":
    main()
