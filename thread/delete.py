import threading
import queue
import time
from abc import ABC, abstractmethod


# 基类定义
class ActionDetector(ABC):
    @abstractmethod
    def detect(self, keypoints):
        pass


# 动作1的检测类
class Action1Detector(ActionDetector):
    def detect(self, keypoints):
        required_keypoints = [1, 2, 3]
        if all(kp in keypoints for kp in required_keypoints):
            return "Action 1 detected"
        else:
            return "Action 1 not detected"


# 动作2的检测类
class Action2Detector(ActionDetector):
    def detect(self, keypoints):
        required_keypoints = [4, 5, 6]
        if all(kp in keypoints for kp in required_keypoints):
            return "Action 2 detected"
        else:
            return "Action 2 not detected"


# 可以继续定义更多的动作检测类...

# 使用字典管理动作检测类实例
actions = {
    1: Action1Detector(),
    2: Action2Detector(),
    # 可以继续添加更多动作及其检测类实例
}

if __name__ == "__main__":
    data_result = {}
    keypoints = [11, 22, 1, 2, 3, 66, 77, 88, 99]
    action_id = 1
    # print(actions)
    detector = actions.get(action_id)
    if detector is None:
        result = "Unknown action"
    else:
        # 调用检测方法
        result = detector.detect(keypoints)

    # 将结果存储到data_result
    data_result[action_id] = result
