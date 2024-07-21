class ActionDetector:
    def detect(self, keypoints_ad):
        raise NotImplementedError("Subclasses should implement this method")


class SitintPositionDetector(ActionDetector):
    def detect(self, keypoints_ad):
        return 1


class Action1Detector(ActionDetector):
    def detect(self, keypoints_ad):
        required_keypoints_indices = [0, 1, 2]  # indices for keypoints required for action 1
        required_keypoints = [keypoints_ad[i][:2] for i in required_keypoints_indices]

        if all(kp[0] is not None and kp[1] is not None for kp in required_keypoints):
            print(required_keypoints)
            return "Action 1 detected"
        else:
            return "Action 1 not detected"


class Action2Detector(ActionDetector):
    def detect(self, keypoints_ad):
        required_keypoints_indices = [3, 4, 5]  # indices for keypoints required for action 2
        required_keypoints = [keypoints_ad[i][:2] for i in required_keypoints_indices]

        if all(kp[0] is not None and kp[1] is not None for kp in required_keypoints):
            return "Action 2 detected"
        else:
            return "Action 2 not detected"


class ActionManager:
    def __init__(self):
        self.detectors = {
            "Action1": Action1Detector(),  # 屈肌协同运动 与 伸肌协同运动
            "Action2": Action2Detector(),  # 4-1：手触腰椎
            "Action3": Action3Detector(),  # 4-2:肩关节屈曲 90 度，肘关节伸直
            "Action4": Action4Detector(),  # 4-3:肩 0 度，肘屈 90 度，前臂旋前、旋后
            # TODO: 补全更多动作
        }

    def detect_action(self, action_name, keypoints_ad):
        if action_name in self.detectors:
            return self.detectors[action_name].detect(keypoints_ad)
        else:
            return f"Unknown action: {action_name}"


if __name__ == "__main__":
    # Example usage
    keypoints = [
        [110, 210, 0.9],  # x, y, confidence for keypoint 0
        [120, 220, 0.8],  # x, y, confidence for keypoint 1
        [130, 230, 0.7],  # x, y, confidence for keypoint 2
        [140, 240, 0.6],  # x, y, confidence for keypoint 35
        [150, 250, 0.5],  # x, y, confidence for keypoint 4
        [160, 260, 0.4],  # x, y, confidence for keypoint 5
        [170, 270, 0.3],  # x, y, confidence for keypoint 6
        # Add more keypoints as needed
    ]

    action_manager = ActionManager()
    print(action_manager.detect_action("Action1", keypoints))  # Should print "Action 1 detected" or "Action 1 not detected"
    print(action_manager.detect_action("Action2", keypoints))  # Should print "Action 2 detected" or "Action 2 not detected"
    print(action_manager.detect_action("Action3", keypoints))  # Should print "Unknown action: Action3"
