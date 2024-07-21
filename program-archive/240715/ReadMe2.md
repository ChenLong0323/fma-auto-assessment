# 项目目的
本项目主要更新func_ActionDetector项目中的函数，多数动作将手划入交互内容

1. 检测函数更新
    1. Action1Detector修改内容
```commandline
        required_keypoints_indices = [9,  # "left_wrist"
                                      10,  # "right_wrist"
                                      12,  # "right_hip"
                                      11,  # "left_hip"
                                      3,  # "left_ear"
                                      4]  # "right_ear"
```
变为