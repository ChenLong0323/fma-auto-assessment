import numpy as np
import matplotlib.pyplot as plt

# 来自标准坐姿
a = np.array(
    [637.19, 231.24, 0.99946, 648.22, 221.49, 0.99796, 626.37, 221.36, 0.99736, 663.64, 236.11, 0.9184, 613.89, 236.28, 0.85859, 690.89, 295.76,
     0.99938, 587.52, 297.76, 0.99945, 708.65, 372.1, 0.99399,
     565.1, 373.45, 0.99471, 696.98, 411.28, 0.99329, 568.35, 408.76, 0.99349, 674.23, 422.55, 0.99933, 600.86, 422.78, 0.99939, 705.45, 481.78,
     0.99888, 556.74, 481.21, 0.9989, 688.47, 637.74, 0.99197,
     563.28, 634.38, 0.99245])

# "keypoints": [
#     "nose", 0
#     "left_eye",3
#     "right_eye",6
#     "left_ear",9
#     "right_ear",12
#     "left_shoulder",15
#     "right_shoulder",18
#     "left_elbow",21
#     "right_elbow",24
#     "left_wrist",27
#     "right_wrist",30
#     "left_hip",33
#     "right_hip",36
#     "left_knee",39
#     "right_knee",42
#     "left_ankle",45
#     "right_ankle"48
# ],

indices = [15, 16,  # 01 "left_shoulder",15
           18, 19,  # 23 "right_shoulder",18
           33, 34,  # 45 "left_hip",33
           36, 37,  # 67 "right_hip",36
           ]

ext = a[indices]
pairs = ext.reshape(-1, 2)

# pairs = a.reshape(-1, 3)

# 提取x和y坐标
x = pairs[:, 0]
y = pairs[:, 1]
# plt.plot(x, y, marker='o', linestyle='', color='b')

x_flipped = np.max(x) - x
y_flipped = np.max(y) - y
plt.plot(x_flipped, y_flipped, marker='o', linestyle='', color='b')

# 添加标题和坐标轴标签
plt.title('Scatter Plot of Points')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.grid(True)
plt.show()
