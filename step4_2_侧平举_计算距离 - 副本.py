import numpy as np
import matplotlib.pyplot as plt

# 来自标准坐姿
a = np.array(
    [650.85, 132.71, 0.99715, 663.93, 122.75, 0.98822, 635.63, 121.9, 0.98837, 681.27, 145.33, 0.87866, 606.39, 144.42,
     0.88962, 701.43, 226.18, 0.99792, 574.1, 221.35, 0.99828, 677.11, 320.04, 0.98918,
     557.29, 315.68, 0.99227, 615.35, 388.11, 0.98967, 551.26, 390.55, 0.99335, 687.83, 378.47, 0.9978, 596.83, 378.85,
     0.9981, 739.26, 460.49, 0.9931, 549.95, 462.13, 0.99352, 693.26, 620.18, 0.94479,
     603.35, 629.14, 0.94697])

indices = [27, 28,  # 01 "left_wrist"
           30, 31,  # 23 "right_wrist",30
           33, 34,  # 45 "left_hip",33
           36, 37,  # 67 "right_hip",36
           9, 10,  # 89 "left_ear", 9
           12, 13]  # 10,11 "right_ear",12

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
