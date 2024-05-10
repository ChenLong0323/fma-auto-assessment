import numpy as np
import matplotlib.pyplot as plt

# 来自标准坐姿
a = np.array(
    [647.14, 141.17, 0.99411, 659.87, 128.25, 0.9826, 633.91, 129.65, 0.97762, 684.1, 145.42, 0.85311, 614.79, 149.23,
     0.78767, 718.22, 230.52, 0.99494, 588.13, 233.17, 0.99584, 743.82, 322.13, 0.96859,
     567.05, 327.3, 0.97494, 740.62, 378.01, 0.95917, 584.27, 377.39, 0.96586, 706.53, 378, 0.99435, 612.68, 380.36,
     0.99492, 752.37, 467.26, 0.98769, 579.77, 472.17, 0.98867, 741.3, 633.3, 0.93614,
     575.12, 639.33, 0.94057])

# 左耳，右耳，左肩，右肩
indices = [9, 10, 12, 13, 15, 16, 18, 19]  # 指定的索引列表


extracted_values = a[indices]
pairs = extracted_values.reshape(-1, 2)

# pairs = a.reshape(-1, 3)

# 提取x和y坐标
x = pairs[:, 0]
y = pairs[:, 1]
plt.plot(x, y, marker='o', linestyle='', color='b')


# x_flipped = np.max(x) - x
# y_flipped = np.max(y) - y
# plt.plot(x_flipped, y_flipped, marker='o', linestyle='', color='b')

# 添加标题和坐标轴标签
plt.title('Scatter Plot of Points')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.grid(True)
plt.show()
