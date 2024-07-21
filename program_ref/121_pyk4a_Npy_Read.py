import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取 .npy 文件
data = np.load('step2_1_Beginning_ir_2.npy')

# 显示图像
#plt.imshow(data, cmap='gray')
#plt.axis('off')  # 关闭坐标轴
# plt.show()

cv2.imshow("rgb", data)
cv2.waitKey(0)  # 等待用户按下键盘上的任意键
cv2.destroyAllWindows()  # 关闭窗口
