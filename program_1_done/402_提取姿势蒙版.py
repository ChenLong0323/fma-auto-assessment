import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.engine.predictor import BasePredictor
# 读取 .npy 文件
gray_image = np.load('sit_depth_2.npy')
print(gray_image.shape)
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
gray_image = clahe.apply(gray_image)
# 设定阈值
T1 = 80
T2 = 150

# 创建一个和原图像大小相同的全零数组，作为结果图像
result_image = np.zeros_like(gray_image)

# 遍历整张图片
for i in range(gray_image.shape[0]):
    for j in range(473, 820):
        pixel_value = gray_image[i, j]
        if pixel_value < T1:
            result_image[i, j] = 0
        elif pixel_value > T2:
            result_image[i, j] = 0
        else:
            result_image[i, j] = 255

# 显示图像
# 定义结构元素（内核）
kernel = np.ones((5, 5), np.uint8)

# 应用闭运算，填补空洞
result_image = cv2.morphologyEx(result_image, cv2.MORPH_CLOSE, kernel)
# 应用开运算，腐蚀白点
result_image = cv2.morphologyEx(result_image, cv2.MORPH_OPEN, kernel)

plt.imshow(result_image, cmap='gray')
plt.axis('off')  # 关闭坐标轴
plt.show()

np.save("sit_mask.npy", result_image)
