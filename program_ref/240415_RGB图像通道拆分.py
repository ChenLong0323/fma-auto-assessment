import cv2

image = cv2.imread('step2_1_Beginning_rgb_2.png')

# 获取每个通道的值
blue_channel = image[:, :, 0]
green_channel = image[:, :, 1]
red_channel = image[:, :, 2]

# 创建彩色图像，每个通道的值设置为0
blue_img = image.copy()
blue_img[:, :, 1] = 0  # 将绿色通道设为0
blue_img[:, :, 2] = 0  # 将红色通道设为0

green_img = image.copy()
green_img[:, :, 0] = 0  # 将蓝色通道设为0
green_img[:, :, 2] = 0  # 将红色通道设为0

red_img = image.copy()
red_img[:, :, 0] = 0  # 将蓝色通道设为0
red_img[:, :, 1] = 0  # 将绿色通道设为0

# 显示每个通道的彩色图像
cv2.imshow("Blue Channel", blue_img)
cv2.imshow("Green Channel", green_img)
cv2.imshow("Red Channel", red_img)

# 等待按下任意键退出窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
