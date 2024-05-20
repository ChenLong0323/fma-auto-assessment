from pyk4a import PyK4A
import os
import cv2
import matplotlib.pyplot as plt
print(os.environ.get("K4A_DLL_DIR"))


# Load camera with the default config
k4a = PyK4A()
k4a.start()

# Get the next capture (blocking function)
capture = k4a.get_capture()
img_color = capture.color

""" cv2 直接输出正常 """
# cv2.imshow("Image with Point", img_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

""" matplotlib 为蓝色，证明原图像为BGR """
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()