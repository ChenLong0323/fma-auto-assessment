import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('bus.jpg')
print(img.shape)
print(type(img))
img_90 = np.rot90(img, k=1)
plt.imshow(img_90)
plt.axis('off')
plt.show()