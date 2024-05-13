import cv2

img = cv2.imread('RGB.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
