from ultralytics import YOLO
import os
import numpy as np
import cv2

# Load a model
model = YOLO('../../models/yolov8s-pose.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('../data_test/wheelchair.jpeg', stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    keypoints = np.ravel(result.keypoints.data[0].tolist())

    indices = [15, 16]
    point = (int(keypoints[indices[0]]), int(keypoints[indices[1]]))
    print(point)
    point1 = (point[0]-100,point[1]-100)
    frame = result.plot()

    # 定义点的半径
    radius = 20

# 使用cv2.circle绘制点
cv2.circle(frame, point, radius, (0, 255, 0), -1)
cv2.circle(frame, point1, radius, (0, 0, 255), -1)
cv2.imshow("Image with Point", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
