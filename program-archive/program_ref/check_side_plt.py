from ultralytics import YOLO
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    point1 = (point[0]-100, point[1]-100)
    frame = result.plot()

    # 定义点的半径
    radius = 20

    # 使用matplotlib绘制点
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.scatter(point[0], point[1], s=radius**2, c='g', marker='o')  # 绿色点
    plt.scatter(point1[0], point1[1], s=radius**2, c='r', marker='o')  # 红色点
    plt.title("Image with Points")
    plt.axis('off')
    plt.show()
