from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt

# Load a model
model = YOLO('models/yolov8l-pose.pt')  # pretrained YOLOv8n model

filename = 'action3_rgb_2.png'
filename_ext = os.path.splitext(filename)[0]
# Run batched inference on a list of images
results = model(filename, stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = np.ravel(result.boxes.data[0].tolist())  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    # 来自标准坐姿
    a = np.ravel(result.keypoints.data[0].tolist())

    indices = [27, 28,  # 01 "left_wrist"
               30, 31,  # 23 "right_wrist",30
               33, 34,  # 45 "left_hip",33
               36, 37,  # 67 "right_hip",36
               15, 16,  # "left_shoulder", 15
               18, 19]  # "right_shoulder",18

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
