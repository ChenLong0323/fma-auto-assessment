from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO('../models/yolov8s-pose.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('step6_Ending_rgb_2.png', stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    # keypoints = result.keypoints.data[0].tolist()
    keypoints = np.ravel(result.keypoints.data[0].tolist())
    frame = result.plot()
    cv2.imshow("rgb", frame)
