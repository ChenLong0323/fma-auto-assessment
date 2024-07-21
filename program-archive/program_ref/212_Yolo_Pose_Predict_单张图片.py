"""
左上角是原点
"""

from ultralytics import YOLO
import os
import cv2

# Load a model
model = YOLO('../../models/yolov8s-pose.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('../data_test/wheelchair.jpeg', stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    boxes1 = result.boxes.data[0].tolist()
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    keypoints1 = result.keypoints.data[0].tolist()
    print(keypoints1)
    probs = result.probs  # Probs object for classification outputs
    result.save(filename='result.jpg')  # save to disk
    if os.path.exists('result.txt'):
        os.remove('result.txt')
    result.save_txt(txt_file='result.txt', save_conf=False)
    # result.show()  # display to screen
    image_show = result.plot()
    print(boxes1)
    # cv2.rectangle(image_show, (int(boxes1[0]), int(boxes1[1])), (int(boxes1[0] + 20), int(boxes1[1]) + 20), (0, 255, 0), 2)
    cv2.line(image_show, (100, 100), (100, 300), (0, 255, 0), 2)
    cv2.line(image_show, (200, 100), (200, 500), (255, 0, 0), 2)
    cv2.imshow('image_show', image_show)
    cv2.waitKey(0)

indices = [15, 16]
