from ultralytics import YOLO
import os

# Load a model
model = YOLO('../models/yolov8s-pose.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('../data_1_test/wheelchair.jpeg', stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    keypoints = result.keypoints.data[0].tolist()
    probs = result.probs  # Probs object for classification outputs
    result.save(filename='result.jpg')  # save to disk
    if os.path.exists('result.txt'):
        os.remove('result.txt')
    result.save_txt(txt_file='result.txt', save_conf=False)
    result.show()  # display to screen

indices = [15, 16]
