# "keypoints": [
#     "nose", 0
#     "left_eye",3
#     "right_eye",6
#     "left_ear",9
#     "right_ear",12
#     "left_shoulder",15
#     "right_shoulder",18
#     "left_elbow",21
#     "right_elbow",24
#     "left_wrist",27
#     "right_wrist",30
#     "left_hip",33
#     "right_hip",36
#     "left_knee",39
#     "right_knee",42
#     "left_ankle",45
#     "right_ankle"48
# ],

from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt

# Load a model
model = YOLO('models/yolov8l-pose.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('upper_action_rgb_1.png', stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = np.ravel(result.boxes.data[0].tolist())  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    keypoints = result.keypoints.data[0].tolist()
    # 提取左右耳，左右肩膀
    keypoints = np.ravel(result.keypoints.data[0].tolist())
    ext = keypoints[[9, 10, 12, 13, 15, 16, 18, 19]]
    probs = result.probs  # Probs object for classification outputs
    result.save(filename='result2.jpg')  # save to disk
    if os.path.exists('result2.txt'):
        os.remove('result2.txt')
    result.save_txt(txt_file='result2.txt', save_conf=False)
    # result.show()  # display to screen
    frame = result.plot()
    plt.imshow(frame)
    plt.axis('off')
