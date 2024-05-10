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
# ]

from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt

# Load a model
model = YOLO('models/yolov8l-pose.pt')  # pretrained YOLOv8n model

filename = 'step2_2_Ending_rgb_2.png'
filename_ext = os.path.splitext(filename)[0]
# Run batched inference on a list of images
results = model(filename, stream=True)  # return a generator of Results objects


indices = [27, 28,  # 01 "left_wrist"
           30, 31,  # 23 "right_wrist",30
           33, 34,  # 45 "left_hip",33
           36, 37,  # 67 "right_hip",36
           15, 16,  # 89 "left_shoulder",15
           18, 19]  # 10,11 "right_shoulder",18

# Process results generator
for result in results:
    boxes = np.ravel(result.boxes.data[0].tolist())  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # 提取左右耳，左右肩膀
    keypoints = np.ravel(result.keypoints.data[0].tolist())
    ext = keypoints[indices]

    probs = result.probs  # Probs object for classification outputs
    result.save(filename=filename_ext + '.jpg')  # save to disk
    if os.path.exists(filename_ext + '.txt'):
        os.remove(filename_ext + '.txt')
    result.save_txt(txt_file=filename_ext + '.txt', save_conf=False)
    # result.show()  # display to screen
    frame = result.plot()
    plt.imshow(frame)
    plt.axis('off')
