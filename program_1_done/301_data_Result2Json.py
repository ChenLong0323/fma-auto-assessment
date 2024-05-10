from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('0.jpeg', stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes.data.cpu().numpy()[0]
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    keypoints = result.keypoints.data.cpu().numpy()[0]

    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk
    result.save_txt(txt_file='result.txt', save_conf=False)