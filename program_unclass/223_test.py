from ultralytics import YOLO

# Load a model
model = YOLO('../yolov8n-pose.pt')  # load an official model
model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category


# Load a model


  # build from YAML and transfer weights

# Train the model
results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)