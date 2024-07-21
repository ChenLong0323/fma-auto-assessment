from ultralytics import YOLO

if __name__ == '__main__':
# Load a model
    model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

# Train the model
    results = model.train(data='coco8-pose.yaml', epochs=5, imgsz=640, device=0, batch=-1)
