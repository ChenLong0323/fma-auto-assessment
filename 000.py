from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n-pose.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('bus.jpg', save=True, save_txt=True, imgsz=320, conf=0.5)
