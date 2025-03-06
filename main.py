from ultralytics import YOLO

# Train YOLO from scratch
model = YOLO("yolov8s.yaml")  # Initialize an untrained YOLO model
model.train(data="data.yaml", epochs=100, imgsz=640)

