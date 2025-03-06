from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

results=model(r"testing_images\1C.jpg",show=True,save=True)
