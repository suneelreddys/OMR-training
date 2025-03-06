from ultralytics import YOLO
import torch

device="cuda" if torch.cuda.is_available() else "cpu"

model=YOLO("runs/detect/train5/weights/best.pt")
model.train(
    data="data.yaml",   
    epochs=20,
    device=device,         
    imgsz=640,          
    batch=16,           
)

print("completed the extra training")