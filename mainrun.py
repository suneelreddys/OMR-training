from ultralytics import YOLO
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model (untrained)
model = YOLO("yolov8s.yaml")  

# Train the model
model.train(
    data="data.yaml",  # Path to data.yaml
    epochs=50,         # Adjust based on need
    imgsz=640,         # Image size
    device=device,     # Use GPU if available
    batch=16,          # Adjust batch size based on GPU memory
    workers=4          # Number of data loading workers
)
  
print("Training complete!")
