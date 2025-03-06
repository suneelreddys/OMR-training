import os
import re
from ultralytics import YOLO

# Function for natural sorting
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Load trained YOLO model
model = YOLO("runs/detect/train6/weights/best.pt")  # Update with your trained model path

# Path to the folder containing images
image_folder = r"C:\Users\ReNot\Desktop\OMR Project\omr_deviding\output\omr"  # Update with your folder path

# List to store results
predicted_labels = []

# Iterate through all images in a sorted order
for image_name in sorted(os.listdir(image_folder), key=natural_sort_key):
    image_path = os.path.join(image_folder, image_name)
    
    # Run inference on each image
    results = model(image_path)
    
    for result in results:
        if result.boxes:  # Check if any object is detected
            confs = result.boxes.conf.tolist()  # Get confidence scores
            labels = result.boxes.cls.tolist()  # Get class labels

            # Get the label with the highest confidence
            max_index = confs.index(max(confs))
            highest_label = int(labels[max_index])

            # Store the result (image name and predicted label)
            predicted_labels.append((image_name, highest_label))
        else:
            predicted_labels.append((image_name, "No object detected"))

# Print all results
print(predicted_labels)
