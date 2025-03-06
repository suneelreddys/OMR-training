import os
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("runs/detect/train5/weights/best.pt")  # Update with your trained model path

# Path to the folder containing images
image_folder = r"images\val"  # Update with your folder path

# List to store results
predicted_labels = []

# Iterate through all images in the folder
for image_name in os.listdir(image_folder):
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
            predicted_labels.append(highest_label)
        else:
            predicted_labels.append((image_name, "No object detected"))

# Print all results
print(predicted_labels)
