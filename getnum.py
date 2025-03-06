from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

results=model(r"testing_images\67D.jpg")



for result in results:
    if result.boxes:  # Check if any object is detected
        confs = result.boxes.conf.tolist()  # Get confidence scores
        labels = result.boxes.cls.tolist()  # Get class labels

        # Find the label with the highest confidence
        max_index = confs.index(max(confs))  # Index of highest confidence
        highest_label = labels[max_index]  # Get corresponding class

        print("Predicted label with highest confidence:", int(highest_label))
    else:
        print("No objects detected.")