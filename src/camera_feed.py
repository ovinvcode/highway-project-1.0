# src/camera_feed.py
import cv2
import numpy as np

# --- Load YOLO Model ---
print("Loading YOLO model...")
net = cv2.dnn.readNet("yolo_model/yolov3-tiny.weights", "yolo_model/yolov3-tiny.cfg")
classes = []
with open("yolo_model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO model loaded.")


def analyze_frame(frame):
    """
    Analyzes a single video frame to detect objects using YOLO.
    """
    height, width, channels = frame.shape
    detected_objects = []

    # --- Prepare the image for YOLO ---
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # --- Process the YOLO output ---
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Get coordinates for the bounding box
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # --- Draw boxes and labels on the image ---
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            detected_objects.append(label)
            x, y, w, h = boxes[i]
            color = (0, 255, 0) # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return detected_objects, frame