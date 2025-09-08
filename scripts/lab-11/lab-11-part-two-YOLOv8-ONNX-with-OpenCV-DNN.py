import cv2
import numpy as np
from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Export to ONNX
model.export(format="onnx", imgsz=640)
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")
# Load class names (COCO dataset)
classes = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
    "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# ------------------------------
# Load Video
# ------------------------------
video_path = 'images/peoples.mp4'
cap = cv2.VideoCapture(video_path)

# YOLO parameters
input_size = 640
conf_threshold = 0.5
nms_threshold = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ------------------------------
    # Preprocess frame
    # ------------------------------
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass
    outputs = net.forward()  # [num_detections, 85]

    # ------------------------------
    # Post-processing
    # ------------------------------
    outputs = outputs[0] if len(outputs.shape) == 3 else outputs
    boxes = []
    confidences = []
    class_ids = []

    for detection in outputs:
        scores = detection[5:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id] * detection[4])  # objectness * class score

        if confidence > conf_threshold and class_id < len(classes):
            cx, cy, bw, bh = detection[0:4] * np.array([w, h, w, h])
            x1 = int(cx - bw/2)
            y1 = int(cy - bh/2)
            boxes.append([x1, int(cy - bh/2), int(bw), int(bh)])
            confidences.append(confidence)
            class_ids.append(class_id)

    # Apply Non-Max Suppression
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold).flatten()
    else:
        indices = []

    # ------------------------------
    # Draw results
    # ------------------------------
    for i in indices:
        if i >= len(class_ids):
            continue
        x, y, bw, bh = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("YOLOv8 ONNX Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()