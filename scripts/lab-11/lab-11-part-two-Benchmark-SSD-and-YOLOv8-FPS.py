import time
import cv2
import numpy as np

# Load SSD
ssd_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
# Load YOLOv8 ONNX
yolo_net = cv2.dnn.readNetFromONNX('yolov8n.onnx')

video_path = 'images/peoples.mp4'
cap = cv2.VideoCapture(video_path)

ssd_times = []
yolo_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]

    # ---------- SSD ----------
    start = time.time()
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
    ssd_net.setInput(blob)
    detections = ssd_net.forward()
    ssd_times.append(time.time() - start)

    # ---------- YOLOv8 ----------
    start = time.time()
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640,640), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward()
    yolo_times.append(time.time() - start)

cap.release()

print(f"SSD FPS: {1/np.mean(ssd_times):.2f}")
print(f"YOLOv8 FPS: {1/np.mean(yolo_times):.2f}")
