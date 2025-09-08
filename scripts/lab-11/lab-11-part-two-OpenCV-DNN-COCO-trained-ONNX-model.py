import cv2
import numpy as np

# Load YOLOv8 ONNX (COCO-pretrained)
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640,640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()

    # Minimal post-processing example
    for det in outputs[0]:
        scores = det[5:]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id] * det[4])
        if conf > 0.5:
            cx, cy, bw, bh = det[0:4] * np.array([w, h, w, h])
            x1, y1 = int(cx-bw/2), int(cy-bh/2)
            x2, y2 = int(cx+bw/2), int(cy+bh/2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{class_id}:{conf:.2f}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("YOLOv8 ONNX + OpenCV DNN", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
