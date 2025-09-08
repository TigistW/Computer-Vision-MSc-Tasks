from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # yolov8s.pt for higher accuracy

video_path = 'images/pedestrians_walking.mp4' 
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run prediction
    results = model.predict(source=frame, conf=0.5, show=False)
    # Draw boxes on frame
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # xyxy boxes
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), conf, cls in zip(boxes, confidences, class_ids):
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
