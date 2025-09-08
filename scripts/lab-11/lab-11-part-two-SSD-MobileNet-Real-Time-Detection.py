import cv2
import urllib.request
import os

# URLs of the files
prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
caffemodel_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

# Local filenames
prototxt_file = "deploy.prototxt"
caffemodel_file = "mobilenet_iter_73000.caffemodel"

# Download if not already present
if not os.path.exists(prototxt_file):
    urllib.request.urlretrieve(prototxt_url, prototxt_file)
if not os.path.exists(caffemodel_file):
    urllib.request.urlretrieve(caffemodel_url, caffemodel_file)

# Load the network
net = cv2.dnn.readNetFromCaffe(prototxt_file, caffemodel_file)

# Test with webcam
# cap = cv2.VideoCapture(0)
video_path = 'images/peoples.mp4' 
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{idx} {confidence:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("SSD Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
