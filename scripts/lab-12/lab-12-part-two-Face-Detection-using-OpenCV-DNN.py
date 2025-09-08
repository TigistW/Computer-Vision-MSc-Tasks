import cv2
import numpy as np
import urllib.request
import os

# URLs for the files
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
caffemodel_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Local paths
prototxt_path = "models/deploy.prototxt"
caffemodel_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

# Download files
urllib.request.urlretrieve(prototxt_url, prototxt_path)
print("Downloaded deploy.prototxt")

urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
print("Downloaded res10_300x300_ssd_iter_140000.caffemodel")

# Load pre-trained model
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
img = cv2.imread('images/face_image_two.jpg')
h, w = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Detected Face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
