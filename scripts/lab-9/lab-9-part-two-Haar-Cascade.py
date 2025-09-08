import cv2
import numpy as np

# -------------------------------
# STEP 1: Haar Cascade - Face Detection
# -------------------------------
print("Running Haar Cascade Face Detection...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread('images/face_image.jpg')

if img is None:
    raise FileNotFoundError("Image not found! Please check the path.")

print("Original image shape:", img.shape)

# Resize for display (width=800 px, keeping aspect ratio)
scale_width = 800
scale = scale_width / img.shape[1]
resized = cv2.resize(img, (scale_width, int(img.shape[0] * scale)))

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Detected Faces', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Haar Cascade Face Detection completed.")