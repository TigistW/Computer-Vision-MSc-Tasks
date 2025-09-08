import cv2
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Load the trained model
model = load('models/svm_digits.joblib')

# Load the scaler
scaler = load('models/scaler.joblib')

# Build a GUI-based digit recognition app with OpenCV
def digit_recognition_gui(model, scaler=None):
  canvas = np.ones((200, 200), dtype=np.uint8) * 255
  drawing = False

  def draw(event, x, y, flags, param):
    nonlocal drawing
    if event == cv2.EVENT_LBUTTONDOWN:
      drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
      cv2.circle(canvas, (x,y), 8, (0,), -1)
    elif event == cv2.EVENT_LBUTTONUP:
      drawing = False


  cv2.namedWindow("Draw Digit")
  cv2.setMouseCallback("Draw Digit", draw)
  while True:
    cv2.imshow("Draw Digit", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'): # Predict
      img_resized = cv2.resize(canvas, (8,8))
      img_processed = (16 - (img_resized // 16)).flatten().reshape(1, -1)
      if scaler is not None:
        img_processed = scaler.transform(img_processed)

      pred = model.predict(img_processed)
      print("Predicted Digit:", pred[0])
    elif key == ord('c'): # Clear canvas
      canvas[:] = 255
    elif key == 27: # ESC to exit
      break
  cv2.destroyAllWindows()

digit_recognition_gui(model, scaler)