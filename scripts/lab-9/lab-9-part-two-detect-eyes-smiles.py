import cv2

# -------------------------------
# Load Haar Cascades
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Load image
img = cv2.imread('images/face_image_two.jpg')  # replace with your image path
if img is None:
    raise FileNotFoundError("Image not found! Check the path.")

# Resize for display if large
scale_width = 800
scale = scale_width / img.shape[1]
img = cv2.resize(img, (scale_width, int(img.shape[0] * scale)))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------------
# Detect Faces
# -------------------------------
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in faces:
    # Draw rectangle around face
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Region of interest for eyes and smile
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # Detect Eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Detect Smiles
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

# Display the result
cv2.imshow("Face, Eyes & Smile Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
