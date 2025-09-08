import cv2
# -------------------------------
# Load Haar Cascade for face detection
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
# -------------------------------
# Initialize HOG + SVM detector for pedestrians
# -------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# -------------------------------
# Open video
# -------------------------------
cap = cv2.VideoCapture('images/pedestrians_walking.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # Detect Faces
    # -------------------------------
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle for faces

    # -------------------------------
    # Detect Pedestrians
    # -------------------------------
    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle for pedestrians

    # Display combined detections
    cv2.imshow("Combined Face + Body Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
