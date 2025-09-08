import cv2

# -------------------------------
# Initialize HOG + SVM detector
# -------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# -------------------------------
# Open Video File
# -------------------------------
video_path = 'images/pedestrians_walking.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("Could not open video file: " + video_path)

print("Press 'q' to quit video processing.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot fetch frame.")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Detect pedestrians in the frame
    (rects, weights) = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    # Draw bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Video Pedestrian Detection", frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
