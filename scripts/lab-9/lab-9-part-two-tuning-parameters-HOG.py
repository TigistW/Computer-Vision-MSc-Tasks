import cv2
import time

# -------------------------------
# Initialize HOG + SVM detector
# -------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load an image for testing
img = cv2.imread('images/pedesterians.jpg')  # Replace with your image path
if img is None:
    raise FileNotFoundError("Image not found! Check the path.")

# Resize image for display
img = cv2.resize(img, (640, 480))

# List of parameters to test
winStride_values = [(4,4), (8,8), (16,16)]
scale_values = [1.03, 1.05, 1.1]

# Iterate over parameter combinations
for winStride in winStride_values:
    for scale in scale_values:
        test_img = img.copy()
        start_time = time.time()
        
        rects, weights = hog.detectMultiScale(
            test_img,
            winStride=winStride,
            padding=(8, 8),
            scale=scale
        )
        
        # Draw detections
        for (x, y, w, h) in rects:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        end_time = time.time()
        elapsed = end_time - start_time

        # Display results
        cv2.imshow(f'HOG Detection winStride={winStride}, scale={scale}', test_img)
        print(f"Parameters: winStride={winStride}, scale={scale} | Detections: {len(rects)} | Time: {elapsed:.3f}s")
        cv2.waitKey(0)

cv2.destroyAllWindows()
