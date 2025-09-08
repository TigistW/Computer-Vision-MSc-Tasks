import dlib
from imutils import face_utils
import cv2

img = cv2.imread('images/face_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
import urllib.request

url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")

# Extract
import bz2
with bz2.open("shape_predictor_68_face_landmarks.dat.bz2", "rb") as f_in:
    with open("shape_predictor_68_face_landmarks.dat", "wb") as f_out:
        f_out.write(f_in.read())
        
# Load pretrained landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

rects = detector(gray, 1)
for rect in rects:
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    for (x, y) in shape:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

cv2.imshow("Facial Landmarks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
