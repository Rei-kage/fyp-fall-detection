import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_head(frame):
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces in frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda  box: box[2] * box[3])

    head_y = y + h // 2

    return (x, y, w, h, head_y)