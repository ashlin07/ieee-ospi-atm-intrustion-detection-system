import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and create a predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the video capture object
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the image to gray scale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find faces in the image
    faces = detector(gray)

    all_points = []

    for face in faces:
        landmarks = predictor(image=gray, box=face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            all_points.append((x, y))

            # Draw a circle
            # cv2.circle(img=frame, center=(x, y), radius=1, color=(0, 255, 0), thickness=-1)
    if len(all_points)==0:
        cv2.putText(frame, "Face covered", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Face", frame)

    if cv2.waitKey(delay=1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


