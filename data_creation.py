import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import csv
import os
import numpy as np

num_coords = 33

landmarks = ['Class']
for i in range(1,num_coords+1):
    landmarks+=['x{}'.format(i),'y{}'.format(i),'z{}'.format(i),'v{}'.format(i)]

with open('coords.csv', mode='a', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)
len(landmarks)

class_name="Looking back_left"
cap = cv2.VideoCapture('looking_back_left.mp4')
with mp_pose.Pose(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            cv2.namedWindow("Mediapipe_Feed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Mediapipe_Feed", 432,768)
           #Recolor to RGB
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable = False


            #Mediapipe Pose estimation
            results = pose.process(image)

            #Recolor to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            pose1 = results.pose_landmarks.landmark

            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]for landmark in pose1]).flatten())
            pose_row.insert(0,class_name)

            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(pose_row)

            # print(results)
            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Mediapipe_Feed',image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
