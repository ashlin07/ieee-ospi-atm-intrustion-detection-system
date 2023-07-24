from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import numpy as np
import tensorflow as tf
import time

# Send Warning of detecting suspicion
def send_warning(frame):
    cv2.putText(frame, "Suspicious", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Video Feed
def run_model(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Initializing threshold values
    looking_back_count = 0
    suspicious_count = 0
    prev_state = None
    suspicious_time = 0
    suspicious_frame_count = 0

    with mp_pose.Pose(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as pose:
        while cap.isOpened():
            ret,frame = cap.read()
            if ret:  
            #Recolor to RGB
                image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                image.flags.writeable = False


                #Mediapipe Pose estimation
                results = pose.process(image)
                
                #Recolor to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

                pose1 = results.pose_landmarks.landmark
                # print(pose1)

                pose_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]for landmark in pose1]).flatten()
                
                model = load_model("C:/Users/rehan/OneDrive/Desktop/OSPI/ieee-ospi-atm-intrustion-detection-system/model_intrusion_new1.h5")
                
                # Check if the person is standing based on the ankle y-coordinate difference
                prediction = model.predict(pose_row.reshape(1,33*(4)),verbose=None)
                prediction_p = tf.nn.softmax(prediction)
                yhat = np.argmax(prediction_p)

                if yhat== 0:  # Adjust the threshold as needed
                    text = "Normal"
                elif yhat==1:
                    text = "Looking back left"
                    if prev_state!="Looking back left":
                        looking_back_count+=1
                elif yhat==2:
                    text="Looking back right"
                    if prev_state!="Looking back right":
                        looking_back_count+=1
                else:
                    text="Suspicious"
                    suspicious_frame_count+=1
                    if prev_state!="Suspicious":
                        suspicious_count+=1
                suspicious_time = suspicious_frame_count/fps

                if looking_back_count>5 or suspicious_count>5 or suspicious_time>10:
                    send_warning(frame)

                prev_state = text
                
                # Display the standing status on the frame
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
                # Display the frame
                cv2.imshow("Atm intrusion detection", frame)
                
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
            else:
                break
        
    cap.release()
    cv2.destroyAllWindows()
    print(looking_back_count)
    print(suspicious_count)
    print(suspicious_time)

run_model(r'videos\test_video.mp4')

