from keras.models import load_model
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import numpy as np
import tensorflow as tf
#Video Feed
cap = cv2.VideoCapture(0)
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
            # print(pose1)

            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]for landmark in pose1]).flatten())
            
            model = load_model("model_instrusion.h5")
        
            # Check if the person is standing based on the ankle y-coordinate difference
            prediction = model.predict(pose_row.reshape(1,33*(4)),verbose=None)
            prediction_p = tf.nn.softmax(prediction)
            yhat = np.argmax(prediction_p)
            if yhat== 0:  # Adjust the threshold as needed
                text = "Normal"
            elif yhat==1:
                text = "Looking back left"
            elif yhat==2:
                text="Looking back right"
            else:
                text="Suspicious"
            
            # Display the standing status on the frame
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display the frame
            cv2.imshow("Atm intrusion detection", frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()
