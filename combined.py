from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import time
from urllib.request import urlopen
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import numpy as np
import requests
import tensorflow as tf
import time
from model.yolo_model import YOLO

def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image
  
def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        if cl==0:
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 1,
                        cv2.LINE_AA)
   
    print()



# Send Warning of detecting suspicion
def send_warning(frame):
    word = 'Hello'
    # response = requests.post("http://127.0.0.1:5000/data", data=word)
    cv2.putText(frame, "Suspicious", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Video Feed


def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)
    global count
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    if boxes is not None:
        
        if(len(boxes)>1):
            count+=1
            
          #show out if more than one person detected

        draw(image, boxes, scores, classes, all_classes)

    return image




def run_model(path):

    # Initializing threshold values
    looking_back_count = 0
    suspicious_count = 0
    prev_state = None
    suspicious_time = 0
    suspicious_frame_count = 0
    looking_back_time=0

    fps = '2'
    yolo = YOLO(0.6, 0.5)
    all_classes=["person"]

    with mp_pose.Pose(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as pose:
        while True:
            start_time = time.time()
            img_arr = np.array(bytearray(urlopen(path).read()),dtype=np.uint8)
            frame = cv2.imdecode(img_arr,-1)
            resized_frame = cv2.resize(frame,(368,656))
            if resized_frame is not None:
                #Recolor to RGB
                image = cv2.cvtColor(resized_frame,cv2.COLOR_BGR2RGB)
                image.flags.writeable = False


                #Mediapipe Pose estimation
                results = pose.process(image)
                
                #Recolor to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                try:
                    pose1 = results.pose_landmarks.landmark
                    # print(pose1)

                    pose_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]for landmark in pose1]).flatten()
                    
                    model = load_model("C:/Users/rehan/OneDrive/Desktop/OSPI/ieee-ospi-atm-intrustion-detection-system/model_intrusion_final.h5")
                    
                    # Check if the person is standing based on the ankle y-coordinate difference
                    prediction = model.predict(pose_row.reshape(1,33*(4)),verbose=None)
                    prediction_p = tf.nn.softmax(prediction)
                    yhat = np.argmax(prediction_p)
                    image = detect_image(frame, yolo, all_classes)
                    if count>2:
                        
                        cv2.putText(image, "multiple people detected", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

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

                    suspicious_time = suspicious_frame_count/float(fps)
                    looking_back_time = looking_back_count/float(fps)

                    if looking_back_count>=3 or suspicious_count>=3 or suspicious_time>5 or looking_back_time>5:
                        send_warning(resized_frame)

                    prev_state = text
                    
                    # Display the standing status on the frame
                    cv2.putText(resized_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # cv2.putText(resized_frame, fps, (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    mp_drawing.draw_landmarks(resized_frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
                except:
                    pass
                # Display the frame
                
                cv2.imshow("Atm intrusion detection", resized_frame)
                cv2.resizeWindow("Atm intrusion detection",368,656)
                end_time = time.time()
                fps = str(1.0/(end_time-start_time))
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
            else:
                break

    cv2.destroyAllWindows()
    print("Looking back count - "+str(looking_back_count))
    print("Suspicious position time - "+str(suspicious_count))
    print("Suspicious positon time - "+str(suspicious_time))

run_model('http://192.168.45.89:8080/shot.jpg')
