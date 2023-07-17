import cv2
import socket
import pickle
import struct
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
import numpy as np


def run_model(path):
    
    
    
  
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

        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Close the client socket and the server socket
    client_socket.close()
    server_socket.close()




receive_live_video()

