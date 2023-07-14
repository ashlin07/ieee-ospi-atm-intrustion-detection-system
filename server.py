import cv2
import socket
import pickle
import struct
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import numpy as np


def receive_live_video(server_port):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to a specific IP address and port
    server_socket.bind(('', server_port))
    
    # Listen for incoming connections
    server_socket.listen(1)
    print("Server is listening for incoming connections...")
    
    # Accept a client connection
    client_socket, client_address = server_socket.accept()
    print("Connected to client:", client_address)
    
    # Create an OpenCV window to display the video
    cv2.namedWindow("Received Video", cv2.WINDOW_NORMAL)
    mp_pose = mp.solutions.pose
    
    # Load the Pose model
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while True:
        # Receive the frame size from the client
        frame_size_data = client_socket.recv(4)
        frame_size = struct.unpack("!I", frame_size_data)[0]
        
        # Receive the serialized frame from the client
        serialized_frame = b""
        while len(serialized_frame) < frame_size:
            serialized_frame += client_socket.recv(frame_size - len(serialized_frame))
        
        # Deserialize the frame using pickle
        frame = pickle.loads(serialized_frame)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with the Pose model
        results = pose.process(frame_rgb)
        
        # Check if any pose landmarks are detected
        
            
            # Check if the person is standing based on the ankle y-coordinate difference
            prediction = model.predict(X_dataset[random_index].reshape(1,33*(4)),verbose=None)
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
        cv2.imshow("Standing Position Detection", frame)

        
        
        
        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Close the client socket and the server socket
    client_socket.close()
    server_socket.close()



server_port = 8000  # Replace with your server port


receive_live_video(server_port)

