import cv2
import socket
import pickle
import struct

def send_live_video(server_ip, server_port):
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Connect to the server
    client_socket.connect((server_ip, server_port))
    print("Connected to server:", server_ip, server_port)
    
    # Create a VideoCapture object to capture video from the webcam
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam
    
    while True:
        # Read a frame from the webcam
        ret, frame = video_capture.read()
        
        # Serialize the frame using pickle
        serialized_frame = pickle.dumps(frame)
        
        # Get the size of the serialized frame
        frame_size = struct.pack("!I", len(serialized_frame))
        
        # Send the frame size to the server
        client_socket.sendall(frame_size)
        
        # Send the serialized frame to the server
        client_socket.sendall(serialized_frame)
        
        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the VideoCapture and close the client socket
    video_capture.release()
    client_socket.close()


server_ip = "192.168.29.238"  
server_port = 8000  
send_live_video(server_ip, server_port)
