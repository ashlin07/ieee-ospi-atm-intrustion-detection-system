from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit
from plyer import notification
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)

# Variable to keep track of the last time the notification was sent
last_notification_time = 0

@app.route('/', methods=['POST'])
def index():
    global last_notification_time

    if request.method == 'POST':
        notification_text = request.form.get('notification_text')
        # You can process the notification_text or send it to the Python file here
        # display it as a notification.
        notification.notify(
            title='Notification from Flask App',
            message=notification_text,
            timeout=10  # Notification will disappear after 10 seconds
        )

        # Send the notification_text to all connected clients through WebSocket
        socketio.emit('new_notification', {'message': notification_text}, namespace='/notifications')

        # Update the last_notification_time to the current time
        last_notification_time = time.time()

        return "Notification sent successfully.", 200
    return "Method Not Allowed", 405

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080)
