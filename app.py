# app.py
from flask import Flask, render_template, request
from plyer import notification

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        notification_text = request.form['notification_text']
        # You can process the notification_text or send it to the Python file here
        # display it as a notification.
        notification.notify(
            title='Notification from Flask App',
            message=notification_text,
            timeout=10  # Notification will disappear after 10 seconds
        )
    return render_template('/index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
