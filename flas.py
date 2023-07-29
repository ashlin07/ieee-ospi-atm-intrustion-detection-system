from flask import Flask, request
app = Flask(__name__)

@app.route('/data', methods=['POST'])
def get_data():
    word = request.data.decode()  # Get data sent with POST request
    print(word)
    return "Success", 200

if __name__ == '__main__':
    app.run(port=5000)
