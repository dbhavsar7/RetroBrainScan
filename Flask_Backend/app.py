# Flask backend
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/message', methods=['POST'])
def receive_message():
    title = 'you should get this message back'
    message = request.json.get('message')
    print("Received:", message)
    return jsonify({"response": title})

if __name__ == '__main__':
    app.run(debug=True)
