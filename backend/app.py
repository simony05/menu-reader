from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/api', methods=["GET", 'POST'])
def post_data():
    content_type = request.headers.get('Content-Type')
    if content_type != 'application/json':
        return jsonify({'error': 'Not json'}), 415
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        return jsonify({'message': 'Data received successfully!'}), 200
    return jsonify({'message': 'Data received successfully!'}), 200

if __name__ == '__main__':
    app.run(debug=True)