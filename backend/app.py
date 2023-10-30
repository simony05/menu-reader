from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from recognition import image64, image_to_text
from translate import translate
#from classify import normalize, classify

app = Flask(__name__)
CORS(app)

app.config["IMAGE_UPLOADS"] = "C:/Flask/Upload/"

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Not file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file:
        texts = image_to_text(file)
        translated_text = []
        for text in texts:
            translated_text.append(translate(text))
        return jsonify({'message': translated_text}), 200

if __name__ == '__main__':
    app.run(debug=True)