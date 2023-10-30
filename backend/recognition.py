import requests
import json
from aws_requests_auth.boto_utils import BotoAWSRequestsAuth
from PIL import Image
import base64
import io
from dotenv import load_dotenv
import os

def configure():
	load_dotenv()

def image64(path):
    with Image.open(path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded_img

def image_to_text(image):
    configure()
    auth = BotoAWSRequestsAuth(aws_host=f"{os.getenv('aws_auth')}.execute-api.us-west-1.amazonaws.com",
                           aws_region='us-west-1',
                           aws_service='execute-api')

    url = f"https://{os.getenv('aws_auth')}.execute-api.us-west-1.amazonaws.com/prod/general-ocr-standard-ml"
    payload = {
        'img': image64(image)
    }
    response = requests.request("POST", url, data = json.dumps(payload), auth = auth)
    outputs = []
    for output in json.loads(response.text):
        outputs.append(output['words'])
    return outputs