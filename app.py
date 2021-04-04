import albumentations
import cv2
import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Torch
import torch
import torch.nn as nn

# Some utilites
import numpy as np
from util import base64_to_pil

# RepVGG models
from repvgg import repvgg_model_convert, create_RepVGG_A0, create_RepVGG_A1,create_RepVGG_A2,create_RepVGG_B0,create_RepVGG_B1, create_RepVGG_B2,create_RepVGG_B3

# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

model = create_RepVGG_A0()
print('Model loaded. Check http://127.0.0.1:5000/')


# Trained model saved with model.save()
MODEL_PATH = 'models/feiyang.pth'

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def image_loader(transformer, file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = transformer(image=img)
    img = res['image']
    img = img.astype(np.float32)
    img = img.transpose(2,0,1)
        
    return torch.tensor(img).unsqueeze(0).float() # Insert a "batch" dimension

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        file_path = "./uploads/image.jpg"
        # Save the image to ./uploads
        img.save(file_path)
        
        # Data transformation
        data_transforms = albumentations.Compose([albumentations.Resize(224, 224),])

        # Make prediction
        model.eval()
        logits = model(image_loader(data_transforms, file_path))
        # preds = model_predict(file_path, model)

        # Process your result for human
        preds = logits.sigmoid()
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        maxid = logits[:,:4].argmax(1)
        preds_ett = torch.zeros(logits[:,:4].shape).scatter(1,maxid.unsqueeze(1).cpu(),1.0)
        preds[:,:4] = preds_ett
        outputs = preds.detach().numpy().astype(np.int)
        
        # Serialize the result, you can add additional fields
        return jsonify(ETT_Abnormal=outputs[0,0], ETT_Borderline=outputs[0,1],
                      ETT_Normal=outputs[0,2], NGT_Abnormal=outputs[0,4],
                      NGT_Borderline=outputs[0,5], NGT_IncompletelyImaged=outputs[0,6],
                      NGT_Normal=outputs[0,7], CVC_Abnormal=outputs[0,8],
                      CVC_Borderline=outputs[0,9], CVC_Normal=outputs[0,10],
                      Swan_Ganz_Catheter_Present=outputs[0,11])

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
