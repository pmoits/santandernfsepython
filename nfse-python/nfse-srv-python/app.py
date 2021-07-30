from flask import Flask, request, jsonify, abort
import os
import processing_lab
import base64
from imageio import imread
import io
import sklearn
import logging
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from cfenv import AppEnv
from sap import xssec
from cf_logging import flask_logging


#
app = Flask(__name__)
env = AppEnv()
flask_logging.init(app, logging.INFO)
#
app.config['DEBUG'] = True
UPLOAD_FOLDER = r'E:\Users\Daniel\OneDrive\CaptchaML\templates'
ALLOWED_EXTENSIONS = ['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']
app.secret_key = "secret key"
#
#
# # MODEL_FILENAME = "/app/result_model_letter.h5"
# # MODEL_LABELS_FILENAME = "/app/model_labels.dat"
#
MODEL_FILENAME = "nfse-srv-python/result_model_letter.h5"
MODEL_LABELS_FILENAME = "nfse-srv-python/model_labels.dat"

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)
uaa_service = env.get_service(label='xsuaa').credentials
#
@app.route('/')
def home():
    text = 'Hello World'
    return text
#
@app.route('/ocr', methods=['POST'])
def predict_text():
    logger = logging.getLogger('route.logger')
    logger.info('Someone accessed us')
    if 'Authorization' not in request.headers:
        logger.info('Authorization not found')
        abort(403)
        # return {'message':'Missing authorization key !!'}
    # scopes =['nfse_admin','nfse_foreigncall','.nfse_admin','.nfse_foreigncall',
    #          'nfse!t2116.nfse_admin','nfse!t2116.nfse_foreigncall',
    #          'nfse!t2116']
    access_token = request.headers.get('Authorization')[7:]
    logger.info(access_token)
    logger.info(uaa_service)
    security_context = xssec.create_security_context(access_token, uaa_service)
    isAuthorized = security_context.check_scope('nfse!t2116.nfse_admin')
    # for scope in scopes:
    #     isAuthorized = security_context.check_scope(scope)
    #     logger.info(isAuthorized)

    logger.info('Authorization found')
    logger.info(isAuthorized)
    if not isAuthorized:
        logger.info('Unauthorized !!')
        abort(403)
        # return {'message':'Unauthorized !!'}
    logger.info('Authorization successful')
    try:
        # FILE
        # name1 = request.files['file']
        # b64_string = base64.b64encode(name1.read())

        # B64 STRING
        name1 = request.form['string']
        b64_string = name1

        img = imread(io.BytesIO(base64.b64decode(b64_string)))
        predictions = []
        raw_img = processing_lab.process_1(img)
        img = processing_lab.get_letters(raw_img)

        for letter in img:
            # rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB)
            # Re-size the letter image to 20x20 pixels to match training data
            letter_image = processing_lab.resize_to_fit(letter, 20, 20)

            # Turn the single image into a 4d list of images to make Keras happy
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # Ask the neural network to make a prediction
            prediction = model.predict(letter_image)

            # Convert the one-hot-encoded prediction back to a normal letter
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

            # Get captcha's text
            captcha_text = "".join(predictions)

        # return {'Predicted': captcha_text}
        return {'Predicted':captcha_text}
    except:
        return 'Captcha error !!'


#
if __name__ == '__main__':
    # app.run(debug=True, use_debugger=False, use_reloader=False)
    #port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))