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
import pytesseract


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
# # DOCKER SERVICE
MODEL_FILENAME = "/app/result_model_letter.h5"
MODEL_LABELS_FILENAME = "/app/model_labels.dat"
MODEL_CLASSIFICATION_FILENAME = "/app/captcha_classification_model.hdf5"
MODEL_CLASSIFICATION_LABELS_FILENAME = "/app/model_classification_labels.dat"

# # PYTHON SERVICE
# MODEL_FILENAME = "nfse-srv-python/result_model_letter.h5"
# MODEL_LABELS_FILENAME = "nfse-srv-python/model_labels.dat"
# MODEL_CLASSIFICATION_FILENAME = "nfse-srv-python/captcha_classification_model.hdf5"
# MODEL_CLASSIFICATION_LABELS_FILENAME = "nfse-srv-python/model_classification_labels.dat"

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

with open(MODEL_CLASSIFICATION_LABELS_FILENAME, "rb") as f:
    lb_class = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)
classification = load_model(MODEL_CLASSIFICATION_FILENAME)

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

    access_token = request.headers.get('Authorization')[7:]
    logger.info(access_token)
    logger.info(uaa_service)
    scope = uaa_service['xsappname']
    security_context = xssec.create_security_context(access_token, uaa_service)
    # escopo PRE: nfse!t2893.nfse_foreigncall
    # escopo DEV: nfse!t2116.nfse_foreigncall
    isAuthorized = security_context.check_scope(scope+'.nfse_foreigncall') #.nfse_admin para teste local

    logger.info('Authorization found')
    logger.info(isAuthorized)
    if not isAuthorized:
        logger.info('Unauthorized !!')
        abort(403)

    logger.info('Authorization successful')
    try:
        ## FILE
        # name1 = request.files['file']
        # b64_string = base64.b64encode(name1.read())

        # B64 STRING
        # name1 = request.form['string']
        name1 = request.form
        key = name1.to_dict(flat=False)
        key, value = list(key.items())[0]
        if key == 'string':
            b64_string = name1['string']

            # read b64 image
            img = imread(io.BytesIO(base64.b64decode(b64_string)))

            # Classify image
            captcha_class = processing_lab.classify_captcha(img)
            captcha_image = processing_lab.resize_to_fit(captcha_class, 20, 20)
            # Turn the single image into a 4d list of images to make Keras happy
            captcha_image = np.expand_dims(captcha_image, axis=2)
            captcha_image = np.expand_dims(captcha_image, axis=0)
            # Ask the neural network to make a prediction
            prediction = classification.predict(captcha_image)
            # Convert the one-hot-encoded prediction back to a normal letter
            captcha_class = int(lb_class.inverse_transform(prediction)[0])
            # print(captcha_class)

            predictions = []

            # apply different model depending on the captcha type
            if captcha_class in [1, 2]:
                img = processing_lab.model1(img)
            elif captcha_class == 7:
                img = processing_lab.model2(img)
            elif captcha_class == 8:
                img = processing_lab.model3(img)

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
                captcha_text = captcha_text.replace("_", "")
                # filename = name1.filename
                # Find real captcha name
                # end = filename.rfind('.')  # last occurence of '.'
                # real = filename[:end]
            return {'Predicted': captcha_text}
        if key == 'code':
            code = name1['code']
            captcha = processing_lab.model4(code, 30)

            return {'Predicted': str(captcha)}
    except:
        return {'Predicted': 'error!!'}

if __name__ == '__main__':
    # app.run(debug=True, use_debugger=False, use_reloader=False)
    #port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))