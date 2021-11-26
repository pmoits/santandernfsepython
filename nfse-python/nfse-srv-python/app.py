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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#
app = Flask(__name__)
env = AppEnv()
flask_logging.init(app, logging.INFO)
#
app.config['DEBUG'] = True
UPLOAD_FOLDER = r'E:\Users\Daniel\OneDrive\CaptchaML\templates'
ALLOWED_CAPTCHAS = [1,2,7,8,11,10,9]
app.secret_key = "secret key"
#
# # DOCKER SERVICE
MODEL_FILENAME = "/app/result_model_letter.h5"
MODEL_LABELS_FILENAME = "/app/model_labels.dat"
MODEL_UPPER = "/app/result_model_letter_upper.h5"
MODEL_LABELS_FILENAME_UPPER = "/app/model_labels_upper.dat"
MODEL_CTC_FILENAME = "/app/result_model_ctc.h5"

# # PYTHON SERVICE
# MODEL_FILENAME = "nfse-srv-python/result_model_letter.h5"
# MODEL_LABELS_FILENAME = "nfse-srv-python/model_labels.dat"
# MODEL_CLASSIFICATION_FILENAME = "nfse-srv-python/captcha_classification_model.hdf5"
# MODEL_CLASSIFICATION_LABELS_FILENAME = "nfse-srv-python/model_classification_labels.dat"
# MODEL_CTC_FILENAME = "nfse-srv-python/result_model_ctc.h5"

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# with open(MODEL_CLASSIFICATION_LABELS_FILENAME, "rb") as f:
#     lb_class = pickle.load(f)

with open(MODEL_LABELS_FILENAME_UPPER, "rb") as f:
    lb_upper = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)
# classification = load_model(MODEL_CLASSIFICATION_FILENAME)
model_upper = load_model(MODEL_UPPER)

#Load CTC model
#Load CTCLayer class
class CTCLayer(layers.Layer):
    def __init__(self, name=None,**kwargs):
        super().__init__(name=name,**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred
model_ctc = load_model(MODEL_CTC_FILENAME,custom_objects={'CTCLayer': CTCLayer})

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
        passed_keys = name1.to_dict(flat=False)
        key, value = list(passed_keys.items())[0]
        if key == 'string':
            b64_string = name1['string']

            # read b64 image
            img = imread(io.BytesIO(base64.b64decode(b64_string)))
            predictions = []

            # apply different model depending on the captcha type
            if int(name1['id']) in [1, 2]:
                img = processing_lab.model1(img)
            elif int(name1['id']) == 7:
                img = processing_lab.model2(img)
            elif int(name1['id']) == 8:
                img = processing_lab.model3(img)
            elif int(name1['id']) == 11:
                img = processing_lab.model6(img)
                for letter in img:
                    # rgb = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB)
                    # Re-size the letter image to 20x20 pixels to match training data
                    letter_image = processing_lab.resize_to_fit(letter, 20, 20)

                    # Turn the single image into a 4d list of images to make Keras happy
                    letter_image = np.expand_dims(letter_image, axis=2)
                    letter_image = np.expand_dims(letter_image, axis=0)

                    # Ask the neural network to make a prediction
                    prediction = model_upper.predict(letter_image)

                    # Convert the one-hot-encoded prediction back to a normal letter
                    letter = lb_upper.inverse_transform(prediction)[0]
                    predictions.append(letter)

                    # Get captcha's text
                    captcha_text = "".join(predictions)
                    captcha_text = captcha_text.replace("_", "")
                return {'Predicted': captcha_text}
            elif int(name1['id']) == 9:
                img = processing_lab.model5(img, 50, 200)
                d = {'8': 1, '6': 2, '2': 3, '7': 4, '9': 5, '1': 6, '5': 7, '3': 8, '10': 9,
                     '4': 0}  # num_to_char dict
                # Get the prediction model by extracting layers till the output layer
                prediction_model = keras.models.Model(
                    model_ctc.get_layer(name="image").input, model_ctc.get_layer(name="dense2").output
                )

                # print(prediction_model.summary())
                pred = prediction_model.predict(img)

                input_len = np.ones(pred.shape[0]) * pred.shape[1]
                # Use greedy search. For complex tasks, you can use beam search
                results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :6]
                # Iterate over the results and get back the text
                output_text = []
                for res in results:
                    res = list(map(str, list(np.array(res))))
                    res = list(map(d.get, res))
                    output_text.append(res)
                return {'Predicted': ''.join(map(str, output_text[0]))}

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

            return {'Predicted': captcha_text}

        elif int(name1['id']) not in ALLOWED_CAPTCHAS:
            return {'Predicted': 'Captcha ID not allowed!!'}

        if key == 'code' and int(name1['id']) == 10:
            code = name1['code']
            captcha = processing_lab.model4(code, 30)
            return {'Predicted': str(captcha)}
        elif key == 'code' and int(name1['id']) != 10:
            return {'REQUEST ERROR': 'Wrong ID for key CODE!! ID must be 10 but found value: ' + str(name1['id'])}

    except Exception as e:
        logger.info('ERROR MSG')
        logger.info(e)
        # Dealing with errors
        if 'id' not in passed_keys:  # not passing ID key
            return {'REQUEST ERROR': 'Missing ID key!!'}

        elif passed_keys['id'][0] == '':  # passing ID with no value
            return {'REQUEST ERROR': 'ID key passed with no value inside!!'}

        elif int(name1['id']) not in ALLOWED_CAPTCHAS:  # not passing an valid ID
            return {'REQUEST ERROR': 'Captcha ID not allowed, verify ID value!! Passed ID was: ' + str(name1['id'])}

        else:
            return {'PYTHON ERROR': 'There was an error reading the image, check python log and b64 string!!',
                    'BASE64': b64_string,'PYTHON ERROR MSG':str(e)}


if __name__ == '__main__':
    # app.run(debug=True, use_debugger=False, use_reloader=False)
    #port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))