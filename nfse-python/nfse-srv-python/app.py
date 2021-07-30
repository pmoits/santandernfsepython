from flask import Flask, request
import os
import processing_lab
import base64
from imageio import imread
import io
import sklearn
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from cfenv import AppEnv
from sap import xssec
import json

#
app = Flask(__name__)
env = AppEnv()
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
#
@app.route('/')
def home():
    text = 'Hello World'
    return text
#
@app.route('/ocr', methods=['POST'])
def predict_text():
    uaa_service = env.get_service(label='xsuaa').credentials
   
    access_token = request.headers.get('authorization')[7:]

    security_context = xssec.create_security_context(access_token, uaa_service)


    if 'authorization' not in request.headers:

        print('Missing authorization key !!')
    isAuthorized = security_context.check_scope('openid')
    if not isAuthorized:
        print('Unauthorized')
    else:
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
            return {'Predicted':captcha_text,'uaa_service':uaa_service,
                    'access_token':access_token,'security_context':security_context}
        except:
            return 'Captcha error !!'


#
if __name__ == '__main__':
    # app.run(debug=True, use_debugger=False, use_reloader=False)
    #port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))