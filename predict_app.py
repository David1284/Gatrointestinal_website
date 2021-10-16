
"Import all the required packages/ dependencies "

import base64
import re
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask



#Create an instance of the flask class using the "app" variable.
app = Flask(__name__)

def get_model():
    """
    :return:
    """
    global model  #Make the model accessible from all functions
    model = load_model('mobilenet_model.h5')  #Calls  the required model
    print("The model has been loaded")





def preprocess_image(image, target_size):
    """
    :param image:
    :param target_size:
    :return: The preprocessed image

    Description: This function actions an instance of a python image library(PIL) image, and a target size for the image
    This function prepossesses the image to the required format it has to be in before passing it to the keras model
    """
    if image.mode != "RGB":   #Checks to see if the image is already in RGB format
        image = image.convert("RGB")  #If its not in RGB, convert it to RGB
    image = image.resize(target_size)  #Resizes the image to the target size

    "It then coverts the image to a Numpy array and expands the dimensions of the image"
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print("Please wait...")
print("Model is loading...")
get_model()  #Load model into memory so model doesnt have to get loaded to memory each time a request comes to our endpoint

@app.route('/Home', methods=["POST"])  #New end point called predict

def predict():
    """
    :return: The MSIMUT or MSS prediction in a jsonify format
    """
    message = request.get_json(force=True)

    "Endpoint is designed to accept json data with at least one key value pair. " \
    "This means an image would have to be uploaded by the user before any output can be produced"


    encoded = message['image']
    image_data = re.sub('^data:image/.+;base64,', '', encoded)
    decoded = base64.b64decode(image_data)   #Decoded the encoded image
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    prediction = model.predict(processed_image).tolist()


    "The purpose of the conditional statements below is to specify what the prediction variables represent" \
    "since we used the sigmoid function the result given by the model is within the range of 0 and 1 and " \
    "doesnt represent the probabilistic outcome of MSS and MSIMUT Individually"
    "Therefore the code below helps calculate the probabilistic outcomes of both possible occurrences. "
    "if the prediction value is greater than 0.5, the image uploaded is more likely to be MSIMUT as the prediction value approaches 1" \
    "If the prediction value is less than 0.5, the image uploaded is more likely to be MSS as the prediction value approaches 0" \

    "i.e"
    "if the prediction value is 0.5, It is Neither MSS or MSIMUT " \
    "if it is 0 it is 100% MSS and if its 1, it's 100% MSIMUT"

    if prediction[0][0] > 0.5:
        prediction[0][0] = (prediction[0][0]/0.5)-1
    elif prediction[0][0] < 0.5:
        prediction[0][0] = prediction[0][0]/0.5
    elif prediction[0][0] == 0.5:
        prediction[0][0] = 0.5
    else:
        return None

    response = {
        'prediction': {
            'MSIMUT': prediction[0][0],
            'MSS': 1 - prediction[0][0]
        }
    }
    try:
        return jsonify(response)
    except Exception as e:
        print(e)
