import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np


# imageOps is used for changing the size of the image


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)

    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalize_image_array = image_array.astype(np.float32) / 127.5 - 1
    # as every pixel goes from 0 to 255 if we divide it by 127.5 we are going to make every icxel to go from 0 to 2
    # so if we subtract from - 1  we are going to habve the data from -1 to 1 so we have normalized our data

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # this 1 , 224 , 224 ,3 means we are going to have 1 image at a time with dimension 224 adn 224 adn the color channel is going to be 3
    data[0] = normalize_image_array
    # make prediction
    prediction = model.predict(data)
    # index = np.argmax(prediction)
    # this way we are goign to the maximm value of the prediction adn this is going to give us the index
    index = 0 if prediction[0][0] > 0.95 else 1
    # we have notced while testing that the confidene value when detecting a pneumnia as pneumonia the confidence
    # value si pretty high but in canse of errors where we are detecting a normal as pneumonia what we are going to
    # do is for the lower of the images where the confidence value i less than 95 we are going to make them bormal ,
    # as if it would have been pneumoi the classfier would be able to detct with more than 95 accuracy so we have
    # done this based on a lot of observations
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# the result has a confusion matrix where we have a classifier that is detecting most of the image as
# pneumonia and this is known as biased classifier and t=this can happen as w ehave the dataset pretty mic lookalike
# and this can be a problem
# the problem is can be fixed  removing the argmax and using a differnet approach

# and changing this and removing argmax with a typical we  were able to improve the confuson matri and get a uch
# better accuracy
