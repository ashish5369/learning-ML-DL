import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify

# used for making the weba application
# streamlit run main.py  ----> this is how we'll be hosting out website in the local


# set title
st.title('pneumonia classification')

# set header
st.header('please upload an image of a chest X-Ray')

# upload filee
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
# here we are specify which kind of file we want to be allowed to be uploaded

# load classifier
model = load_model('/home/golu/PycharmProjectspneumoniaTraining/model/pneumonia_classifier.h5')

# load class names
with open('/home/golu/PycharmProjectspneumoniaTraining/model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    # here using the labels we have stored the label that is  normal and pneumonia and these ar enot stored in
    # class_names

    # here we are splitting as we have the data stored ina way like first the nteger 0 adn 1 and then the name in the
    # labels.txt sso we are giving space and then [1] this means to store the first 2nd word which is the class i.e.
    # pneumonia or normal display image
    f.close()
# print(class_names)

# dsiplay image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
# this will show the image uploaded by the user to the user on the website
# classify image
class_name, conf_score = classify(image, model, class_names)

# write classification
st.write("## {}".format(class_name))
st.write("### score: {}".format(conf_score))
# this is going to return thr class name and the confidence score,and ## i and ### is used to setermine the size of
# the word
