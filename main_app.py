import numpy as np
import streamlit as st
#import cv2
from keras.models import load_model # type: ignore
import csv
from PIL import Image

#loading trained model
model = load_model('traffic_sign_model.h5')

#classes
#read the label.csv file 
# Load the CSV file
with open('labels.csv', 'r') as file:
    reader = csv.reader(file)
    data = [list(row) for row in reader]  # Convert strings to integers

CLASS_NAMES = []
for i in range(len(data)):
    CLASS_NAMES.append(''.join(data[i]))
 

#setting title
st.title("TRAFFIC SIGN CALSSIFICATION BY BADASZ")
st.markdown("Upload an image of a traffic sign")

#uploading image 
traffic_sign_image = st.file_uploader("Choos an image(JPEG)", type = "JPEG")
submit = st.button('CLASSIFY')

if submit:
    if traffic_sign_image is not None:
        #read image
        traffic_sign_image = Image.open(traffic_sign_image)

        #resize and convert image into a numpy array
        traffic_sign_image = traffic_sign_image.resize((32,32))
        np_array = np.array(traffic_sign_image)

        #rescale image for compatibility with model
        img_rescaled = np_array.astype('float32') / 255

        #making the prediction
        prediction = model.predict(np.expand_dims(img_rescaled, axis=0))
        Y_pred = np.argmax(prediction, axis=1)

        st.title(str("The traffic sign is " + CLASS_NAMES[Y_pred[0]])) 
        st.title(str("You're welcome broski" ) )

