import os
import time
from io import BytesIO

import PIL.Image
import matplotlib.image as mpimg
import numpy as np
import requests
import matplotlib.pyplot as plt
from skimage import io
import skimage
import streamlit as st
import torch
from fastai.vision.all import *
from natsort import natsorted
from os.path import join
from skimage.transform import resize

# App title
st.title("NatureGeoDiscoverer MVP: Detect Bouldering Areas")

def display_img(picture):
    fig = plt.figure(figsize=(5, 5), tight_layout=True, clear=True)
    # ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    plt.axis('off')
    plt.ioff()
    ax = plt.axes()
    plt.imshow(picture, norm=matplotlib.colors.Normalize(),
               interpolation="lanczos", alpha=1, zorder=1)
    plt.show()

def predict(img, img_path):
    # Display the test image
    display_img(img)

    # Temporarily displays a message while executing
    with st.spinner('thinking...'):
        time.sleep(3)

    # Load model and make prediction
    try:
        model = load_learner('resnet101classifier02_naturegeodiscoverer.pkl')
    except:
        print("unable to load locally. downloading model file")
        model_backup = "https://www.dropbox.com/s/41013dx67lk9ztj/resnet101classifier02_naturegeodiscoverer.pkl?dl=1"
        model_response = requests.get(model_backup)
        model = load_learner(BytesIO(model_response.content))


    pred_class = model.predict(img_path)[0]
    pred_prob = round(torch.max(model.predict(img_path)[2]).item() * 100)

    # Display the prediction
    if str(pred_class) == 'mtp':
        st.success("Image is a climbing area, confidence level is" + str(pred_prob) + '%.')
    else:
        st.success("Area in image not great for climbing. confidence level is " + str(pred_prob) + '%.')


# Image source selection
option = st.radio('', ['Analyze example image', 'Analyze custom image'])

if option == 'Analyze example image':

    # Test image selection
    working_dir = os.path.join(os.getcwd(), "test_images")
    test_images = natsorted([f for f in os.listdir(working_dir) if os.path.isfile(os.path.join(working_dir, f))])
    test_image = st.selectbox(
        'Please select a test image:', test_images)

    # Read the image
    file_path = os.path.join(working_dir, test_image)
    img = skimage.io.imread(file_path)
    img = resize(img, (256, 256))

    # Predict and display the image
    predict(img, file_path)

else:
    url = st.text_input("Please input a url:")

    if url != "":
        try:
            # Read image from the url
            response = requests.get(url)
            pil_img = PIL.Image.open(BytesIO(response.content))
            display_img = np.asarray(pil_img)  # Image to display

            # Transform the image to feed into the model
            img = pil_img.convert('RGB')
            save_path = os.path.join(os.getcwd(), "test_picture.png")
            skimage.io.imsave(save_path)
            # Predict and display the image
            predict(img, save_path)

        except:
            st.text("Invalid url!")
