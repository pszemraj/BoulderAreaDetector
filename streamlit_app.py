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
import pathlib
import platform
import numpy as np

if platform.system() == "Windows":
    # model originally saved on Linux, strange things happen
    print("on Windows OS - adjusting PosixPath")
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# App title
st.title("NatureGeoDiscoverer MVP: Detect Bouldering Areas")
st.markdown("by Peter Szemraj | [GitHub](https://github.com/pszemraj)")
with st.beta_container():
    st.header("Basic Instructions")
    st.markdown("*This app assesses a satellite/arial image of land and decides whether it is suitable for "
                "outdoor rock climbing.*")
    st.markdown("- Choose an option to use the model to assess an image")
    st.markdown("- If you are experiencing a shortage of satellite images to test, do not fear. This folder ["
                "here](https://www.dropbox.com/sh/0hz4lh9h8v30a8d/AACFwlIAvdnDdc6RvrcXVpnsa?dl=0) contains "
                "images not used in the creation of the model")
st.markdown("---")

# Fxn
@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

def predict(img, img_path):
    # Display the test image
    st.image(img, caption="Chosen Image to Analyze", use_column_width=True)

    # Temporarily displays a message while executing
    with st.spinner('thinking...'):
        time.sleep(5)
        # Load model and make prediction
    try:
        path_to_model = r"Res101_cls_set4.pkl"
        model = load_learner(path_to_model, cpu=True)
    except:
        print("unable to load locally. downloading model file")
        model_backup = "https://www.dropbox.com/s/41013dx67lk9ztj/resnet101classifier02_naturegeodiscoverer.pkl?dl=1"
        model_response = requests.get(model_backup)
        model = load_learner(BytesIO(model_response.content), cpu=True)


    pred_class, pred_items, pred_prob = model.predict(img_path)
    prob_np = pred_prob.numpy()

    # Display the prediction
    if str(pred_class) == 'climb_area':
        st.balloons()
        st.markdown("##Submitted img is **most likely a climbing area**")
    else:
        st.markdown("##Area in submitted image *most likely* does not make a great climbing area")


# Image source selection
option1_text = 'Use an example image'
option2_text = 'Upload a custom image for analysis'
option = st.radio('Choose an option to continue:', [option1_text, option2_text])


if option == option1_text:
    # Test image selection
    working_dir = os.path.join(os.getcwd(), "test_images")
    test_images = natsorted([f for f in os.listdir(working_dir) if os.path.isfile(os.path.join(working_dir, f))])
    test_image = st.selectbox('Please select a test image:', test_images)

    if st.button('Analyze!'):
        # Read the image
        file_path = os.path.join(working_dir, test_image)
        img = skimage.io.imread(file_path)
        img = resize(img, (256, 256))

        # Predict and display the image
        predict(img, file_path)
else:
    image_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])

    if image_file is not None:
        base_img = np.array(load_image(image_file))
        file_details = {"Filename": image_file.name,
                        "FileType": image_file.type,
                        "FileSize": image_file.size}
        st.write(type(base_img))
        st.write(base_img.shape)
        img = resize(base_img, (256, 256))
        st.image(img, width=250, height=250)
        save_path = os.path.join(os.getcwd(), "custom_picture.png")
        skimage.io.imsave(save_path)
        # Predict and display the image
        predict(img, save_path)
