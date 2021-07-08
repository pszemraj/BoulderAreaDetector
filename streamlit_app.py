import pathlib
import time
from io import BytesIO
import os
from os.path import join

import skimage
import streamlit as st
from fastai.vision.all import *
from natsort import natsorted
from skimage import io
from skimage.transform import resize

# account for posixpath
if platform.system() == "Windows":
    # model originally saved on Linux, strange things happen
    print("on Windows OS - adjusting PosixPath")
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# App title and intro
supplemental_dir = os.path.join(os.getcwd(), "info")
fp_header = os.path.join(supplemental_dir, "climb_area_examples.png")

st.title("NatureGeoDiscoverer MVP: Detect Bouldering Areas")
st.markdown("by [Peter Szemraj](https://peterszemraj.ch/) | [GitHub](https://github.com/pszemraj)")
with st.beta_container():
    st.header("Basic Instructions")
    st.markdown("*This app assesses a satellite or arial image of land chosen by the user (scroll down) and "
                "decides whether it is suitable for outdoor bouldering.*")
st.markdown("---")
st.markdown("**Examples of Images in the *climb area* class**")
st.image(skimage.io.imread(fp_header))
st.markdown("---")
with st.beta_container():
    st.subheader("Sample Images")
    st.markdown("If lacking satellite images, the dropbox folder ["
                "here](https://www.dropbox.com/sh/0hz4lh9h8v30a8d/AACFwlIAvdnDdc6RvrcXVpnsa?dl=0) contains "
                "images that were not used for model training.")


# Fxn
@st.cache
def load_image(image_file):
    # loads uploaded images
    img = Image.open(image_file)
    return img


def load_best_model():
    try:
        path_to_archive = r"model-resnetv2_50x1_bigtransfer.zip"
        best_model_name = "model-resnetv2_50x1_bigtransfer.pkl"
        shutil.unpack_archive(path_to_archive)
        best_model = load_learner(join(os.getcwd(), best_model_name), cpu=True)
    except:
        print("unable to load locally. downloading model file")
        model_b_best = "https://www.dropbox.com/s/9c1ovx6dclp8uve/model-resnetv2_50x1_bigtransfer.pkl?dl=1"
        best_model_response = requests.get(model_b_best)
        best_model = load_learner(BytesIO(best_model_response.content), cpu=True)
    st.write("loaded model")
    return best_model


def load_mixnet_model():
    try:
        mixnet_name = r"model-mixnetXL-20epoch.pkl"
        model = load_learner(join(os.getcwd(), mixnet_name), cpu=True)
    except:
        print("unable to load locally. downloading model file")
        model_backup = "https://www.dropbox.com/s/bwfar78vds9ou1r/model-mixnetXL-20epoch.pkl?dl=1"
        model_response = requests.get(model_backup)
        model = load_learner(BytesIO(model_response.content), cpu=True)
    st.write("loaded model")

    return model


# load the trained model

# use_best_model = False  # takes a bit longer to load because it needs to be unzipped
# st.write("got past setting")
# if use_best_model:
#     model = load_best_model()
# else:
#     model = load_mixnet_model()
# st.write("loaded a model")


# prediction function
def predict(img, img_flex):
    # NOTE: it's called img_flex because it can either be an object itself, or a path to one
    # Display the test image
    st.image(img, caption="Chosen Image to Analyze", use_column_width=True)

    # Temporarily displays a message while executing
    with st.spinner('thinking...'):
        time.sleep(3)
        model_pred = load_mixnet_model()
        # make prediction
        if not isinstance(img_flex, str):
            fancy_class = PILImage(img_flex)
            model_pred.precompute = False
            pred_class, pred_items, pred_prob = model_pred.predict(fancy_class)
        else:
            # loads from a file so it's fine
            pred_class, pred_items, pred_prob = model_pred.predict(img_flex)
        prob_np = pred_prob.numpy()

    # Display the prediction
    if str(pred_class) == 'climb_area':
        st.balloons()
        st.subheader("Area in test image is good for climbing! {}% confident.".format(round(100 * prob_np[0],
                                                                                            2)))
    else:
        st.subheader("Area in test image not great for climbing :/ - {}% confident.".format(
            100 - round(100 * prob_np[0], 2)))


# Image source selection
option1_text = 'Use an example image'
option2_text = 'Upload a custom image for analysis'
option = st.radio('Choose a method to load an image:', [option1_text, option2_text])

# provide different options based on selection
if option == option1_text:
    # Test image selection
    working_dir = os.path.join(os.getcwd(), "test_images")
    test_images = natsorted(
        [f for f in os.listdir(working_dir) if os.path.isfile(os.path.join(working_dir, f))])
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
    if st.button('Analyze!'):
        if image_file is not None:
            file_details = {"Filename": image_file.name,
                            "FileType": image_file.type,
                            "FileSize": image_file.size}
            base_img = load_image(image_file)
            img = base_img.resize((256, 256))
            img = img.convert("RGB")
            # Predict and display the image
            predict(img, img)
st.markdown("---")
st.subheader("How it Works:")
st.markdown("**BoulderAreaDetector** uses Convolutional Neural Network (CNN) trained on a labeled dataset ("
            "approx. 3000 satellite images, each 256x256 in two classes) with two classes. More "
            "specifically, the primary model is [MixNet-XL](https://paperswithcode.com/method/mixconv).")
