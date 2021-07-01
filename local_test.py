"""
Run the model on local machine. Useful for debugging

"""

import os
import time
from io import BytesIO

# Importing Image class from PIL module
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import requests
import matplotlib.pyplot as plt
from skimage import io
import skimage
import torch
from fastai.vision.all import *
from natsort import natsorted
from os.path import join
import pathlib
import platform

if platform.system() == "Windows":
    # model originally saved on Linux, strange things happen
    print("on windows - adjusting PosixPath")
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

def predict(img, img_flex, print_model=True):
    # Display the test image
    img.show(title="Image to be predicted")
    # Load model and make prediction
    try:
        path_to_model = r"Res101_cls_set4.pkl"
        model = load_learner(path_to_model, cpu=True)
    except:
        print("unable to load locally. downloading model file")
        model_backup = "https://www.dropbox.com/s/41013dx67lk9ztj/resnet101classifier02_naturegeodiscoverer.pkl?dl=1"
        model_response = requests.get(model_backup)
        model = load_learner(BytesIO(model_response.content), cpu=True)

    if not isinstance(img_flex, str):
        # convert image to fast AI PIL object
        fancy_class = PILImage(img_flex)
        model.precompute = False
        pred_class, pred_items, pred_prob = model.predict(fancy_class)
    else:
        # standard case
        # loads from a file so it's fine
        pred_class, pred_items, pred_prob = model.predict(img_flex)
    if print_model: print(model.model)

    prob_np = pred_prob.numpy()
    # Display the prediction
    if str(pred_class) == 'climb_area':
        print("Submitted img is a climbing area, confidence level is {}".format(round(100*prob_np[0]),2))
    else:
        print("Area in submitted image not great for climbing. confidence level is {}".format(round(100*prob_np[0]),2))

# main code. can update the below to load images from a file in a for loop to batch classify

test_image = "test_img_boulder.png"
# Read the image
working_dir = os.path.join(os.getcwd(), "test_images")
file_path = os.path.join(working_dir, test_image)
img = load_image(file_path)
img = img.resize((256, 256))
img = img.convert("RGB")
# Predict and display the image
predict(img, file_path, print_model=False)
