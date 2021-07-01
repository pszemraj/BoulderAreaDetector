"""
tests out how things work on the local machine

"""

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
import torch
from fastai.vision.all import *
from natsort import natsorted
from os.path import join
from skimage.transform import resize
import pathlib
import platform

if platform.system() == "Windows":
    # model originally saved on Linux, strange things happen
    print("on winmdows - adjusting PosixPath")
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

def predict(img, img_path):
    # Display the test image
    skimage.io.imshow(img)


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
        print("Submitted img is a climbing area, confidence level is {}".format(round(100*prob_np[0]),2))
    else:
        print("Area in submitted image not great for climbing. confidence level is {}".format(round(100*prob_np[0]),2))

test_image = "test_img_boulder.png"
# Read the image
working_dir = os.path.join(os.getcwd(), "test_images")
file_path = os.path.join(working_dir, test_image)
img = skimage.io.imread(file_path)
img = resize(img, (256, 256))

# Predict and display the image
predict(img, file_path)
