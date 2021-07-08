"""
Run the models on local machine. Useful for debugging

"""

import os
import pathlib
import pprint as pp
import shutil
from io import BytesIO
from os.path import basename, join

from fastai.vision.all import *
import timm
from natsort import natsorted



if platform.system() == "Windows":
    # model originally saved on Linux, strange things happen
    print("on Windows OS - adjusting PosixPath")
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


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

    return best_model


def load_mixnet_model():
    try:
        path_to_model = r"model-mixnetXL-20epoch.pkl"
        model = load_learner(path_to_model, cpu=True)
    except:
        print("unable to load locally. downloading model file")
        model_backup = "https://www.dropbox.com/s/bwfar78vds9ou1r/model-mixnetXL-20epoch.pkl?dl=1"
        model_response = requests.get(model_backup)
        model = load_learner(BytesIO(model_response.content), cpu=True)

    return model


def load_dir_files(directory, req_extension=".txt", return_type="list",
                   verbose=False):
    appr_files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(directory):
        for prefile in f:
            if prefile.endswith(req_extension):
                fullpath = os.path.join(r, prefile)
                appr_files.append(fullpath)

    appr_files = natsorted(appr_files)

    if verbose:
        print("A list of files in the {} directory are: \n".format(directory))
        if len(appr_files) < 10:
            pp.pprint(appr_files)
        else:
            pp.pprint(appr_files[:10])
            print("\n and more. There are a total of {} files".format(len(appr_files)))

    if return_type.lower() == "list":
        return appr_files
    else:
        if verbose: print("returning dictionary")

        appr_file_dict = {}
        for this_file in appr_files:
            appr_file_dict[basename(this_file)] = this_file

        return appr_file_dict


def predict(img, img_flex, model_pred, print_model=True, show_image=True, ):
    # Display the test image
    if show_image: img.show(title="Image to be predicted")
    # Load model and make prediction

    if not isinstance(img_flex, str):
        # convert image to fast AI PIL object
        fancy_class = PILImage(img_flex)
        model_pred.precompute = False
        pred_class, pred_items, pred_prob = model_pred.predict(fancy_class)
    else:
        # standard case
        # loads from a file so it's fine
        pred_class, pred_items, pred_prob = model_pred.predict(img_flex)
    if print_model: print(model_pred.model)

    prob_np = pred_prob.numpy()
    # Display the prediction
    if str(pred_class) == 'climb_area':
        print(
            "Area in test image is good for climbing! {}% confident.".format(round(100 * prob_np[0], 2)))
    else:
        print("Area in test image NOT great for climbing: {}% confident.".format(100 - round(100 * prob_np[0],
                                                                                             2)))


if __name__ == "__main__":
    # main code. can update the below to load images from a file in a for loop to batch classify
    # key parameters
    best_model_name = "model-resnetv2_50x1_bigtransfer.pkl"
    main_dir = os.getcwd()
    # Read the image
    working_dir = os.path.join(main_dir, "test_images")
    test_image_files = load_dir_files(working_dir, req_extension=".png", verbose=True)
    spacer = "\n\n"
    use_best_model = False  # takes a bit longer to load because it needs to be unzipped
    if use_best_model:
        model = load_best_model()
    else:
        model = load_mixnet_model()

    for image_path in test_image_files:
        img = load_image(image_path)
        img = img.resize((256, 256))
        img = img.convert("RGB")
        # Predict and display the image
        print(spacer)
        this_img = os.path.basename(image_path)
        print("Predicting images for test image {}".format(this_img))
        predict(img, image_path, model, print_model=False, show_image=False)


    if os.path.exists(join(main_dir, best_model_name)):
        os.remove(join(main_dir, best_model_name))
        print("Removed the unpacked .pkl model file {} from {}".format(best_model_name, main_dir))
