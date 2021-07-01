# BoulderAreaDetector

Deploys a deep learning CNN classifying satellite imagery to [streamlit](https://share.streamlit.io/pszemraj/boulderareadetector) for user testing. The app MVP demo of GeoNatureDiscoverer (more info to be released.. TBD). The original idea for GeoNatureDiscoverer originated in the June 2021 CASSINI Hackathon, that repo is [here](https://github.com/JonathanLehner/cassini_2021_nature_discoverer).

An example of model predictions on a holdout set:

![Predicted Class-climb_area Examples](https://user-images.githubusercontent.com/74869040/124186053-0b1ba500-dabc-11eb-892d-5330deea51a5.png)

## Model Stats - ResNet101 CNN Classifier

In short, fastai w/ resnet101 trained on labeled dataset with two classes.

A decent writeup on how to create, train, and save a fastai computer vision model is in [this Medium article](https://medium.com/analytics-vidhya/understanding-fastai-v2-training-with-a-computer-vision-example-part-1-the-resnet-model-dd9270450bb8). BoulderAreaDetector uses a decently sized labeled dataset (approx 3000 satellite images, each 256x256 in two classes), but has not had any significant level of hyperparameter optimization yet beyond fast.ai basics.

### Model itself

- [ResNet101](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet101)
- package: fast.ai (pytorch)
- trained for 20 epochs
- Loss:  FlattenedLoss of CrossEntropyLoss()
- Optimizer: Adam

### Confusion Matrix & Metrics

![confusion matrix resnet101](https://user-images.githubusercontent.com/74869040/124186386-88dfb080-dabc-11eb-8699-91715f024458.png)
More details:
```
              precision    recall  f1-score   support

  climb_area       0.79      0.76      0.77       101
       other       0.97      0.97      0.97       800

    accuracy                           0.95       901
   macro avg       0.88      0.87      0.87       901
weighted avg       0.95      0.95      0.95       901
```


### Probability Distributions (on a holdout set)

![Probability Distributions (on a holdout set)](https://user-images.githubusercontent.com/74869040/124186513-b3ca0480-dabc-11eb-89dc-60cd15bce8af.png)

## Examples / Inference

Will add more details at a later time, but for now please see the website and/or [this short compilation of images](https://www.dropbox.com/s/x7cyu3r1u6ohtzx/holdout%20class%20prediction%20examples%20-%20resnet101%20model%2002%20dataset4.pdf?dl=1) in PDF format.

### Highest Loss Images (test set)

These were on the test set (not the holdout):

![images with the highest loss](https://user-images.githubusercontent.com/74869040/124186983-60a48180-dabd-11eb-8a6a-45a08034ffa3.png)
## Citations

ResNet
```
@misc{he2015deep,
      title={Deep Residual Learning for Image Recognition}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
