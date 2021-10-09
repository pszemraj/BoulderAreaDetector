# BoulderAreaDetector

Deploys a deep learning CNN classifying satellite imagery to [streamlit](https://share.streamlit.io/pszemraj/boulderareadetector) for user testing.

-   The app is an MVP demo of [BoulderSpot](https://boulderspot.io/). The original idea for BoulderSpot originated in the June 2021 CASSINI Hackathon, that repo is [here](https://github.com/JonathanLehner/cassini_2021_nature_discoverer).
-   [BoulderSpot](https://boulderspot.io/) uses a similar model to the one included here to classify whether aerial images are potential boulder areas or not. The class results are then used as part of a graph-like framework to analyze aerial imagery all across Switzerland. You can find more details on the website!

An example of model predictions on a holdout set:

![Predicted Class-climb_area Examples](https://user-images.githubusercontent.com/74869040/124186053-0b1ba500-dabc-11eb-892d-5330deea51a5.png)

A picture of some of the boulders the (full) model found after an in-person data validation trip:

![Boulderspot-trip-03-Valhalla-06-min](https://user-images.githubusercontent.com/74869040/136666878-bfa590a2-9463-44c0-94b9-210d637ea22f.png)

## Model Stats - CNN Classifier

In short, the predictor under-the-hood is: fastai library using a convolutional neural network
trained on a labeled dataset of several thousand images with two classes (climb_area, other).
Source image data for training is mostly arial (possibly some satellite) sampled from Switzerland.

**Note: the model deployed in the streamlit app has changed.** the original model used in this app
was [ResNet101](https://arxiv.org/abs/1512.03385) and the trained model file is ~170 MB. As GitHub
has limits / special rules around files greater than 100 mb in size, the model has been updated
to [MixNet-XL](https://paperswithcode.com/method/mixconv), which exhibits similar performance but is
smaller (in parameters, and therefore file size).

> Also included in the repo is a zipped model file of a trained [Big Transfer model](https://paperswithcode.com/lib/timm/big-transfer) that is more accurate than either of the
> two. As this model is > 100 mb and streamlit unzipping+predicting performance is yet to be
> tested, it is not deployed to the app yet, but can be used locally.

A decent writeup on how to create, train, and save a fastai computer vision model is
in [this Medium article](https://medium.com/analytics-vidhya/understanding-fastai-v2-training-with-a-computer-vision-example-part-1-the-resnet-model-dd9270450bb8). BoulderAreaDetector uses a decently
sized labeled dataset (several thousand satellite images, each 256x256 in the two classes), but has
not had any significant level of hyperparameter optimization yet beyond fast. ai basics.

### MixNet: Model itself

-   [MixNet](https://github.com/rwightman/pytorch-image-models/blob/54a6cca27a9a3e092a07457f5d56709da56e3cf5/timm/models/efficientnet.py)
    `*Note: the above links to timm source code as the MixNet paper is already linked above*`
-   package: fast.ai (pytorch)
-   trained for 20 epochs
-   Loss:  FlattenedLoss of CrossEntropyLoss()
-   Optimizer: Adam
-   Total params: 11,940,824

### MixNet:Confusion Matrix & Metrics

![MixNet-XL Confusion Matrix](https://www.dropbox.com/s/yscr06wn03ikouo/mixnet_xl%20%20-%20CK%2BA%20-%2020epconfusion%20matrix.png?dl=1)

                  precision    recall  f1-score   support

      climb_area       0.84      0.57      0.68       206
           other       0.98      0.99      0.99      3854

        accuracy                           0.97      4060
       macro avg       0.91      0.78      0.83      4060
    weighted avg       0.97      0.97      0.97      4060

**More details can be found in `/info`**

### Probability Distributions (on a holdout set)

`#TODO`

## Examples / Inference

`#TODO`

### Highest Loss Images (test set)

The following images had the highest loss when evaluated as part of the test (not holdout) set
during training:

![highest loss MixNet imgs](https://www.dropbox.com/s/7nlo210srtq9xwg/mixnet_xl%20%20-%20CK%2BA%20-%2020ephighest_loss_images.png?dl=1)

* * *

### Details on Original ResNet101 Fine-Tuned Model

This was the original model that was replaced as the file size was too large.

-   [ResNet101](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet101)
-   package: fast.ai (pytorch)
-   trained for 20 epochs
-   Loss:  FlattenedLoss of CrossEntropyLoss()
-   Optimizer: Adam

![confusion matrix resnet101](https://user-images.githubusercontent.com/74869040/124186386-88dfb080-dabc-11eb-8699-91715f024458.png)
More details:

                  precision    recall  f1-score   support

      climb_area       0.79      0.76      0.77       101
           other       0.97      0.97      0.97       800

        accuracy                           0.95       901
       macro avg       0.88      0.87      0.87       901
    weighted avg       0.95      0.95      0.95       901

* * *

# Citations

MixNet

```bazaar
@misc{tan2019mixconv,
      title={MixConv: Mixed Depthwise Convolutional Kernels},
      author={Mingxing Tan and Quoc V. Le},
      year={2019},
      eprint={1907.09595},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Big Transfer

```bazaar
@misc{kolesnikov2020big,
      title={Big Transfer (BiT): General Visual Representation Learning},
      author={Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Joan Puigcerver and
      Jessica Yung and Sylvain Gelly and Neil Houlsby},
      year={2020},
      eprint={1912.11370},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

ResNet

```bazaar
@misc{he2015deep,
      title={Deep Residual Learning for Image Recognition},
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
