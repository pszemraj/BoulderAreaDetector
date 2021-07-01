# BoulderAreaDetector

Deploys a deep learning CNN classifying satellite imagery to [streamlit](https://share.streamlit.io/pszemraj/boulderareadetector) for user testing.
an MVP demo of GeoNatureDiscoverer


## Model Stats - ResNet101 CNN Classifier

It uses a decently sized labeled dataset, but has not had any significant level of hyperparameter optimization yet beyond fast.ai basics.
```
              precision    recall  f1-score   support

  climb_area       0.79      0.76      0.77       101
       other       0.97      0.97      0.97       800

    accuracy                           0.95       901
   macro avg       0.88      0.87      0.87       901
weighted avg       0.95      0.95      0.95       901
```

## Examples / Inference

Will add more details at a later time, but for now please see the website and/or [this short compilation of images](https://www.dropbox.com/s/x7cyu3r1u6ohtzx/holdout%20class%20prediction%20examples%20-%20resnet101%20model%2002%20dataset4.pdf?dl=1) in PDF format.
