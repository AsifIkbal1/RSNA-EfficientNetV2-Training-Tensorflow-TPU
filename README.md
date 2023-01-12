# RSNA-EfficientNetV2-Training-Tensorflow-TPU
RSNA EfficientNetV2 Training Tensorflow TPU

Md Asif Ikbal
Hello fellow Kagglers,

This notebook demonstrates the training process on a TPU in Tensorflow.

Thanks to the use of a [TPU (Tensor Processing Unit)](https://cloud.google.com/tpu) training takes about an hour.

The TFREcord dataset contains cropped images sized 1344x768, created in [this notebook](https://www.kaggle.com/code/markwijkhuizen/rsna-preprocessing-tfrecords-640x512-dataset).

20% of the data is used for validation, which reaches ~0.20 pF1 with the best threshold.

**Things that did not work for me:**

* [SigmoidFocalCrossEntropy](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy)
* Increasing model size to for example EfficientNetV2S

**Things that did work for me:**

* Class weights: give minority class weight of 10
* Training on TPU instead of GPU: larger batch size (16x2->16x8) giving larger probability of having positive sample in batch
* Cropping Images
* Using Cropped Image Ratio

I enjoy this competition and will update this notebook frequently, stay tuned!

**V2**

* Cropped images in 1344x768 resolution
* EfficientNetV2T
* Added augmentations
* Single image modal instead of both CC and MLO views as input

[Inference Notebook](https://www.kaggle.com/markwijkhuizen/rsna-efficientnetv2-inference-tensorflow)
