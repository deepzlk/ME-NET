# ME-Net
## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.7, PyTorch 1.4.0, and CUDA 11.0. But it should be runnable with recent PyTorch versions >=1.4.0

## Experimental Setup
Our experiments are designed to systematically investigate five key factors in multi-exit networks:

Backbone network \
Structure of intermediate classifiers \
Position of intermediate classifiers \
Number of intermediate classifiers \
Training strategy 

Each factor can be configured and evaluated independently as described below. 

## Train

**Backbone Networks**

````bash
python train_backbone.py 
````

**Multi-Exit Network (MENET)**

````bash
python train_menm.py 
````

**Intermediate Classifier Structure & Position**

The structure and insertion positions of intermediate classifiers are configurable.
````bash
Location:models/util 
````
**Training Strategy of Intermediate Classifiers**

The number of exits (intermediate classifiers) is treated as a key variable.

````bash
python train_menm8.py
````
## Test
````bash
python test_menm.py
````
## Dataset

**CIFAR-10**

The CIFAR-10 dataset consists of 60,000 32×32 color images in 10 classes, with 50,000 training images and 10,000 test images.

Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. University of Toronto.
Available at: https://www.cs.toronto.edu/~kriz/cifar.html

**CIFAR-100**

The CIFAR-100 dataset is similar to CIFAR-10 but contains 100 classes, with 600 images per class (500 training and 100 testing images).

Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. University of Toronto.
Available at: https://www.cs.toronto.edu/~kriz/cifar.html

**Tiny-ImageNet**

Tiny-ImageNet is a subset of the ImageNet dataset, containing 200 classes with 500 training images, 50 validation images, and 50 test images per class. All images are resized to 64×64.

https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet

## Acknowledgement
This repository is built upon [MSDNET](https://github.com/gaohuang/MSDNet) and [RepDistiller](https://github.com/HobbitLong/RepDistiller)
