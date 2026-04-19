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

## Acknowledgement
This repository is built upon [MSDNET](https://github.com/gaohuang/MSDNet) and [RepDistiller](https://github.com/HobbitLong/RepDistiller)
