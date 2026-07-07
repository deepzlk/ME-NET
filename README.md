# ME-Net: A Unified Experimental Benchmark for Multi-Exit Network Design

## Overview

ME-Net is a unified experimental framework for systematically studying the design factors of Multi-Exit Networks (MENs). Multi-exit networks improve inference efficiency by introducing intermediate classifiers at different depths of a neural network, allowing predictions to be generated before reaching the final layer.

Existing studies often employ different network architectures, classifier structures, training settings, and evaluation protocols, making fair comparison difficult. This repository provides a reproducible benchmark for analyzing the influence of key design factors under controlled experimental conditions.

The framework supports multiple backbone networks, configurable intermediate classifiers, flexible exit placement strategies, varying numbers of exits, and different training methods.

The implementation is based on PyTorch and has been evaluated on CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets.

This repository provides the complete implementation used in our paper, including backbone training, multi-exit network construction, intermediate classifier design, training, evaluation, and benchmarking.

It also contains the code for reproducing all experimental results reported in the paper.

---

# Project Structure

```text
ME-Net/
│
├── data/
├── datasets/
├── helper/
├── models/
├── train_menm.py
├── train_menm8.py
├── test_iconly.py
├── test_menm.py
│
└── README.md
```

---

# Research Objectives

This project investigates the following key design factors in Multi-Exit Networks:

1. Backbone Network Architecture
2. Intermediate Classifier Structure
3. Intermediate Classifier Placement
4. Number of Intermediate Classifiers
5. Training Strategy

All factors can be independently configured and evaluated.

---

# Supported Datasets

## CIFAR-10

CIFAR-10 contains 60,000 color images of size 32×32 distributed among 10 classes.

- Training Images: 50,000
- Test Images: 10,000

Reference:

Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*.

Website:

https://www.cs.toronto.edu/~kriz/cifar.html

https://github.com/deepzlk/ME-NET/releases/tag/Cifar

---

## CIFAR-100

CIFAR-100 is similar to CIFAR-10 but contains 100 object categories.

- Training Images: 50,000
- Test Images: 10,000
- Classes: 100

Reference:

Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*.

Website:

https://www.cs.toronto.edu/~kriz/cifar.html

https://github.com/deepzlk/ME-NET/releases/tag/Cifar

---

## Tiny-ImageNet

Tiny-ImageNet is a subset of ImageNet containing 200 classes.

- Training Images per Class: 500
- Validation Images per Class: 50
- Test Images per Class: 50
- Image Size: 64×64

Dataset Source:

https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet

---

# Requirements

## Hardware

- NVIDIA GPU (recommended)
- CUDA-compatible environment

## Software

The original experiments were conducted using:

- Ubuntu 16.04.5 LTS
- Python 3.7
- PyTorch 1.4.0
- CUDA 11.0

The code should also work with newer PyTorch versions (≥ 1.4.0).

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your_repository/ME-Net.git
cd ME-Net
```

Install dependencies:

```bash
pip install torch torchvision numpy scipy tqdm
```

Verify PyTorch installation:

```bash
python -c "import torch; print(torch.__version__)"
```

---

# Code Information

The repository contains the following main scripts and modules for training and evaluating Multi-Exit Networks.

## Training Scripts

### Backbone Network Training

Train a baseline backbone network without intermediate exits.

```bash
python train_backbone.py
```

Supported backbone networks include:

- ResNet18
- MobileNetV1
- Other configurable architectures

---

### Multi-Exit Network Training

Train a standard four-exit Multi-Exit Network.

```bash
python train_menm.py
```

---

### Multi-Exit Network Training (Eight Exits)

Train a Multi-Exit Network with eight intermediate classifiers.

```bash
python train_menm8.py
```

This script is mainly used to investigate the influence of the number of exits.

---

## Evaluation

Evaluate the trained model.

```bash
python test_menm.py
```

The testing script reports:

- Classification accuracy of each exit
- Final classifier accuracy
- Overall model performance

---

## Intermediate Classifier Configuration

The intermediate classifier (IC) configurations are located in

```text
models/util/
```

Researchers can customize:

- Intermediate classifier structures
- Exit positions
- Classifier depth
- Feature aggregation strategies
- Pooling operations
- Fully connected layers

This flexible design enables systematic comparison of different Multi-Exit Network architectures under a unified experimental framework.

---

# Usage Instructions

The implementation is designed to provide a simple and reproducible experimental workflow. Users can reproduce the reported results by following the steps below.

## Step 1. Clone the Repository

```bash
git clone https://github.com/deepzlk/ME-NET.git
cd ME-NET
```

---

## Step 2. Install Dependencies

```bash
pip install torch torchvision numpy scipy tqdm
```

---

## Step 3. Prepare the Datasets

Download the required datasets from the links provided in the **Dataset Information** section and place them in the `datasets/` directory.

```text
ME-Net/
│
├── datasets/
│   ├── cifar10/
│   ├── cifar100/
│   └── tiny-imagenet/
```

---

## Step 4. Train the Networks

Run the training script to sequentially train the backbone network and the corresponding Multi-Exit Network.

```bash
bash train.sh
```

---

## Step 5. Evaluate the Model

```bash
python test_menm.py
```

The testing script reports the classification accuracy of all exits and the final classifier.

---

## Step 6. Customize the Network

Modify the intermediate classifier configuration in

```text
models/util/
```

to investigate different backbone architectures, exit configurations, and intermediate classifier designs. After modifying the configuration, retrain the network before evaluation.

# Reproducibility

To ensure fair comparison:

- The same datasets are used across experiments.
- Training hyperparameters remain consistent.
- Only one design factor is modified at a time.
- Multiple backbone networks are evaluated under identical settings.

This allows the isolated study of each design choice.

---

# Citation

If you use this repository in your research, please cite:

```bibtex
@article{ME-Net2025,
  title={A Unified Experimental Benchmark for Multi-Exit Network Design},
  author={Anonymous},
  journal={Under Review},
  year={2025}
}
```

If the paper has been accepted, please replace the above entry with the final publication information.

---

# License

This project is released for academic and research purposes.

You may modify and redistribute the code with proper attribution.

For commercial use, please contact the authors.

---

# Acknowledgements

This repository is built upon the following excellent open-source projects:

### MSDNet

https://github.com/gaohuang/MSDNet

Huang et al., *Multi-Scale Dense Networks for Resource Efficient Image Classification*, ICLR 2018.

### RepDistiller

https://github.com/HobbitLong/RepDistiller

Tian et al., *Contrastive Representation Distillation*, ICLR 2020.

We sincerely thank the authors for making their code publicly available.


