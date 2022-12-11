
## Filtered Batch Normalization
It is a common assumption that the activation of different layers in neural networks follow Gaussian distribution. This distribution can be transformed using normalization techniques, such as batch-normalization, increasing convergence speed and improving accuracy. In this paper we would like to demonstrate, that activations do not necessarily follow Gaussian distribution in all layers. Neurons in deeper layers are more selective and specific which can result extremely large, out-ofdistribution activations.

This repository contains our imlpementation and experiments with filtered batch normalization

## Paper
Our paper published at ICPR:
https://arxiv.org/pdf/2010.08251.pdf

## Codes

Our code is structured as the following:


Folder activations contain scripts to display the ditrbiution of the activations in certain layers of neuron networks.
These codes were used to generate the data for Section 2.


batchnormlib.py - contains our implementation of filtered batchnormalization which can be used as a possible replacement of batchnorm layers in pytorch.

### Activations

activations/ImagenetLoaderValidation.py
Is a data loader to load the validation set of ImageNet

activations/plot_activation_distributions_resnet.py
is a script which generates distribution plots about layer activations for the convolutional layers of Resnet and ResNeXt architectures.


activations/plot_activation_distributionvgg.py
is a script which generates distribution plots about layer activations for the convolutional layers of VGG


activations/plot_outlier_images.py
This plot displays images which invokes a larger than threshold activations in a selected layer and kernel in the VGG architecture.

### FitleredBatchNormLib
FitleredBatchNormLib/FilteredBatchNorm.py
This is a small library in which we have collected our implmentations of filtered batch normalization both for one dimensional (after fully connected) and two dimensional (after 2d convolutional layers) data. We have not investigated teh applicaiton of our algorithm in case of higher dimensional data 3d or 4d, but it can be imeplmented in a strightforward manner.
This way our method can be investigated and evaluated with any pytroch code in a plug and play manner.

### MNIST

This folder contains scripts which were used to genrate the results in section 4.1

MNIST/mnist_train.py
This script contains the training script and comparison for batch norm and filtered batch norm on the mnist dataset with a simple architecture. Two convolutional and two fully connected layers.

MNIST/mnist_lenet5_test.py
This script contains the training script and comparison for batch norm and filtered batch norm on the mnist dataset with the LeNet5 architecture on the MNISt dataset.

MNIST/mnist_batchsize.py
This script investigates the effect of filtered batch normalization with differetn Tsigma parameters and batch sizes.

MNIST/mnist_loss_test.py
This script generates loss and gradient landscapes on the MNISt dataset.

### CIFAR


This folder contains scripts which were used to genrate the results in section 4.2

CIFAR/alexnet_batchnorm_double.py
This sciprt contains the training of the alexnet archtiecture on the CIFAR-10 dataset with filtered batch normalization.

CIFAR/alexnet_small_large_compare.py
This script compares large and small batch sizes and how the mean and variance of the large batch size can be approximated by the small one.

### Detectron2


/home/horan/Data/robust_batchnorm/supmat/codes/detectron2/batch_norm.py
This folder contained the modified batch_norm layer of detectron 2. Originally in the deteton2/layers folder in the repo:
https://github.com/facebookresearch/detectron2
This normalizaiton was used for instance segemntation in section 4.5

### ImageNet

This folder contains scripts which were used in section 4.3 and 4.4

ImageNet/ImagenetLoader.py
ImageNet/ImagenetLoaderValidation.py
these scripts contain dataloader for the train and test set of the ImageNet datasets.

ImageNet/vgg_train.py
This script was used for training differetn architectures on ImageNet with different normalization methods

ImageNet/vgg_validation.py
This sciprt was used to evaluate a trained arhcitecture on the validation set of ImageNet

ImageNet/vgg_loss_landscape.py
This script was used to generate the loss and gradient landscape ot IamgeNet training.
