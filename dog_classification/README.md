# Project Overview

In this project, we'll walk through an end to end deep learning project using Tensorflow and Keras.  We'll read in a dataset of dog images, then train a convolutional neural network to classify them by breed.

By the end, you'll know how to use keras to train and optimize a neural network.  You'll also learn about how to work with images using Python.

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/dog_classification).

File overview:

* `classifier.ipynb` - a Jupyter notebook that loads the images and trains a neural network.

# Local Setup

## Installation

To follow this project, please install the following locally:

* JupyerLab
* Python 3.8+
* Python packages
    * tensorflow
    * Pillow
    * pandas
    * matplotlib
    
You will also need to have a GPU on your machine and configured.  To set things up, you'll need to install [GPU support for tensorflow](https://www.tensorflow.org/install/gpu).
 
If you have issues installing tensorflow and/or don't have a GPU, please use [Google Colaboratory](https://colab.research.google.com/).  Colaboratory will give you a Jupyter notebook in the cloud with full GPU support.

## Data

You'll need to download the dog image dataset to follow this project:

* [dog_images.zip](https://drive.google.com/uc?export=download&id=1sj62C-9WKD09-8iYSeEvXmAGQoY2oFFQ) - please unzip this file into a folder called `images`.

The data is originally from [Stanford](http://vision.stanford.edu/aditya86/ImageNetDogs/).  The original dataset has many more breeds included, which you can use to extend your analysis.

