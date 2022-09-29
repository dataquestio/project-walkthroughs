# Project Overview

In this project, we'll figure out what emotion a dog is feeling based on a picture.  We'll use deep learning to analyze the image and predict the emotion.  We'll start out by downloading the images from Flickr.  Then we'll clean the images using PIL.  We'll remove any images that don't have a dog in them using timm and a pretrained deep learning model.  Then we'll then train a neural network to make predictions about what emotion a dog is feeling.  We'll end by evaluating how to improve the model.

We'll use Pytorch and torchvision to train the model.

At the end, you'll have a trained deep learning model that can predict dog emotions.

**Project Steps**

* Download images from Flickr
* Clean images using PIL
* Remove any images that don't have a dog in them
* Train a deep learning model to predict dog emotions
* Evaluate the model

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/dog_emotions)

File overview:

* `download_images.ipynb` - code to download the images from Flickr.
* `clean_images.ipynb` - clean the images using PIL.
* `remove_images.ipynb` - remove any images that don't include a dog.
* `prediction.ipynb` - train a neural network to predict if a dog is in an image.
* `settings.py` - has constants that are needed in other files.

# Prerequisites

To complete this project, you'll need to have a good understanding of:

* Python syntax, including functions, if statements, and data structures
* Data cleaning
* Pandas syntax
* Using Jupyter notebook

You'll also need to know the basics of the command line and machine learning.

Please make sure you've completed these Dataquest courses (or know the material) before trying this project:

* [Python Introduction](https://www.dataquest.io/course/introduction-to-python/)
* [For Loops and If Statements](https://www.dataquest.io/course/for-loops-and-conditional-statements-in-python/)
* [Dictionaries In Python](https://www.dataquest.io/course/dictionaries-frequency-tables-and-functions-in-python/)
* [Functions and Jupyter Notebook](https://www.dataquest.io/course/python-functions-and-jupyter-notebook/)
* [Python Intermediate](https://www.dataquest.io/course/python-for-data-science-intermediate/)
* [Pandas and NumPy Fundamentals](https://www.dataquest.io/course/pandas-fundamentals/)
* [Data Cleaning](https://www.dataquest.io/course/python-datacleaning/)
* [Command Line](https://www.dataquest.io/course/command-line-elements/)
* [Machine Learning Fundamentals](https://www.dataquest.io/course/machine-learning-fundamentals/)

# Local Setup

## Installation

To follow this project, please install the following locally:

* JupyerLab
* Python 3.8+
* Python packages
    * pandas
    * numpy
    * timm
    * torch
    * torchvision
    * Pillow
    * flickrapi

## Data

We'll download the image data using Flickr.  

* If you don't want to get an API key and download the data, you can find the images [here](https://drive.google.com/uc?export=download&id=1RZV1Rnix8aIfSVzikBiW09wn9bg20jNDg).
* If you don't want to clean the images, and want to jump into deep learning, you can download cleaned images [here](https://drive.google.com/uc?export=download&id=1VN4TnRaBtajJ7m4FiGCOQCEgbs2bquem).
* The model with pretrained weights is [here](https://drive.google.com/uc?export=download&id=1p0xE16XMN1JdR653PELaf9SvEXlzQXXn)