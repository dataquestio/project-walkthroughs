# Project Overview

In this project, we'll build a k-means clustering algorithm from scratch.  Clustering is an unsupervised machine learning technique that can find patterns in your data.  K-means is one of the most popular forms of clustering.

We'll create our algorithm using python and pandas.  We'll then compare it to the reference implementation from scikit-learn.

**Project Steps**

* Write out pseudocode for the algorithm
* Code the k-means algorithm
* Plot the clusters from the algorithm
* Compare performance to the scikit-learn algorithm

# K-means overview

K-means is an unsupervised machine learning technique that allow us to cluster data points.  This enables us to find patterns in the data that can help us analyze it more effectively.  K-means is an iterative algorithm, which means that it will converge to the optimal clustering over time.

To run a k-means clustering:

1. Specify the number of clusters you want (usually referred to as `k`).
2. Randomly initialize the `centroid` for each cluster.  The centroid is the data point that is in the center of the cluster.  
3. Determine which data points belong to which cluster by finding the closest `centroid` to each data point.
4. Update the centroids based on the geometric mean of all the data points in the cluster.
5. Run 3 and 4 until the `centroids` stop changing.  Each run is referred to as an iteration.

![k-means convergence](images/K-means_convergence.gif)

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/kmeans).

File overview:

* `clustering.ipynb` - the full code from this project

# Local Setup

## Installation

To follow this project, please install the following locally:

* Python 3.8+
* Python packages
    * pandas
    * numpy
    * scikit-learn


## Data

We'll be using data from FIFA, which you can download [here](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset).  We'll use the file `players_22.csv`.