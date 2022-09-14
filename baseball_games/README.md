# Project Overview

In this project, we'll predict future season stats for baseball players.  Specifically, we'll predict how many home runs a player will hit next season.  We'll first download baseball season data using pybaseball and clean it.  We'll do feature selection using a sequential feature selector to identify the most promising predictors for machine learning.  We'll then train an xgboost model to predict future season HRs.  We'll measure error and improve the model.

At the end, you'll have a model that can predict future season HRs, along with next steps to improve the model.

**Project Steps**

* Download baseball season data
* Clean the data and prepare it for ML
* Run feature selection
* Create an xgboost model and estimate accuracy
* Improve accuracy

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/baseball_games)

File overview:

* `next_hr.ipynb` - a Jupyter notebook that contains the code to predict next season WAR.

# Local Setup

## Installation

To follow this project, please install the following locally:

* JupyerLab
* Python 3.8+
* Python packages
    * pybaseball
    * pandas
    * xgboost
    * scikit-learn
    * numpy

## Data

We'll download the data during this project, using the `pybaseball` package.