# Project Overview

In this tutorial, we'll cover the full process of building a beginner machine learning project. This includes creating a hypothesis, setting up the model, and measuring error. By the end, you'll understand how to build an end-to-end machine learning project using Python and Jupyter.

To make this interesting, we'll use a fun dataset. We'll use data from historical Olympic games. We'll try to predict how many medals a country will win based on historical and current data.


# Machine learning project steps

Most machine learning projects follow a similar outline, which we'll also follow here.  This outline will help you tackle any machine learning problem.

**Project Steps**

1. Form a hypothesis.
2. Find and explore the data.
3. (If necessary) Reshape the data to predict your target.
4. Clean the data for ML.
5. Pick an error metric.
6. Split your data.
7. Train a model.

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/beginner_ml).

File overview:

* `machine_learning.ipynb` - the main project code
* `data_prep.ipynb` - the code to generate the team-level dataset from an athlete-level dataset

# Local Setup

## Installation

To follow this project, please install the following locally:

* Python 3.8+
* Python packages
    * pandas
    * numpy
    * scikit-learn
    * seaborn


## Data

We'll be using data from the Olympics, which was originally on [Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results).

You can download the files we'll use in this project here:

* [teams.csv](https://drive.google.com/uc?export=download&id=1L3YAlts8tijccIndVPB-mOsRpEpVawk7) - the team-level data that we use in this project.
* [athlete_events.csv](https://drive.google.com/uc?export=download&id=1Ah4wOyNFMGREq8Yw_Jbv7u2CeI_6tpn5) - this is the original athlete-level data