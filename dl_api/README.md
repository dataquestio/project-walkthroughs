# Project Overview

In this project, we'll build a translation API using deep learning.  Using FastAPI, we'll create a web server that exposes a `/translate` route and a `/results` route.  Clients will post their translation request to the `/translate` route, and get the translation results from `/results`.  The server will use a sqlite database to store the translations.  On the backend, we'll use async and a pretrained deep learning language model to run the translation job.

By the end, we'll have a web server that can run translation jobs quickly.  This server can easily be extended to translate more languages, or add more options.

**Project Steps**
* Build API routes
* Add in models to store data to database
* Create tasks to run the translation

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/dl_api)

File overview:

* `requirements.txt` - packages you'll need to install
* `languages.txt` - list of languages that are supported for translation
* `main.py` - defines the web server routes
* `models.py` - defines database models
* `tasks.py` - runs our backend tasks, including the translation

# Prerequisites

To complete this project, you'll need to have a good understanding of:

* Python syntax, including functions, if statements, and data structures
* Data cleaning
* Pandas syntax
* Using Jupyter notebook
* APIs
* The basics of machine learning.

Please make sure you've completed these Dataquest courses (or know the material) before trying this project:

* [Python Introduction](https://www.dataquest.io/course/introduction-to-python/)
* [For Loops and If Statements](https://www.dataquest.io/course/for-loops-and-conditional-statements-in-python/)
* [Dictionaries In Python](https://www.dataquest.io/course/dictionaries-frequency-tables-and-functions-in-python/)
* [Functions and Jupyter Notebook](https://www.dataquest.io/course/python-functions-and-jupyter-notebook/)
* [Python Intermediate](https://www.dataquest.io/course/python-for-data-science-intermediate/)
* [Pandas and NumPy Fundamentals](https://www.dataquest.io/course/pandas-fundamentals/)
* [Data Cleaning](https://www.dataquest.io/course/python-datacleaning/)
* [Machine Learning Fundamentals](https://www.dataquest.io/course/machine-learning-fundamentals/)
* [APIs and Web Scraping](https://www.dataquest.io/course/apis-and-scraping/)

# Local Setup

## Installation

To follow this project, please install the following locally:

* Python 3.8+
* The packages defined in `requirements.txt`
* It's recommended to install PyCharm or VSCode.

## Data

During this project, we'll download a pretrained language model.