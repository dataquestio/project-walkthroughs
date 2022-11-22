# Project Overview

In this project, we'll predict the weather using the Facebook prophet algorithm.  Prophet uses an additive model to add up seasonal effects and trends to make a prediction.  The advantage of prophet is that it automatically identifies seasonality in the data - and weather data has strong seasonal effects.  So without any feature engineering, you can get good baseline accuracy.  It can also scale to multiple time series (think data from adjacent weather stations) easily.

By the end, we'll have a model that predicts the weather, and can be extended to improve accuracy.

**Project Steps**
* Load in and clean data
* Define targets and predictors
* Train model
* Scale model to entire dataset using cv
* Make future predictions

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/prophet_weather)

File overview:

* `predict.ipynb` - notebook to make predictions

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
* It's recommended to use JupyterLab.

## Data

You can find the data for this project in the `weather.csv` file.  You can download the file [here](https://raw.githubusercontent.com/dataquestio/project-walkthroughs/master/prophet_weather/weather.csv).  Just click file -> save as in your browser to save the file.

If you want to download different data for your area, you can follow these instructions:

1. Go to [NOAA](https://www.ncdc.noaa.gov/cdo-web/search)
2. Enter the years you want data for (I recommend starting with 1970), and search for the closest airport to you
    * ![download_1](imgs/download_1.png)
3. Click add to cart on the airport you want
    * If there is no airport near you, try your city or country name instead
    * ![download_2](imgs/download_2.png)
4. Search for additional airports or cities and select them if you want additional data sources 
5. Go to the cart at https://www.ncdc.noaa.gov/cdo-web/cart
6. Select the csv format and click continue
    * ![download_3](imgs/download_3.png)
7. Select all of the checkboxes for data types
    * ![download_4](imgs/download_4.png)
8. Enter your email and click continue
    * ![download_5](imgs/download_5.png)
9. You'll get an email with a link to download the data
    * ![download_6](imgs/download_6.png)
10. Make sure to take a look at the [data documentation](https://www1.ncdc.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf) as well