# Project Overview

In this project, we'll predict future house prices.  We'll use data from the Federal Reserve, along with house price data from Zillow.  We'll merge and combine this data, then use it to train a Random Forest model.  The model will predict if house prices will increase or decrease in the future.  We'll measure error and improve our model with new predictors.

**Project Steps**

* Load in data
* Clean and merge data
* Create an initial machine learning model and estimate accuracy
* Improve the accuracy of the model
* Run diagnostics to figure out how we can improve

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/house_prices)

File overview:

* `prices.ipynb` - a Jupyter notebook that contains all of the code.

# Local Setup

## Installation

To follow this project, please install the following locally:

* JupyerLab
* Python 3.8+
* Python packages
    * pandas
    * yfinance
    * scikit-learn

## Data

You'll need to download a few csv files to run this project:

* Federal reserve data
    * [CPI dataset](https://fred.stlouisfed.org/series/CPIAUCSL)
    * [Rental vacancy rate](https://fred.stlouisfed.org/series/RRVRUSQ156N)
    * [Mortgage interest rates](https://fred.stlouisfed.org/series/MORTGAGE30US)
* [Zillow data](https://www.zillow.com/research/data/)
    * ZHVI (raw, weekly)
    * 