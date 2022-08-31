# Project Overview

In this project, we'll predict future house prices.  We'll use data from the Federal Reserve, along with house price data from Zillow.  We'll merge and combine this data, then use it to train a random forest model.  The model will predict if house prices will increase or decrease in the future.  We'll measure error using backtesting, then improve our model with new predictors.

This project can be customized to predict house prices in your metro area if you live in the US.

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

You'll need to download a few csv files to run this project.  These files are included in this repo.  You can also download them all [here](https://drive.google.com/uc?export=download&id=1HlHw_JyRckfPOlwwxUHS-sdDqfZQ732p).

If you want to get newer versions:

* Federal reserve data
    * [CPI dataset](https://fred.stlouisfed.org/series/CPIAUCSL) - CPIAUCSL.csv
    * [Rental vacancy rate](https://fred.stlouisfed.org/series/RRVRUSQ156N) - RRVRUSQ156N.csv
    * [Mortgage interest rates](https://fred.stlouisfed.org/series/MORTGAGE30US) - MORTGAGE30US.csv
* [Zillow data](https://www.zillow.com/research/data/)
    * ZHVI (raw, weekly) - Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv
    * Median sale price (raw, all homes, weekly) - Metro_median_sale_price_uc_sfrcondo_week.csv