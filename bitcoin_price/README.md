# Project Overview

In this project, we'll predict the future price of Bitcoin.  We'll use historical data on the price of Bitcoin, along with data from Wikipedia about edits to the Bitcoin page.  We'll merge and combine this data, then use it to train a random forest model that will tell us if Bitcoin prices will increase or decrease tomorrow.  We'll then switch to an XGBoost model and better predictors to improve accuracy.

We'll develop a backtesting system and use a robust error metric so we can tell if the algorithm is performing well.

This project can be extending to other cryptocurrencies as well.

**Project Steps**

* Load in data
* Clean and merge data
* Create an initial machine learning model and estimate accuracy
* Switch to a more powerful model and improve our predictors

## Code

You can find the code for this project [here](https://github.com/dataquestio/project-walkthroughs/tree/master/bitcoin_price)

File overview:

* `prediction.ipynb` - a Jupyter notebook that contains the code to predict Bitcoin prices
* `sentiment.ipynb` - a Jupyter notebook that creates our wikipedia edit dataset

# Local Setup

## Installation

To follow this project, please install the following locally:

* JupyerLab
* Python 3.8+
* Python packages
    * pandas
    * yfinance
    * scikit-learn
    * xgboost
    * mwclient

## Data

Computing the Wikipedia edit data takes time.  It can be faster to run the version that's already been generated.  It's in this repository, and called `wikipedia_edits.csv`.  We'll be downloading the bitcoin price data as part of the proh

If you want to get newer versions:

* Federal reserve data
    * [CPI dataset](https://fred.stlouisfed.org/series/CPIAUCSL) - CPIAUCSL.csv
    * [Rental vacancy rate](https://fred.stlouisfed.org/series/RRVRUSQ156N) - RRVRUSQ156N.csv
    * [Mortgage interest rates](https://fred.stlouisfed.org/series/MORTGAGE30US) - MORTGAGE30US.csv
* [Zillow data](https://www.zillow.com/research/data/)
    * ZHVI (raw, weekly) - Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv
    * Median sale price (raw, all homes, weekly) - Metro_median_sale_price_uc_sfrcondo_week.csv