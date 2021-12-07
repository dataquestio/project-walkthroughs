{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0f145a2c-78c1-4e51-8943-4438df9c6154",
   "metadata": {},
   "source": [
    "# Stock Price Prediction\n",
    "\n",
    "Let's say we want to make money by buying stocks.  Since we want to make money, we only want to buy stock on days when the price will go up (we're against shorting the stock).  We'll create a machine learning algorithm to predict if the stock price will increase tomorrow.  If the algorithm says that the price will increase, we'll buy stock.  If the algorithm says that the price will go down, we won't do anything.\n",
    "\n",
    "We want to maximize our `true positives` - days when the algorithm predicts that the price will go up, and it actually goes go up.  Therefore, we'll be using precision as our error metric for our algorithm, which is `true positives / (false positives + true positives)`.  This will ensure that we minimize how much money we lose with `false positives` (days when we buy the stock, but the price actually goes down).\n",
    "\n",
    "This means that we will have to accept a lot of `false negatives` - days when we predict that the price will go down, but it actually goes up.  This is okay, since we'd rather minimize our potential losses than maximize our potential gains.\n",
    "\n",
    "## Method\n",
    "\n",
    "Before we get to the machine learning, we need to do a lot of work to acquire and clean up the data.  Here are the steps we'll follow:\n",
    "\n",
    "* Download historical stock prices from Yahoo finance\n",
    "* Explore the data\n",
    "* Setup the dataset to predict future prices using historical prices\n",
    "* Test a machine learning model\n",
    "* Setup a backtesting engine\n",
    "* Improve the accuracy of the model\n",
    "\n",
    "At the end, we'll document some potential future directions we can go in to improve the technique."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "46c2533c-6032-49df-9547-4a7aac1bf0ed",
   "metadata": {},
   "source": [
    "## Downloading the data\n",
    "\n",
    "First, we'll download the data from Yahoo Finance.  We'll save the data after we download it, so we don't have to re-download it every time (this could cause our IP to get blocked).\n",
    "\n",
    "We'll use data for a single stock (Microsoft) from when it started trading to the present."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "id": "abf9c5d8-94b1-40d4-9565-757736f4932a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import finance API and get historical stock data\n",
    "\n",
    "import yfinance as yf\n",
    "import os\n",
    "import json\n",
    "import pandas as pd\n",
    "\n",
    "DATA_PATH = \"msft_data.json\"\n",
    "\n",
    "if os.path.exists(DATA_PATH):\n",
    "    # Read from file if we've already downloaded the data.\n",
    "    with open(DATA_PATH) as f:\n",
    "        msft_hist = pd.read_json(DATA_PATH)\n",
    "else:\n",
    "    msft = yf.Ticker(\"MSFT\")\n",
    "    msft_hist = msft.history(period=\"max\")\n",
    "\n",
    "    # Save file to json in case we need it later.  This prevents us from having to re-download it every time.\n",
    "    msft_hist.to_json(DATA_PATH)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9693d9aa-81bb-471f-8601-0970c7da88ca",
   "metadata": {},
   "source": [
    "As we can see, we have one row of data for each day that Microsoft stock was traded.  Here are the columns:\n",
    "\n",
    "* Open - the price the stock opened at.\n",
    "* High - the highest price during the day\n",
    "* Low - the lowest price during the day\n",
    "* Close - the closing price on the trading day\n",
    "* Volume - how many shares were traded\n",
    "\n",
    "Stock doesn't trade every day (there is no trading on weekends and holidays), so some dates are missing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "id": "e33e6f49-7233-40b7-86ea-96d4a6978a01",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Open</th>\n",
       "      <th>High</th>\n",
       "      <th>Low</th>\n",
       "      <th>Close</th>\n",
       "      <th>Volume</th>\n",
       "      <th>Dividends</th>\n",
       "      <th>Stock Splits</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1986-03-13</th>\n",
       "      <td>0.055898</td>\n",
       "      <td>0.064119</td>\n",
       "      <td>0.055898</td>\n",
       "      <td>0.061378</td>\n",
       "      <td>1031788800</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1986-03-14</th>\n",
       "      <td>0.061378</td>\n",
       "      <td>0.064667</td>\n",
       "      <td>0.061378</td>\n",
       "      <td>0.063570</td>\n",
       "      <td>308160000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1986-03-17</th>\n",
       "      <td>0.063570</td>\n",
       "      <td>0.065215</td>\n",
       "      <td>0.063570</td>\n",
       "      <td>0.064667</td>\n",
       "      <td>133171200</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1986-03-18</th>\n",
       "      <td>0.064667</td>\n",
       "      <td>0.065215</td>\n",
       "      <td>0.062474</td>\n",
       "      <td>0.063022</td>\n",
       "      <td>67766400</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1986-03-19</th>\n",
       "      <td>0.063022</td>\n",
       "      <td>0.063570</td>\n",
       "      <td>0.061378</td>\n",
       "      <td>0.061926</td>\n",
       "      <td>47894400</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                Open      High       Low     Close      Volume  Dividends  \\\n",
       "1986-03-13  0.055898  0.064119  0.055898  0.061378  1031788800        0.0   \n",
       "1986-03-14  0.061378  0.064667  0.061378  0.063570   308160000        0.0   \n",
       "1986-03-17  0.063570  0.065215  0.063570  0.064667   133171200        0.0   \n",
       "1986-03-18  0.064667  0.065215  0.062474  0.063022    67766400        0.0   \n",
       "1986-03-19  0.063022  0.063570  0.061378  0.061926    47894400        0.0   \n",
       "\n",
       "            Stock Splits  \n",
       "1986-03-13           0.0  \n",
       "1986-03-14           0.0  \n",
       "1986-03-17           0.0  \n",
       "1986-03-18           0.0  \n",
       "1986-03-19           0.0  "
      ]
     },
     "execution_count": 192,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Display microsoft stock price history so we can look at the structure of the data\n",
    "msft_hist.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0f876a5d-3256-496b-b74d-06356789b07a",
   "metadata": {},
   "source": [
    "Next, we'll plot the data so we can see how the stock price has changed over time.  This gives us another overview of the structure of the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "id": "fadf28ca-5a8c-421e-b4ef-cd02def0a895",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 193,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualize microsoft stock prices\n",
    "msft_hist.plot.line(y=\"Close\", use_index=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "71a6f21e-affe-43ee-aa38-729acc40bbb0",
   "metadata": {},
   "source": [
    "## Preparing the data\n",
    "\n",
    "Ok, hopefully you've stopped kicking yourself for not buying Microsoft stock at any point in the past 30 years now. \n",
    "\n",
    "Now, let's prepare the data so we can make predictions.  We'll be predicting if the price will go up or down tomorrow based on data from today.\n",
    "\n",
    "First, we'll identify a target that we're trying to predict.  Our target will be if the price will go up or down tomorrow.  If the price went up, the target will be `1` and if it went down, the target will be `0`.\n",
    "\n",
    "Next, we'll shift the data from previous days \"forward\" one day, so we can use it to predict the target price.  This ensures that we don't accidentally use data from the same day to make predictions! (a very common mistake)\n",
    "\n",
    "Then, we'll combine both so we have our training data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "id": "05310dc8-05bc-4513-aac6-2ae85b82db69",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ensure we know the actual closing price\n",
    "data = msft_hist[[\"Close\"]]\n",
    "data = data.rename(columns = {'Close':'Actual_Close'})\n",
    "\n",
    "# Setup our target.  This identifies if the price went up or down\n",
    "data[\"Target\"] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])[\"Close\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "id": "fd644358-2713-4004-b2e5-be7150ae38ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.\n",
    "msft_prev = msft_hist.copy()\n",
    "msft_prev = msft_prev.shift(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "id": "eea608f5-c7ef-48e3-993e-d2f577a6e34c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create our training data\n",
    "predictors = [\"Close\", \"Volume\", \"Open\", \"High\", \"Low\"]\n",
    "data = data.join(msft_prev[predictors]).iloc[1:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "id": "b46f735a-2b79-42a6-ae48-3327bdc200f3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual_Close</th>\n",
       "      <th>Target</th>\n",
       "      <th>Close</th>\n",
       "      <th>Volume</th>\n",
       "      <th>Open</th>\n",
       "      <th>High</th>\n",
       "      <th>Low</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1986-03-14</th>\n",
       "      <td>0.063570</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.061378</td>\n",
       "      <td>1.031789e+09</td>\n",
       "      <td>0.055898</td>\n",
       "      <td>0.064119</td>\n",
       "      <td>0.055898</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1986-03-17</th>\n",
       "      <td>0.064667</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.063570</td>\n",
       "      <td>3.081600e+08</td>\n",
       "      <td>0.061378</td>\n",
       "      <td>0.064667</td>\n",
       "      <td>0.061378</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1986-03-18</th>\n",
       "      <td>0.063022</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.064667</td>\n",
       "      <td>1.331712e+08</td>\n",
       "      <td>0.063570</td>\n",
       "      <td>0.065215</td>\n",
       "      <td>0.063570</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1986-03-19</th>\n",
       "      <td>0.061926</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.063022</td>\n",
       "      <td>6.776640e+07</td>\n",
       "      <td>0.064667</td>\n",
       "      <td>0.065215</td>\n",
       "      <td>0.062474</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1986-03-20</th>\n",
       "      <td>0.060282</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.061926</td>\n",
       "      <td>4.789440e+07</td>\n",
       "      <td>0.063022</td>\n",
       "      <td>0.063570</td>\n",
       "      <td>0.061378</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Actual_Close  Target     Close        Volume      Open      High  \\\n",
       "1986-03-14      0.063570     1.0  0.061378  1.031789e+09  0.055898  0.064119   \n",
       "1986-03-17      0.064667     1.0  0.063570  3.081600e+08  0.061378  0.064667   \n",
       "1986-03-18      0.063022     0.0  0.064667  1.331712e+08  0.063570  0.065215   \n",
       "1986-03-19      0.061926     0.0  0.063022  6.776640e+07  0.064667  0.065215   \n",
       "1986-03-20      0.060282     0.0  0.061926  4.789440e+07  0.063022  0.063570   \n",
       "\n",
       "                 Low  \n",
       "1986-03-14  0.055898  \n",
       "1986-03-17  0.061378  \n",
       "1986-03-18  0.063570  \n",
       "1986-03-19  0.062474  \n",
       "1986-03-20  0.061378  "
      ]
     },
     "execution_count": 197,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c04059b2-5bba-4233-8250-4d11f243130a",
   "metadata": {},
   "source": [
    "## Creating a machine learning model\n",
    "\n",
    "Next, we'll create a machine learning model to see how accurately we can predict the stock price.\n",
    "\n",
    "Because we're dealing with time series data, we can't just use cross-validation to create predictions for the whole dataset.  This will cause leakage where data from the future will be used to predict past prices.  This doesn't match with the real world, and will make us think that our algorithm is much better than it actually is.\n",
    "\n",
    "Instead, we'll split the data sequentially.  We'll start off by predicting just the last 100 rows using the other rows.\n",
    "\n",
    "We'll use a random forest classifier to generate our predictions.  This is a good \"default\" model for a lot of applications, because it can pick up nonlinear relationships in the data, and is somewhat robust to overfitting with the right parameters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "id": "b39915f4-912e-4ccd-b1a6-045a1e3ebe9c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(min_samples_split=200, random_state=1)"
      ]
     },
     "execution_count": 198,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import numpy as np\n",
    "\n",
    "# Create a random forest classification model.  Set min_samples_split high to ensure we don't overfit.\n",
    "model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)\n",
    "\n",
    "# Create a train and test set\n",
    "train = data.iloc[:-100]\n",
    "test = data.iloc[-100:]\n",
    "\n",
    "model.fit(train[predictors], train[\"Target\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1e7c7e8f-987f-415d-a650-bae50cb9c623",
   "metadata": {},
   "source": [
    "Next, we'll need to check how accurate the model was.  Earlier, we mentioned using `precision` to measure error.  We can do this by using the `precision_score` function from scikit-learn."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "id": "5166dc96-9c6b-476a-9755-6f2e6a2e45f1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.51"
      ]
     },
     "execution_count": 199,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import precision_score\n",
    "\n",
    "# Evaluate error of predictions\n",
    "preds = model.predict(test[predictors])\n",
    "preds = pd.Series(preds, index=test.index)\n",
    "precision_score(test[\"Target\"], preds)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ad90cf27-9308-41b8-87eb-fe0b99f09e08",
   "metadata": {},
   "source": [
    "So our model is directionally accurate 51% of the time.  This is only a little bit better than a coin flip!  We can take a deeper look at the individual predictions and the actuals, and see where we're off."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 200,
   "id": "bf74e107-1fea-447d-a659-ddf18c016b5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 200,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "combined = pd.concat({\"Target\": test[\"Target\"],\"Predictions\": preds}, axis=1)\n",
    "combined.plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "deed266d-549c-4fc7-9934-5b9b249d0376",
   "metadata": {},
   "source": [
    "## Backtesting\n",
    "\n",
    "Our model isn't great, but luckily we can still improve it.  Before we do that, let's figure out how to make predictions across the entire dataset, not just the last 100 rows.  This will give us a more robust error estimate.  The last 100 days may have has atypical market conditions or other issues that make error metrics on those days unrealistic for future predictions (which are what we really care about).\n",
    "\n",
    "To do this, we'll need to backtest.  Backtesting ensures that we only use data from before the day that we're predicting.  If we use data from after the day we're predicting, the algorithm is unrealistic (in the real world, you won't be able to use future data to predict that past!).\n",
    "\n",
    "Our backtesting method will loop over the dataset, and train a model every 750 rows.  We'll make it a function so we can avoid rewriting the code if we want to backtest again.\n",
    "\n",
    "In the backtesting function, we will:\n",
    "\n",
    "* Split the training and test data\n",
    "* Train a model\n",
    "* Make predictions on the test data using `predict_proba` - this is because we want to really optimize for true positives.  By default, the threshold for splitting 0/1 is .5, but we can set it to different values to tweak the precision.  If we set it too high, we'll make fewer trades, but will have a lower potential for losses."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "id": "9687c1d4-1be0-47f4-9fa7-cf8452c74c72",
   "metadata": {},
   "outputs": [],
   "source": [
    "def backtest(data, model, predictors, start=1000, step=750):\n",
    "    predictions = []\n",
    "    # Loop over the dataset in increments\n",
    "    for i in range(start, data.shape[0], step):\n",
    "        # Split into train and test sets\n",
    "        train = data.iloc[0:i].copy()\n",
    "        test = data.iloc[i:(i+step)].copy()\n",
    "        \n",
    "        # Fit the random forest model\n",
    "        model.fit(train[predictors], train[\"Target\"])\n",
    "        \n",
    "        # Make predictions\n",
    "        preds = model.predict_proba(test[predictors])[:,1]\n",
    "        preds = pd.Series(preds, index=test.index)\n",
    "        preds[preds > .6] = 1\n",
    "        preds[preds<=.6] = 0\n",
    "        \n",
    "        # Combine predictions and test values\n",
    "        combined = pd.concat({\"Target\": test[\"Target\"],\"Predictions\": preds}, axis=1)\n",
    "        \n",
    "        predictions.append(combined)\n",
    "    \n",
    "    return pd.concat(predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "id": "6a5ee41b-cc06-45fd-af4a-e7fbfda0bd47",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = backtest(data, model, predictors)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d04f08f0-93c9-4e48-906b-095c9a48a5e5",
   "metadata": {},
   "source": [
    "As you can see, we're only making 742 trades.  This is because we used `.6` as a threshold for trading."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 203,
   "id": "6920d4b0-98a0-4ede-aa71-daa00d9dee58",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.0    7264\n",
       "1.0     742\n",
       "Name: Predictions, dtype: int64"
      ]
     },
     "execution_count": 203,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predictions[\"Predictions\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "id": "e88b3291-e672-4115-b825-a72f298e0fd2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5053908355795148"
      ]
     },
     "execution_count": 204,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision_score(predictions[\"Target\"], predictions[\"Predictions\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1bfe3afb-c02b-49a1-80b6-61af0a07b680",
   "metadata": {},
   "source": [
    "## Improving accuracy\n",
    "\n",
    "The model isn't very accurate, but at least now we can make predictions across the entire history of the stock.  For this model to be useful, we have to get it to predict more accurately.\n",
    "\n",
    "Let's add some more predictors to see if we can improve accuracy.\n",
    "\n",
    "We'll add in some rolling means, so the model can evaluate the current price against recent prices.  We'll also look at the ratios between different indicators."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 222,
   "id": "c626bb8e-4798-4ae2-a71d-aa35ee3a2f1f",
   "metadata": {},
   "outputs": [],
   "source": [
    "weekly_mean = data.rolling(7).mean()\n",
    "quarterly_mean = data.rolling(90).mean()\n",
    "annual_mean = data.rolling(365).mean()\n",
    "weekly_trend = data.shift(1).rolling(7).mean()[\"Target\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 223,
   "id": "9e5f7d62-50fc-428f-9bbb-91976265d61c",
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"weekly_mean\"] = weekly_mean[\"Close\"] / data[\"Close\"]\n",
    "data[\"quarterly_mean\"] = quarterly_mean[\"Close\"] / data[\"Close\"]\n",
    "data[\"annual_mean\"] = annual_mean[\"Close\"] / data[\"Close\"]\n",
    "\n",
    "data[\"annual_weekly_mean\"] = data[\"annual_mean\"] / data[\"weekly_mean\"]\n",
    "data[\"annual_quarterly_mean\"] = data[\"annual_mean\"] / data[\"quarterly_mean\"]\n",
    "data[\"weekly_trend\"] = weekly_trend\n",
    "\n",
    "data[\"open_close_ratio\"] = data[\"Open\"] / data[\"Close\"]\n",
    "data[\"high_close_ratio\"] = data[\"High\"] / data[\"Close\"]\n",
    "data[\"low_close_ratio\"] = data[\"Low\"] / data[\"Close\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 224,
   "id": "5248e566-f28f-4b0b-b042-31c794193bce",
   "metadata": {},
   "outputs": [],
   "source": [
    "full_predictors = predictors + [\"weekly_mean\", \"quarterly_mean\", \"annual_mean\", \"annual_weekly_mean\", \"annual_quarterly_mean\", \"open_close_ratio\", \"high_close_ratio\", \"low_close_ratio\", \"weekly_trend\"]\n",
    "predictions = backtest(data.iloc[365:], model, full_predictors)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 225,
   "id": "67a69920-e030-4975-ac80-dced781d546c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6616161616161617"
      ]
     },
     "execution_count": 225,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision_score(predictions[\"Target\"], predictions[\"Predictions\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 226,
   "id": "d0be9777-de2e-4b42-8190-b6aff9760ac1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.0    7443\n",
       "1.0     198\n",
       "Name: Predictions, dtype: int64"
      ]
     },
     "execution_count": 226,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Show how many trades we would make\n",
    "\n",
    "predictions[\"Predictions\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 210,
   "id": "943f1100-44a4-4b62-b9dc-309dccfe8c61",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 210,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "predictions.iloc[-100:].plot()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6743b773-8383-4615-857c-ffccee6ef0c8",
   "metadata": {},
   "source": [
    "## Next steps\n",
    "\n",
    "We've come far in this project!  We've downloaded and cleaned data, and setup a backtesting engine.  We now have an algorithm that we can add more predictors to and continue to improve the accuracy of.\n",
    "\n",
    "There are a lot of next steps we could take to improve our predictions:\n",
    "\n",
    "### Improve the technique\n",
    "\n",
    "* Calculate how much money you'd make if you traded with this algorithm\n",
    "\n",
    "### Improve the algorithm\n",
    "\n",
    "* Run with a reduced step size!  This will take longer, but increase accuracy\n",
    "* Try discarding older data (only keeping data in a certain window)\n",
    "* Try a different machine learning algorithm\n",
    "* Tweak random forest parameters, or the prediction threshold\n",
    "\n",
    "### Add in more predictors\n",
    "\n",
    "* Account for activity post-close and pre-open\n",
    "    * Early trading\n",
    "    * Trading on other exchanges that open before the NYSE (to see what the global sentiment is)\n",
    "* Economic indicators\n",
    "    * Interest rates\n",
    "    * Other important economic news\n",
    "* Key dates\n",
    "    * Dividends\n",
    "    * External factors like elections\n",
    "* Company milestones\n",
    "    * Earnings calls\n",
    "    * Analyst ratings\n",
    "    * Major announcements\n",
    "* Prices of related stocks\n",
    "    * Other companies in the same sector\n",
    "    * Key partners, customers, etc."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
