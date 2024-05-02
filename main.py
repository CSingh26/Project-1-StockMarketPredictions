#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

#loading dataset to view it
data = pd.read_csv('data/infolimpioavanzadoTarget.csv')

# Displaying the first few rows, shape, and basic statistics of the dataset
print(data.head())
print(data.shape)
print(data.describe())

# Exploratory Data Analysis

# Time series stock graph (first 4)
def stockPlot(stockName, ax):
    # Filtering data for the specific stock
    stockData = data[data['ticker'] == stockName]
    stockData = stockData[['date','open', 'high', 'low', 'close']]
    stockData['date'] = pd.to_datetime(stockData['date'])

    # Plotting open, high, low, close prices over time
    colors = ['blue', 'green', 'red', 'orange']
    labels = ['Open Price', 'close Price', 'High Price', 'Low Price']
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(stockData['date'], stockData.iloc[:, i+1], color=color, label=label)

    # Adding labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Date')
    ax.set_title(stockName + ' Prices')
    ax.legend()

    # Formatting x-axis with month labels
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

# Creating subplots for each stock's time series graph
fig, axes = plt.subplots(2, 2, figsize=(100,10))

for ax, stock in zip(axes.flatten(), (data['ticker'].unique()[:4])):
    stockPlot(stock, ax)

plt.tight_layout()
plt.show()

# Avg Stock Prices
# Calculating the weighted average stock price for each stock
avgStockPrice = {}
for stock in data['ticker'].unique():
    ticker_data = data[data['ticker'] == stock]
    weighted_avg = np.average(ticker_data['close'], weights=ticker_data['volume'])
    avgStockPrice[stock] = weighted_avg

# Filtering out non-numeric average stock prices
numericAvgStockPrice = {key: value for key, value in avgStockPrice.items() if isinstance(value, (int, float))}

# Plotting average stock prices
plt.figure(figsize=(100,10))
plt.bar(range(len(numericAvgStockPrice)), numericAvgStockPrice.values(), align='center')
plt.xlabel('Stock Names')
plt.ylabel('Avg Price')
plt.xticks(range(len(numericAvgStockPrice)), list(numericAvgStockPrice.keys()), rotation='vertical')
plt.title('Avg Stock Prices')

# Adding value labels on top of each bar
for i, v in enumerate(numericAvgStockPrice.values()):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')

plt.show()

# Realtive Strength Index (RSI) for first 4 stocks
# Function to calculate Relative Strength Index (RSI)
def rsi(data, window=14):
    delta = data['close'].diff()
    gain = delta.where(delta > 0.0)
    loss = -delta.where(delta < 0.0)
    avgGain = gain.rolling(window=window, min_periods=1).mean()
    avgLoss = loss.rolling(window=window, min_periods=1).mean()
    rsi = 100 - (100 / (1 + (avgGain / avgLoss)))
    return rsi

# Function to plot RSI for a stock
def plotRSI(data, ticker, ax):
    data['date'] = pd.to_datetime(data['date'])
    ax.plot(data['date'], data['RSI'], label='RSI', color='orange')
    ax.set_title(f'Relative Strength Index (RSI) for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    ax.legend()
    ax.grid(True)

# Creating subplots for RSI of first 4 stocks
fig, axes = plt.subplots(2, 2, figsize=(100,10))

for ax, stock in zip(axes.flatten(), (data['ticker'].unique()[:4])):
    stockName = data[data['ticker'] == stock].copy() 
    stockName['RSI'] = rsi(stockName)
    plotRSI(stockName, stock, ax)

plt.tight_layout()
plt.show()

# Predictive Modeling 

# Doing for single stocks (Example: 'ATLC')
# Preparing data for prediction
asleStock = data[data['ticker'] == 'ATLC']
asleStock = asleStock[['date', 'close', 'open', 'high', 'low']]

# Creating target variable (1 if tomorrow's close price > today's close price, else 0)
asleStock['tomorrow'] = asleStock['close'].shift(-1)
asleStock['target'] = (asleStock['tomorrow'] > asleStock['close']).astype(int)

# Initializing Random Forest classifier
model = RandomForestClassifier(n_estimators=250, min_samples_split= 50, random_state=1)

# Splitting data into train and test sets
train = asleStock.iloc[:-100]
test = asleStock.iloc[-100:]

# Features for prediction
predictors = ["open", "high", "low", "close"]

# Fitting the model
model.fit(train[predictors], train['target'])

# Function to make predictions and evaluate precision
def predict(train, test, predictors, model):
    model.fit(train[predictors], train['target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test['target'], preds], axis=1)
    return combined

# Function to perform backtesting
def backtest(data, model, predictors, start=150, step=50):
    allPredictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i]
        test = data.iloc[i:(i+step)]
        predictions = predict(train, test, predictors, model)
        allPredictions.append(predictions)
    return pd.concat(allPredictions)

# Backtesting and evaluating precision
predictions = backtest(asleStock, model, predictors)
print(precision_score(predictions['target'], predictions['Predictions']))

# Refining prediction model for all stocks in the datasets
# Function to train and predict for all stocks
def trainPredict(data, modelParams, horizonParams, backTestParams):
    results = {}
    for stock in data['ticker'].unique():
        stockData = data[data['ticker'] == stock]
        stockData = stockData[['date', 'close', 'open', 'high', 'low']]
        stockData['tomorrow'] = stockData['close'].shift(-1)
        stockData['target'] = (stockData['tomorrow'] > stockData['close']).astype(int)
        model = RandomForestClassifier(**modelParams)
        horizons = horizonParams['horizons']
        predictors = []
        stockData.iloc[:, 1:] = stockData.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        for horizon in horizons:
            avgs = stockData.iloc[:, 1:].rolling(horizon, axis=0).mean()
            ratio = f"closeRatio_{horizon}"
            stockData[ratio] = stockData['close'] / avgs['close']
            trend = f"tred_{horizon}"
            stockData[trend] = stockData.iloc[:, 1:].shift(1).rolling(horizon).sum()["target"]
            predictors += [ratio, trend]
        predictions = backtest(stockData, model, predictors, **backTestParams)
        precision = precision_score(predictions['target'], predictions['Predictions'])
        results[stock] = {
                'precision': precision,
                'predictions': predictions
        }
    return results

# Training and predicting for all stocks
modelParams = {'n_estimators': 250, 'min_samples_split': 50, 'random_state': 1}
horizonParams = {'horizons': [2, 5, 60, 250]}
backtestParams = {'start': 150, 'step': 50}

results = trainPredict(data, modelParams, horizonParams, backtestParams)

# Printing precision and value counts of predictions for each stock
for stock, res in results.items():
    print(f"Ticker: {stock}")
    print(f"Precision: {res['precision']}")
    print(res['predictions']['Predictions'].value_counts())

# Prediction model
# Splitting data into features (X) and target (y)
X = data[['open', 'high', 'low', 'volume']]  
y = data['close']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

# Initializing Linear Regression model
linReg = LinearRegression()

# Fitting the model
linReg.fit(X_train, y_train)

# Sample Data
date_to_predict = '2024-04-29'  
opening_price = 6.95 
high_price = 7.10  
low_price = 6.89  
volume = 318659

# Predicting closing price for sample data
X_pred = np.array([[opening_price, high_price, low_price, volume]])
predicted_price = linReg.predict(X_pred)
print("Predicted closing price for", date_to_predict, ":", predicted_price)