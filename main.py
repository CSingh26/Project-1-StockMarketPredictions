#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

#loading dataset to view it
data = pd.read_csv('data/infolimpioavanzadoTarget.csv')

# print(data.head())
# print(data.shape)
# print(data.describe())

# Time series stock graph (first 4)
def stockPlot(stockName, ax):
    stockData = data[data['ticker'] == stockName]
    stockData = stockData[['date','open', 'high', 'low', 'close']]
    stockData['date'] = pd.to_datetime(stockData['date'])

    colors = ['blue', 'green', 'red', 'orange']
    labels = ['Open Price', 'Close Price', 'High Price', 'Low Price']
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(stockData['date'], stockData.iloc[:, i+1], color=color, label=label)

    ax.set_xlabel('Date')
    ax.set_ylabel('Date')
    ax.set_title(stockName + ' Prices')
    ax.legend()

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

fig, axes = plt.subplots(2, 2, figsize=(100,10))

for ax, stock in zip(axes.flatten(), (data['ticker'].unique()[:4])):
    stockPlot(stock, ax)

plt.tight_layout()
plt.show()

#Avg Stock Prices
avgStockPrice = {}

for stock in data['ticker'].unique():
    ticker_data = data[data['ticker'] == stock]
    weighted_avg = np.average(ticker_data['close'], weights=ticker_data['volume'])
    avgStockPrice[stock] = weighted_avg

numericAvgStockPrice = {key: value for key, value in avgStockPrice.items() if isinstance(value, (int, float))}

plt.figure(figsize=(100,10))
plt.bar(range(len(numericAvgStockPrice)), numericAvgStockPrice.values(), align='center')
plt.xlabel('Stock Names')
plt.ylabel('Avg Price')

plt.xticks(range(len(numericAvgStockPrice)), list(numericAvgStockPrice.keys()), rotation='vertical')

plt.title('Avg Stock Prices')

for i, v in enumerate(numericAvgStockPrice.values()):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')

plt.show()

# Realtive Strength Index (RSI) for first 4 stocks
def rsi(data, window=14):
    delta = data['close'].diff()

    gain = delta.where(delta > 0.0)
    loss = -delta.where(delta < 0.0)

    avgGain = gain.rolling(window=window, min_periods=1).mean()
    avgLoss = loss.rolling(window=window, min_periods=1).mean()

    rsi = 100 - (100 / (1 + (avgGain / avgLoss)))

    return rsi

def plotRSI(data, ticker, ax):
    data['date'] = pd.to_datetime(data['date'])
    ax.plot(data['date'], data['RSI'], label='RSI', color='orange')
    ax.set_title(f'Relative Strength Index (RSI) for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    ax.legend()
    ax.grid(True)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

fig, axes = plt.subplots(2, 2, figsize=(100,10))

for ax, stock in zip(axes.flatten(), (data['ticker'].unique()[:4])):
    stockName = data[data['ticker'] == stock].copy() 
    stockName['RSI'] = rsi(stockName)
    plotRSI(stockName, stock, ax)

plt.tight_layout()
plt.show()