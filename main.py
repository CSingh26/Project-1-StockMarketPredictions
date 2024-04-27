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

#Time series stock graph (first 4)
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
