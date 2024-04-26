#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#loading dataset to view it
data = pd.read_csv('data/infolimpioavanzadoTarget.csv')

# print(data.head())
# print(data.shape)
# print(data.describe())

#ASLE stock data and prediction 
# asleData = data[data['ticker'] == 'ASLE']
# asleData = asleData[['date','open', 'high', 'low', 'close']]
# asleData['date'] = pd.to_datetime(asleData['date'])
# asleData['month'] = pd.to_datetime(asleData['date']).dt.strftime('%B') 
# asleData['day'] = pd.to_datetime(asleData['date']).dt.day

# # print(asleData.head())

# plt.figure(figsize=(50, 10))
# plt.plot(asleData['date'], asleData['open'], color='blue', label='Open Price')
# plt.plot(asleData['date'], asleData['close'], color='green', label='Close Price')
# plt.plot(asleData['date'], asleData['high'], color='red', label='High Price')
# plt.plot(asleData['date'], asleData['low'], color='orange', label='Low Price')

# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.title('ASLE Stock Prices')
# plt.legend()

# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))

# plt.tight_layout()
# plt.show()

#ASLN stock data
# aslnData = data[data['ticker'] == 'ASLN']
# aslnData = aslnData[['date','open', 'high', 'low', 'close']]
# aslnData['date'] = pd.to_datetime(aslnData['date'])
# aslnData['month'] = pd.to_datetime(aslnData['date']).dt.strftime('%B') 
# aslnData['day'] = pd.to_datetime(aslnData['date']).dt.day

# # print(aslnData.head())

# plt.figure(figsize=(50, 10))
# plt.plot(aslnData['date'], aslnData['open'], color='blue', label='Open Price')
# plt.plot(aslnData['date'], aslnData['close'], color='green', label='Close Price')
# plt.plot(aslnData['date'], aslnData['high'], color='red', label='High Price')
# plt.plot(aslnData['date'], aslnData['low'], color='orange', label='Low Price')

# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.title('ASLN Stock Prices')
# plt.legend()

# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))

# plt.tight_layout()
# plt.show()

#ASMB stock data
# asmbData = data[data['ticker'] == 'ASMB']
# asmbData = asmbData[['date','open', 'high', 'low', 'close']]
# asmbData['date'] = pd.to_datetime(asmbData['date'])
# asmbData['month'] = pd.to_datetime(asmbData['date']).dt.strftime('%B') 
# asmbData['day'] = pd.to_datetime(asmbData['date']).dt.day

# # print(asmbData.head())

# plt.figure(figsize=(50, 10))
# plt.plot(asmbData['date'], asmbData['open'], color='blue', label='Open Price')
# plt.plot(asmbData['date'], asmbData['close'], color='green', label='Close Price')
# plt.plot(asmbData['date'], asmbData['high'], color='red', label='High Price')
# plt.plot(asmbData['date'], asmbData['low'], color='orange', label='Low Price')

# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.title('ASMB Stock Prices')
# plt.legend()

# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))

# plt.tight_layout()
# plt.show()

#ASML stock data
# asmlData = data[data['ticker'] == 'ASML']
# asmlData = asmlData[['date','open', 'high', 'low', 'close']]
# asmlData['date'] = pd.to_datetime(asmlData['date'])
# asmlData['month'] = pd.to_datetime(asmlData['date']).dt.strftime('%B') 
# asmlData['day'] = pd.to_datetime(asmlData['date']).dt.day

# # print(asmlData.head())

# plt.figure(figsize=(50, 10))
# plt.plot(asmlData['date'], asmlData['open'], color='blue', label='Open Price')
# plt.plot(asmlData['date'], asmlData['close'], color='green', label='Close Price')
# plt.plot(asmlData['date'], asmlData['high'], color='red', label='High Price')
# plt.plot(asmlData['date'], asmlData['low'], color='orange', label='Low Price')

# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.title('ASML Stock Prices')
# plt.legend()

# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))

# plt.tight_layout()
# plt.show()

