import pandas as pd
import numpy as np
from pandas_datareader import data,wb
import matplotlib as mpl
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import datetime
import matplotlib.pyplot as plt

#import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.offline import init_notebook_mode, plot, iplot


start = datetime.datetime(2017, 1, 1)
end = datetime.datetime(2019, 1, 27)
d = data.DataReader('F', 'yahoo', start, end)
#print(d.head(10))

# Convert dates to integer values for later use with the matplotlib candlestick charting function
d['Dates'] = mdates.date2num(d.index)

# Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
nine_period_high = d['High'].rolling(window=9).max()
nine_period_low = d['Low'].rolling(window=9).min()
d['tenkan_sen'] = (nine_period_high + nine_period_low)/2

# Kijun-sen (Base Line): (26-period high + 26-period low)/2
period26_high = d['High'].rolling(window=26).max()
period26_low = d['Low'].rolling(window=26).min()
d['kijun_sen'] = (period26_high + period26_low)/2

# Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
d['senkou_span_a'] = ((d['tenkan_sen'] + d['kijun_sen'])/2).shift(26)

# Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
period52_high = d['High'].rolling(window=52).max()
period52_low = d['Low'].rolling(window=52).min()
d['senkou_span_b'] = ((period52_high + period52_low)/2).shift(52)

# The most current closing price plotted 26 time periods behind
d['chikou_span'] = d['Close'].shift(-26)

INCREASING_COLOR = '#17BECF'
DECREASING_COLOR = '#7F7F7F'

d.drop(['Dates', 'Volume'], axis=1).plot(figsize=(15, 8))
#plt.show()

quotes = [tuple(x) for x in d[['Dates','Open','High','Low','Close']].values]

fig, ax = plt.subplots(figsize=(15,8))
d[['tenkan_sen','kijun_sen','senkou_span_a','senkou_span_b','chikou_span']].plot(ax=ax,linewidth=0.5)
candlestick_ohlc(ax,quotes,width=1.0,colorup='g',colordown='r')
#plt.show()

d.dropna(inplace=True)

d['above_cloud'] = 0
d['above_cloud'] = np.where((d['Low'] > d['senkou_span_a']) & (d['Low'] > d['senkou_span_b']), 1, d['above_cloud'])
d['above_cloud'] = np.where((d['High'] < d['senkou_span_a']) & (d['High'] < d['senkou_span_b']), -1, d['above_cloud'])
d['A_above_B'] = np.where((d['senkou_span_a'] > d['senkou_span_b']), 1, -1)

d['tenkan_kiju_cross'] = np.NaN
d['tenkan_kiju_cross'] = np.where((d['tenkan_sen'].shift(1) <= d['kijun_sen'].shift(1)) & (d['tenkan_sen'] > d['kijun_sen']), 1, d['tenkan_kiju_cross'])
d['tenkan_kiju_cross'] = np.where((d['tenkan_sen'].shift(1) >= d['kijun_sen'].shift(1)) & (d['tenkan_sen'] < d['kijun_sen']), 1, d['tenkan_kiju_cross'])

d['price_tenkan_cross'] = np.NaN
d['price_tenkan_cross'] = np.where((d['Open'].shift(1) <= d['tenkan_sen'].shift(1)) & (d['Open'] > d['tenkan_sen']), 1, d['price_tenkan_cross'])
d['price_tenkan_cross'] = np.where((d['Open'].shift(1) >= d['tenkan_sen'].shift(1)) & (d['Open'] < d['tenkan_sen']), -1, d['price_tenkan_cross'])

d['buy'] = np.NaN
d['buy'] = np.where((d['above_cloud'].shift(1) == 1) & (d['A_above_B'].shift(1) == 1) & ((d['tenkan_kiju_cross'].shift(1) == 1) | (d['price_tenkan_cross'].shift(1) == 1)), 1, d['buy'])
d['buy'] = np.where(d['tenkan_kiju_cross'].shift(1) == -1, 0, d['buy'])
d['buy'].ffill(inplace=True)

d['sell'] = np.NaN
d['sell'] = np.where((d['above_cloud'].shift(1) == -1) & (d['A_above_B'].shift(1) == -1) & ((d['tenkan_kiju_cross'].shift(1) == -1) | (d['price_tenkan_cross'].shift(1) == -1)), -1, d['sell'])
d['sell'] = np.where(d['tenkan_kiju_cross'].shift(1) == 1, 0, d['sell'])
d['sell'].ffill(inplace=True)

d['position'] = d['buy'] + d['sell']

d['stock_returns'] = np.log(d['Open']) - np.log(d['Open'].shift(1))
d['strategy_returns'] = d['stock_returns'] * d['position']

d[['stock_returns', 'strategy_returns']].cumsum().plot(figsize=(15,8))
plt.show()



def ichimoku(ticker, start, end):
    d=data.DataReader(ticker, 'yahoo', start, end)[['Open','High','Low','Close']]

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
    nine_period_high = d['High'].rolling(window= 9).max()
    nine_period_low = d['Low'].rolling(window= 9).min()
    d['tenkan_sen'] = (nine_period_high + nine_period_low) /2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    period26_high = d['High'].rolling(window=26).max()
    period26_low = d['Low'].rolling(window=26).min()
    d['kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    d['senkou_span_a'] = ((d['tenkan_sen'] + d['kijun_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    period52_high = d['High'].rolling(window=52).max()
    period52_low = d['Low'].rolling(window=52).min()
    d['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(52)

    # The most current closing price plotted 26 time periods behind (optional)
    d['chikou_span'] = d['Close'].shift(-26)

    d.dropna(inplace=True)

    d['above_cloud'] = 0
    d['above_cloud'] = np.where((d['Low'] > d['senkou_span_a'])  & (d['Low'] > d['senkou_span_b'] ), 1, d['above_cloud'])
    d['above_cloud'] = np.where((d['High'] < d['senkou_span_a']) & (d['High'] < d['senkou_span_b']), -1, d['above_cloud'])
    d['A_above_B'] = np.where((d['senkou_span_a'] > d['senkou_span_b']), 1, -1)

    d['tenkan_kiju_cross'] = np.NaN
    d['tenkan_kiju_cross'] = np.where((d['tenkan_sen'].shift(1) <= d['kijun_sen'].shift(1)) & (d['tenkan_sen'] > d['kijun_sen']), 1, d['tenkan_kiju_cross'])
    d['tenkan_kiju_cross'] = np.where((d['tenkan_sen'].shift(1) >= d['kijun_sen'].shift(1)) & (d['tenkan_sen'] < d['kijun_sen']), -1, d['tenkan_kiju_cross'])

    d['price_tenkan_cross'] = np.NaN
    d['price_tenkan_cross'] = np.where((d['Open'].shift(1) <= d['tenkan_sen'].shift(1)) & (d['Open'] > d['tenkan_sen']), 1, d['price_tenkan_cross'])
    d['price_tenkan_cross'] = np.where((d['Open'].shift(1) >= d['tenkan_sen'].shift(1)) & (d['Open'] < d['tenkan_sen']), -1, d['price_tenkan_cross'])

    d['buy'] = np.NaN
    d['buy'] = np.where((d['above_cloud'].shift(1) == 1) & (d['A_above_B'].shift(1) == 1) & ((d['tenkan_kiju_cross'].shift(1) == 1) | (d['price_tenkan_cross'].shift(1) == 1)), 1, d['buy'])
    d['buy'] = np.where(d['tenkan_kiju_cross'].shift(1) == -1, 0, d['buy'])
    d['buy'].ffill(inplace=True)

    d['sell'] = np.NaN
    d['sell'] = np.where((d['above_cloud'].shift(1) == -1) & (d['A_above_B'].shift(1) == -1) & ((d['tenkan_kiju_cross'].shift(1) == -1) | (d['price_tenkan_cross'].shift(1) == -1)), -1, d['sell'])
    d['sell'] = np.where(d['tenkan_kiju_cross'].shift(1) == 1, 0, d['sell'])
    d['sell'].ffill(inplace=True)

    d['position'] = d['buy'] + d['sell']
    d['stock_returns'] = np.log(d['Open']) - np.log(d['Open'].shift(1))
    d['strategy_returns'] = d['stock_returns'] * d['position']
    d[['stock_returns','strategy_returns']].cumsum().plot(figsize=(15,8))


