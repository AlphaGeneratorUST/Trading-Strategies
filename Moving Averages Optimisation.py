import pandas as pd
import numpy as np
from pandas_datareader import data
from math import sqrt
import matplotlib.pyplot as plt


def ma_strat(ticker, short_ma, long_ma):
    # Read in data from Yahoo Finance for the relevant ticker
    sp500 = data.DataReader(ticker, data_source='yahoo', start='2016-01-01')
    sp500['short_ma'] = np.round(sp500['Close'].rolling(window=short_ma).mean(),2)
    sp500['long_ma'] = np.round(sp500['Close'].rolling(window=long_ma).mean(),2)
    sp500['short_ma-long_ma'] = sp500['short_ma'] - sp500['long_ma']

    X = 50
    sp500['Stance'] = np.where(sp500['short_ma-long_ma'] > X, 1, 0)
    sp500['Stance'] = np.where(sp500['short_ma-long_ma'] < -X, -1, sp500['Stance'])
    sp500['Stance'].value_counts()

    # Create columns containing daily market log returns and strategy daily log returns
    sp500['Market Returns'] = np.log(sp500['Close']/sp500['Close'].shift(1))
    sp500['Strategy'] = sp500['Market Returns'] * sp500['Stance'].shift(1)

    # Set strategy starting equity to 1 and generate equity curve
    sp500['Strategy Curve'] = sp500['Strategy'].cumsum() + 1
    sharpe = annualised_sharpe(sp500['Strategy'])

    return (sp500['Strategy'].cumsum()[-1], sharpe)


# Calculate Sharpe ratio
def annualised_sharpe(returns, N=252):
    return np.sqrt(N) * returns.mean() / returns.std()

short_ma = np.linspace(10,60,25,dtype=int)
long_ma = np.linspace(220,270,25,dtype=int)

results_pnl = np.zeros((len(short_ma), len(long_ma)))
results_sharpe = np.zeros((len(short_ma), len(long_ma)))

for i, shortma in enumerate(short_ma):
    for j, longma in enumerate(long_ma):
        pnl, sharpe = ma_strat('^GSPC', shortma, longma)
        results_pnl[i,j] = pnl
        results_sharpe[i,j] = sharpe

plt.pcolor(short_ma, long_ma, results_pnl)
plt.colorbar()
plt.show()

plt.pcolor(short_ma, long_ma, results_sharpe)
plt.colorbar()
plt.show()