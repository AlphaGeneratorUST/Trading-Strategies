import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt


# Download data into DataFrame
df = data.DataReader('AAPL','yahoo',start='1/1/2016')
print(df.head())

df['L14'] = df['Low'].rolling(window=14).min()
df['H14'] = df['High'].rolling(window=14).max()
df['%K'] = 100*((df['Close']-df['L14'])/(df['H14']-df['L14']))
df['%D'] = df['%K'].rolling(window=3).mean()

fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(20,10))
df['Close'].plot(ax=axes[0])
axes[0].set_title('Close')
df[['%K','%D']].plot(ax=axes[1])
axes[1].set_title('Oscillator')
#plt.show()

# A sell is initialed when the %K line crosses down through the %D line and the value of the oscillator is above 80
df['Sell Entry'] = ((df['%K']<df['%D']) & (df['%K'].shift(1)>df['%D'].shift(1))) & (df['%D']>80)

# A sell exit signal is given when the %K line crosses back up through the %D line
df['Sell Exit'] = (df['%K']>df['%D']) & (df['%K'].shift(1)<df['%D'].shift(1))

df['Short'] = np.nan
df.loc[df['Sell Entry'], 'Short'] = -1
df.loc[df['Sell Exit'], 'Short'] = 0

# Set the initial position on day 1 to flat
df['Short'][0] = 0

# Forward fill the position columns to represent the holding of positions through time
df['Short'] = df['Short'].fillna(method='pad')

# Buy signal
df['Buy Entry'] = ((df['%K']>df['%D']) & (df['%K'].shift(1)>df['%D'].shift(1))) & (df['%D']<20)
df['Buy Exit'] = (df['%K']<df['%D']) & (df['%K'].shift(1)>df['%D'].shift(1))

df['Long'] = np.nan
df.loc[df['Buy Entry'], 'Long'] = 1
df.loc[df['Buy Exit'], 'Long'] = 0

df['Long'][0] = 0
df['Long'] = df['Long'].fillna(method='pad')

# Add the Long and Short positions together to get final strategy positions (1 for long, -1 for short and 0 for flat)
df['Position'] = df['Long'] + df['Short']
df['Position'].plot(figsize=(20,10))
#plt.show()

# Set up a column holding the daily Apple returns
df['Market Returns'] = df['Close'].pct_change()

# Create column for strategy returns by multiplying the daily Apple returns by the position that was held at close of the previous day
df['Strategy Returns'] = df['Market Returns'] * df['Position'].shift(1)

df[['Strategy Returns', 'Market Returns']].cumsum().plot()
plt.show()

