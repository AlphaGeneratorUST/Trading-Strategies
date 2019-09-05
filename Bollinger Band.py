import pandas as pd
from pandas_datareader import data, wb
import numpy as np
import matplotlib.pyplot as plt
import quandl

df = quandl.get("CHRIS/EUREX_FDAX1", start_date="2015-01-01")
df.head()

window = 21
no_of_std = 2

rolling_mean = df['Settle'].rolling(window).mean()
rolling_std = df['Settle'].rolling(window).std()

df['Rolling Mean'] = rolling_mean
df['Bollinger High'] = rolling_mean + (no_of_std * rolling_std)
df['Bolling Low'] = rolling_mean - (no_of_std * rolling_std)

df[['Settle', 'Bollinger High', 'Bolling Low']].plot()

#Strategy
df['Position'] = None

for row in range(len(df)):
    if (df['Settle'].iloc[row] > df['Bollinger High'].iloc[row]) and (df['Settle'].iloc[row-1] < df['Bollinger High'].iloc[row-1]):
        df['Position'].iloc[row] = -1

    if (df['Settle'].iloc[row] < df['Bolling Low'].iloc[row]) and (df['Settle'].iloc[row-1] > df['Bolling Low'].iloc[row-1]):
        df['Position'].iloc[row] = 1

#Forward fill our position column to replace the "None" values with the correct long/shor positions to represent the "holding" of our position
df['Position'].fillna(method='ffill', inplace=True)

#Calcutlate the daily market return and multiply that by the position to determine strategy returns
df['Market Return'] = np.log(df['Settle']/df['Settle'].shift(1))
df['Strategy Return'] = df['Market Return'] * df['Position']

df['Strategy Return'].cumsum().plot()

#Define a "Bollinger Band trading strategy
def bollinger_strat(df, window, std):
    rolling_mean = df['Settle'].rolling(window).mean()
    rolling_std = df['Settle'].rolling(window).std()

    df['Bollinger High'] = rolling_mean + (rolling_std * no_of_std)
    df['Bollinger Low'] = rolling_mean - (rolling_std * no_of_std)

    df['Long'] = None
    df['Short'] = None
    df['Position'] = None

    for row in range(len(df)):
        if (df['Settle'].iloc[row] > df['Bollinger High'].iloc[row]) and (df['Settle'].iloc[row-1] < df['Bollinger High'].iloc[row-1]):
            df['Position'].iloc[row] = -1

        if (df['Settle'].iloc[row] < df['Bollinger Low'].iloc[row]) and (df['Settle'].iloc[row-1] > df['Bollinger Low'].iloc[row-1]):
            df['Position'].iloc[row] = 1

    df['Position'].fillna(method='ffill', inplace=True)

    df['Market Return'] = np.log(df['Settle']/df['Settle'].shift(1))
    df['Strategy Return'] = df['Market Return'] * df['Position']

    df['Strategy Retun'].cumsum().plot()

#Set up "daily look back period" and "number of standard deviation" vectors
windows = np.linspace(10,100,20,dtype=int)
stds = np.linspace(1,3,10)

#Iterate through them both, runing the strategy function each time
for window in windows:
    for std in stds:
        bollinger_strat(df,window,std)
