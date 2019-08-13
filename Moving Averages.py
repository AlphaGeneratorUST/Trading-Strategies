import pandas as pd
import numpy as np
from pandas_datareader import data
from math import sqrt
import matplotlib.pyplot as plt


#Download data into DataFrame and create moving averages columns
sp500 = data.DataReader('^GSPC','yahoo',start='2000-1-1')
print(sp500.head())

sp500 = pd.DataFrame()
sp500['42d'] = np.round(sp500['Close'].rolling(window=42).mean(), 2)
sp500['252d'] = np.round(sp500['Close'].rolling(window=252).mean(), 2)

#Create column with moving average spread differential
sp500['42-252'] = sp500['42d'] - sp500['252d']

#Set desired number of points as threshold for spread difference and create column containing strategy 'Stance'
X = 50
sp500['Stance'] = np.where(sp500['42-252'] > X, 1, 0)
sp500['Stance'] = np.where(sp500['42-252'] < X, -1, sp500['Stance'])
sp500['Stance'].value_counts()

#Create columns containing daily market log returns and strategy daily log returns
sp500['Market Returns'] = np.log(sp500['close'] / sp500['close'].shift(1))
sp500['Strategy'] = sp500['Market Returns'] * sp500['Stance'].shift(1)

#Set strategy starting equity to 1 and generate equity curve
sp500['Strategy Curve'] = sp500['Strategy'].cumsum() + 1

#Show chart of equity curve
sp500['Strategy Curve'].plot()

strat = pd.DataFrame([sp500['Strategy Equity'], sp500['Strategy']]).transpose()

#Create columns that signifies whether each days return was positive, negative or flat
strat['win'] = np.where(strat['Strategy']>0, 1, 0)
strat['loss'] = np.where(strat['Strategy']<0, 1, 0)
strat['scratch'] = np.where(strat['Strategy']==0, 1, 0)

strat['wincum'] = np.where(strat['Strategy']>0, 1, 0).cumsum()
strat['losscum'] = np.where(strat['Strategy']<0, 1, 0).cumsum()
strat['scratchcum'] = np.where(strat['Strategy']==0, 1, 0).cumsum()

# Create columns that holds a running sum of trading days
strat['days'] = strat['wincum'] + strat['losscum'] + strat['scratchcum']

# Create columns that shows the 252 days rolling sum of the winning/lossing/flat days
strat['rollwin'] = strat['win'].rolling(window=252).sum()
strat['rollloss'] = strat['loss'].rolling(window=252).sum()
strat['rollscratch'] = strat['scratch'].rolling(window=252).sum()

# Create columns with hit ratio and loss ratio data
strat['hitratio'] = strat['wincum'] / (strat['wincum'] + strat['losscum'])
strat['lossratio'] = 1 - strat['hitratio']

# Create column with rolling 12 month return
strat['roll12mret'] = strat['Strategt'].rolling(window=252).sum()

# Create column with average win, average loss and average daily return data
strat['averagewin'] = strat['Strategy'][strat['Strategy']>0].mean()
strat['averageloss'] = strat['Strategy'][strat['Strategy']<0].mean()
strat['averagedailyret'] = strat['Strategy'].mean()

strat['roll12mstdev'] = strat['Strategy'].rolling(window=252).std()
strat['roll12mannualisedvol'] = strat['roll12mstdev'] * sqrt(252)

strat['roll12mannualisedvol'].plot(grid=True,figsize=(8,5),title='Rolling 1 Year Annualised Volatility')
strat['rollhitratio'].plot(grid=True,figsize=(8,5),title='Rolling 1 Year Hit Ratio')
strat['roll12mret'].plot(grid=True,figsize=(8,5),title='Rolling 1 Year Return')
strat['Strategy'].plot(grid=True,figsize=(8,5),title='Daily Return')
strat['Strategy'].plot(kind='hist',figsize=(8,5),title='Daily Return Distribution',bins=100)

print('Skew:', round(strat['Strategy'].skew(),4))
print('Kurtosis:', round(strat['Strategy'].kurt(),4))


# Create a new DataFrame to hold our monthly data and populate it with the data from the daily returns column of our original DataFrame and sum it by month
stratm = pd.DataFrame(strat['Strategy'].resample('M').sum())

# Build the monthly data equity curve
stratm['Strategy Equity'] = strat['Strategy'].cumsum() + 1

# Add a column that holds the numerical monthly index
stratm['month'] = stratm.index.month
stratm.head()

# 1)Annualised return
days = (strat.index[-1] - strat.index[1]).days
cagr = (strat['Strategy Equity'][-1] / strat['Strategy Equity'][1])**(365.0/days) - 1
print('CAGR=',str(round(cagr,4)*100)+'%')

# 2)Last 12 months return
stratm['last12mret'] = stratm['Strategy'].rolling(window=12,center=False).sum()
last12mret = stratm['last12mret'][-1]
print('last 12 month return =', str(round(last12mret*100,4))+'%')

# 3)Volatility
voldaily = strat['Strategy'].std()*sqrt(252)
volmonthly = stratm['Strategy'].std()*sqrt(12)
print('Annualised volatility using daily data =', str(round(voldaily,4)*100)+'%')
print('Annualised volatility using monthly data =', str(round(volmonthly,4)*100)+'%')

# 4)Sharpe ratio
dailysharpe = cagr / voldaily
monthlysharpe = cagr / volmonthly
print("daily sharpe ratio =", round(dailysharpe,2))
print("daily sharpe ratio =", round(dailysharpe,2))

# 5)Maximum drawdown
# Create max drawdown function
def max_drawdown(X):
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x)/peak
        if dd > mdd:
            mdd = dd
    return mdd

mdd_daily = max_drawdown(strat['Strategy Curve'])
mdd_monthly = max_drawdown(stratm['Strategy Curve'])
print('max drawdown daily data =', str(round(mdd_daily,4)*100)+'%')
print('max drawdown monthly data =', str(round(mdd_monthly,4)*100)+'%')

# 6)Calmar ratio
calmar = cagr/mdd_daily
print('Calmar ratio =', round(calmar,2))

# 7)Best and worst month performance
bestmonth = max(stratm['Strategy'])
worstmonth = min(stratm['Strategy'])

# 8)Percentage of profitable months and non-profitable months
positive_months = len(stratm['Strategy'][stratm['Strategy']>0])
negative_months = len(stratm['Strategy'][stratm['Strategy']<0])
flat_months = len(stratm['Strategy'][stratm['Strategy']==0])
perc_positive_months = positive_months / (positive_months+negative_months+flat_months)
perc_negative_months = negative_months / (positive_months+negative_months+flat_months)
prof_unprof_months = positive_months / negative_months

# 9)Average month profit and loss
av_monthly_pos = (stratm['Strategy'][stratm['Strategy']>0]).mean()
av_monthly_neg = (stratm['Strategy'][stratm['Strategy']<0]).mean()
pos_neg_month = abs(av_monthly_pos/av_monthly_neg)

monthly_table = stratm['Strategy','month'].pivot_table(stratm[['Strategy', 'month']], index=stratm.index, columns='month', aggfunc=np.sum).resample('A')
monthly_table = monthly_table.aggregate('sum')

monthly_table.columns = monthly_table.columns.droplevel()
monthly_table.index = monthly_table.index.year
monthly_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def ma_strat(ticker, short_ma, long_ma):
    # Read in data from Yahoo Finance for the relevant ticker
    sp500 = data.DataReader(ticker, data_source='yahoo', start='2000-01-01')
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

results_pnl = np.zeros(len(short_ma), len(long_ma))
results_sharpe = np.zeros(len(short_ma), len(long_ma))

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

