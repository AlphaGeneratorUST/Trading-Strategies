import pandas_datareader.data as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stock = ['AAPL', 'AMZN', 'MSFT', 'YHOO']
#stock = ['AMZN']
data = web.DataReader(stock,data_source='yahoo',start='2010/01/01')['Adj Close']

print(data.head(10))

data.sort_index(inplace=True)
returns = data.pct_change()
mean_return = returns.mean()
return_stdev = returns.std()

annualised_return = round(mean_return * 252,2)
annualised_stdev = round(return_stdev * np.sqrt(252),2)

#print('The annualised mean return of stock {} is {}, and the annulised volatility is {}'.format(stock[0], annualised_return, annualised_stdev))

mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

weights = np.asarray([0.5, 0.2, 0.2, 0.1])

portfolio_return = round(np.sum(mean_daily_returns*weights)*252, 2)
portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))*np.sqrt(252), 2)

print('Portfolio expected annualised return is {} and volatility is {}').format(portfolio_return, portfolio_std_dev)

num_portfolios = 25000

results = np.zeros(3,num_portfolios)

for i in range(num_portfolios):
    weights = np.random.random(4)
    weights /= np.sum(weights)

    portfolio_return = np.sum(mean_daily_returns*weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights))) * np.sqrt(252)

    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    results[2,i] = results[0,i]/results[1,i]

results_frame = pd.DataFrame(results.T, columns=['ret','stdev','sharpe'])

plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlGn')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar()
