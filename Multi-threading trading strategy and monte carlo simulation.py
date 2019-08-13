import numpy as np
import pandas as pd
import itertools
import time
from multiprocessing.pool import ThreadPool as Pool


# Function to calculate Sharpe Ratio
def annualised_sharpe(returns, N=252):
    return np.sqrt(N) * returns.mean() / returns.std()


def ma_strat(data, short_ma, long_ma):
    data['short_ma'] = np.round(data['Close'].rolling(window=short_ma).mean(), 2)
    data['long_ma'] = np.round(data['Close'].rolling(window=long_ma).mean(), 2)
    data['short_ma-long_ma'] = data['short_ma'] - data['long_ma']

    DIFF = 5
    data['Stance'] = np.where(data['short_ma-long_ma'] > DIFF, 1, 0)
    data['Stance'] = np.where(data['short_ma-long_ma'] < -DIFF, -1, data['Stance'])
    data['Stance'].value_counts()

    data['Market Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Strategy'] = data['Market Returns'] * data['Stance'].shift(1)
    data['Strategy Curve'] = data['Strategy'].cumsum()

    # Calculate Sharpe Ratio, try/except to escape case of division by zero
    try:
        sharpe = annualised_sharpe(data['Strategy'])
    except:
        sharpe = 0

    return data['Strategy'].cumsum(), sharpe, data['Strategy'].mean(), data['Strategy'].std()


def monte_carlo_strat(data, inputs, iters):
    days = 252
    perf, sharpe, mu, sigma = ma_strat(data, inputs[0], inputs[1])

    mc_results = []
    mc_results_final_val = []

    for j in range(iters):
        daily_returns = np.random.normal(mu, sigma, days) + 1
        price_list = [1]
        for x in daily_returns:
            price_list.append(price_list[-1] * x)

        mc_results.append(price_list)
        mc_results_final_val.append(price_list[-1])

    return (inputs, perf, sharpe, mu, sigma, mc_results, mc_results_final_val)


if __name__ == '__main__':
    data = pd.read_csv('F.csv', index_col='Date', parse_dates=True)

    short_mas = np.linspace(20,50,30,dtype=int)
    long_mas = np.linspace(100,200,30,dtype=int)

    # Generate a list of tuples containing all combinations of long and short window length possibilities
    mas_combined = list(itertools.product(short_mas, long_mas))

    iters = 2000
    results = []
    start_time = time.time()

    for inputs in mas_combined:
        res = monte_carlo_strat(data, inputs, iters)
        results.append(res)

    print('MP--- %s seconds for single---' % (time.time() - start_time))


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it,size)), ())

def monte_carlo_strat(data, inputs, iters):
    days = 252

    for input_slice in inputs:
        perf, sharpe, mu, sigma = ma_strat(data, input_slice[0], input_slice[1])

    mc_results = []
    mc_results_final_val = []

    for j in range(iters):
        daily_returns = np.random.normal(mu, sigma, days) + 1
        price_list = [1]
        for x in daily_returns:
            price_list.append(price_list[-1]*x)

        mc_results.append(price_list)
        mc_results_final_val.append(price_list[-1])

    return (inputs, perf, sharpe, mu, sigma, mc_results, mc_results_final_val)

def parallel_monte_carlo(data, inputs, iters):
    pool = Pool(5)
    future_res = [pool.apply_async(monte_carlo_strat, args=(data,inputs[i],iters)) for i in range(len(inputs))]
    samples = [f.get() for f in future_res]

    return samples


if __name__ == '__main__':
    data = pd.read_csv('F.csv', index_col='Date', parse_dates=True)

    short_mas = np.linspace(20,50,30,dtype=int)
    long_mas = np.linspace(100,200,30,dtype=int)

    # Generate a list of tuples containing all combinations of long and short window length possibilities
    mas_combined = list(itertools.product(short_mas, long_mas))
    # Use helper function to split the moving average tuples list into slices of length 180
    mas_combined_split = list(chunk(mas_combined, 180))

    iters = 2000
    results = []
    start_time = time.time()

    results = parallel_monte_carlo(data, mas_combined_split, iters)

    print('MP--- %s seconds for parallel---' % (time.time() - start_time))


