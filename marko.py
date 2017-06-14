import numpy as np
import pandas as pd
import pandas_datareader as web   # Package and modules for importing data; this code may change depending on pandas version
import datetime
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

## NUMBER OF ASSETS
n_assets = 4

## NUMBER OF OBSERVATIONS
n_obs = 1000
n_portfolios = 500

# Turn off progress printing 
solvers.options['show_progress'] = False

# start and end date
start = datetime.datetime(2015,1,1)
end = datetime.date.today()


def rand_weights(n):
    k = np.random.rand(n)
    return k/sum(k)

def random_portfolio(returns):

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks






#return_vec = np.random.randn(n_assets, n_obs)
 
apple = web.DataReader("AAPL", "yahoo", start, end)
microsoft = web.DataReader("MSFT", "yahoo", start, end)
google = web.DataReader("GOOG", "yahoo", start, end)
twitter = web.DataReader("TWTR", "yahoo", start, end)
facebook = web.DataReader("FB", "yahoo", start, end)
 
# Below I create a DataFrame consisting of the adjusted closing price of these stocks, first by making a list of these objects and using the join method
stocks = pd.DataFrame({"AAPL": apple["Adj Close"],
                      "MSFT": microsoft["Adj Close"],
                       "GOOG": google["Adj Close"],
                        "TWTR": twitter["Adj Close"],
                        "FB": facebook["Adj Close"],})

stock_return = stocks.apply(lambda x: (x / x[0] - 1))

means, stds = np.column_stack([
    random_portfolio(stock_return.values.T) 
    for _ in xrange(n_portfolios)
])

weights, returns, risks = optimal_portfolio(stock_return.values.T)

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.ylim(-.5, .5)
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
plt.show()
#print np.random.randn(n_assets, n_obs)
#print stock_return.values.T
