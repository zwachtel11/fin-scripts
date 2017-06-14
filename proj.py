import numpy as np
import pandas as pd
import pandas_datareader as web   
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
from scipy.optimize import minimize
import matplotlib.dates as mdates

## NUMBER OF ASSETS
n_assets = 4
N = 10
M = 50
## NUMBER OF OBSERVATIONS
n_obs = 1000
n_portfolios = 500


# start and end date
start = datetime.datetime(2008, 1, 2)
end = datetime.datetime(2012, 1, 2)

def sharpe(w, p,C):
    mu = w * p.T
    sigma = np.sqrt(np.dot(np.dot(w,C), w))
    sharpe = mu/sigma
    return 1/sharpe


def solve_weights(returns):
    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    n = returns.shape[0]
    W = np.ones([n])/n
    b_ = [(-1.,1.) for i in range(n)]

    c_ = ({'type':'eq', 'fun': lambda W: sum(np.absolute(W))-1.})

    optimized = minimize(sharpe, W, (p,C),
            method='SLSQP', constraints=c_, bounds=b_)

    sharpe1 = sharpe(optimized.x, p,C)

    return optimized.x, sharpe1



def rand_weights(n):
    k = np.random.uniform(-1, 1, size=10)
    return k/sum(abs(k))

def random_portfolio(returns):

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    if sigma > 2:
        return random_portfolio(returns)

    sharpe = float(mu)/float(sigma)

    return sharpe, w


#return_vec = np.random.randn(n_assets, n_obs)

apple = web.DataReader("AAPL", "yahoo", start, end)
microsoft = web.DataReader("MSFT", "yahoo", start, end)
google = web.DataReader("GOOG", "yahoo", start, end)
amazon = web.DataReader("AMZN", "yahoo", start, end)
netflix = web.DataReader("NFLX", "yahoo", start, end)
priceline = web.DataReader("PCLN", "yahoo", start, end)
ibm = web.DataReader("IBM", "yahoo", start, end)
cisco = web.DataReader("CSCO", "yahoo", start, end)
intel = web.DataReader("INTC", "yahoo", start, end)
baidu = web.DataReader("BIDU", "yahoo", start, end)
yahoo = web.DataReader("YHOO", "yahoo", start, end)

# Below I create a DataFrame consisting of the adjusted closing price of these stocks, first by making a list of these objects and using the join method
stocks = pd.DataFrame({"AAPL": apple["Adj Close"],
                      "MSFT": microsoft["Adj Close"],
                        "AMZN": amazon["Adj Close"],
                        "NFLX": netflix["Adj Close"],
                       "PCLN": priceline["Adj Close"],
                       "IBM": ibm["Adj Close"],
                        "CSCO": cisco["Adj Close"],
                        "INTC": intel["Adj Close"],
                       "BIDU": baidu["Adj Close"],
                       "YHOO": yahoo["Adj Close"],})

stock_return = stocks.apply(lambda x: (((x.shift(-1) - x) / x)))

names = ["AAPL", "MSFT", "AMZN", "PCLN", "IBM", "CSCO", "INTC", "BIDU", "YHOO"]

mean = []
sigma = []
covar = []
maxS = 0
weight = []
maxSharpe = []
stocksd = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
z = M
portfolio = []
V = 1000000000
peak = -99999
MDD = 0
while (z < len(stock_return.values)):

    if (z % N == 0):
        maxS = 0

        '''

        for i in xrange(0,1000000):
            sharpe, w = random_portfolio(stock_return.values[z-M:z].T)
            if (sharpe > maxS):
                maxS = sharpe
                maxW = w
        '''
        maxW, maxS = solve_weights(stock_return.values[z-M:z].T)
        maxSS = 1/float(maxS)
        maxSharpe.append(maxSS)
        weight.append(maxW)

        ee = 0
        for j in stocks.columns:
            qw = (maxW[ee])*V
            stocksd[ee] = qw / stocks[j].iloc[z]
            ee = ee + 1

    jj = 0
    for j in stocks.columns:
        if (stocksd[jj] > 0):
            V = V + stocksd[jj]*(stocks[j].iloc[z] - stocks[j].iloc[z - 1])
        else:
            V = V - stocksd[jj]*(stocks[j].iloc[z] - stocks[j].iloc[z - 1])
        jj = jj + 1

    if (V > peak):
        peak = V

    DD = 100.0 * (peak - V) / peak

    if (DD > MDD):
        MDD = DD

    portfolio.append(V)
    z = z + 1

print maxSharpe

total = ((V - 1000000000) / float(1000000000))*100
print total
print MDD

start1 = datetime.datetime(2008, 3, 14)
spyder = web.DataReader("SPY", "yahoo", start1, end)
netflix = web.DataReader("NFLX", "yahoo", start1, end)

spy = pd.DataFrame({"SPY": spyder["Adj Close"],
                    "NFLX": netflix["Adj Close"],
                    "RETURNS": portfolio})

spy_return = spy.apply(lambda x: ( (x / x[0])*1000000000))

spy_return.plot(grid = True)
plt.title('Returns vs SPY vs Netflix')
plt.savefig('spyvsnet.png')
plt.show()

'''
plt.plot(portfolio)
#plt.plot(weight)
#plt.xticks( range(2), ('2008', '2009', '2010', '2011', '2012'))
plt.grid()
plt.title('Returns for N = 20 and M= 10')
plt.xlabel('Days')
plt.ylabel('Dollars')
plt.savefig('n20m10S.png')
plt.show()

#stocks.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
times = pd.date_range(start ="2008-02-01", end = "2012-02-01")#, periods=len(stock_return.values))
fig, ax = plt.subplots(1)
fig.autofmt_xdate()
plt.plot(times, portfolio)

xfmt = mdates.DateFormatter('%m-%d-%y')
ax.xaxis.set_major_formatter(xfmt)

plt.show()
'''
