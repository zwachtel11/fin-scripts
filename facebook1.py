import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime
import matplotlib.pyplot as plt
import math


# start and end date
start = datetime.datetime(2017, 3, 28)
end = datetime.datetime(2017, 4, 26)



facebook = web.DataReader("FB", "yahoo", start, end)


stocks = pd.DataFrame({"FB": facebook["Adj Close"],})


stocks_return = stocks.apply(lambda x: ( np.log((x.shift(-1)/ x))))

returns = stocks_return.values

mu = np.mean(returns[0: -1])
sigma = np.std(returns[0:-1])
print mu
print sigma
model = []
returnsN = []
firstday = math.exp((mu + sigma*np.random.normal(mu, sigma, 1))) * facebook["Adj Close"][-1]
firstdayR = (firstday - facebook["Adj Close"][-1]) /facebook["Adj Close"][-1]

model.append(firstday)
returnsN.append(firstdayR)

for i in range(1,252):
    curr = (1 + mu + sigma*np.random.normal(mu, sigma, 1))*model[-1]
    #currR = (curr - model[-1] )/ model[-1]
    model.append(curr)
   # returnsN.append(currR)

plt.plot(model)
plt.savefig('facebook-l-252.png')
plt.show()
