import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime
import matplotlib.pyplot as plt


# start and end date
start = datetime.datetime(2007, 1, 3)
end = datetime.datetime(2010, 1, 3)



apple = web.DataReader("AMZN", "yahoo", start, end)


stocks = pd.DataFrame({"AAPL": apple["Adj Close"],})

N = [20, 30, 45, 50, 75, 90]

for n in N:

    S = stocks.values
    V = 1000000
    Vl = []

    for i in range(n+1, len(S)-1):
        I = V * np.sign(S[i] - S[i-n])
        rho = (S[i+1] - S[i])/ S[i]
        V = V +  I* rho
        Vl.append(V)


        plt.plot(Vl)
        
plt.savefig('amazon.png')
plt.show()
