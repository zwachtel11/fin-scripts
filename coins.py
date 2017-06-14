import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

trials = 100000


def rand_weights(n):
    w = np.random.rand(n)
    return w / sum(w)

def get_sharpe(w):
    mean = [0,0,0,0,0,1]
    sd = [0,0,0,0,0, 0]
    for i in range(0,4):
        p = (.3 + (i+1)/10)
        mean[i] = p*(1 - (i+1)/float(10))
        sd[i] = p*(1-p)
    mean[5] = .1
    
    top = 0
    bottom = 0
    for i in range(0,5):
        top = top + mean[i]*w[i]
        bottom = bottom + (sd[i]*w[i]**2)
    S = top / (bottom**(.5))*2
    return S

sharpe_arr = []
S = 0
best = 0
best_weights = rand_weights(6)
for k in range(0, trials):
    weights = rand_weights(6)
    S = get_sharpe(weights)
    sharpe_arr.append(S)
    if (S > best):
        best = S
        best_weights = weights

x = np.arange(0, 100000, 1)
plt.plot(x, sharpe_arr)
plt.xlabel('trials')
plt.ylabel('sharpe ratios')
plt.savefig('coins.png')
plt.show()
print best
print weights










