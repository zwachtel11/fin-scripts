import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton
from scipy import stats

facev = 100000
crStart = .10
price = 150000
Ce = 0

def setC(t,p):
 global Ce
 if (t == 1):
  Ce = .10
 Ce = getC(t,Ce,p)
 return (Ce/2)*facev
 

def getC(t,C,p):
 if (t == 1 or t % 2 == 0 or C <= .01):
  return C

 if random.random() < p:
  return C + .02
 
 else:
  return C-.01

 
def graphIt(p):
 x = np.arange(0.0, .2, 0.001)
 dt = range(1,20)
 ytm_func = lambda(x): sum([setC(t,p)/((1 + x/2)**(t)) for t in dt]) + facev/((1 + x/2)**(20)) - price
 return newton(ytm_func, .10)

def graphIt2(x, p):
 return sum([setC(t, p)/((1 + x/2)**(t)) for t in range(0,20)]) + facev/((1 + x/2)**(20)) - price

pa = np.arange(0, 1, .01)
y = []
for i in range (0,100):
 ytm = []
 for v in range (0,10):
  z = np.arange(0.00, .2, .001)
  p = i/float(100)
  slope, intercept = np.polyfit(graphIt2(z,p), z, 1)
  ytm.append(intercept)
 y.append(sum(ytm)/len(ytm))
 del ytm[:]

z = np.arange(0.00, .2, .001)
ps = .66
for k in range (0, 20):
 plt.plot(z, graphIt2(z, ps))
 
#plt.plot(pa, y)
plt.xlabel('YTM')
plt.ylabel('Price')
plt.grid(True)
plt.savefig('YTMvsPrice.png')
plt.show()



