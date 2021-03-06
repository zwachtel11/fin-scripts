import numpy as np
import sympy as sy
from sympy.solvers import solve
from scipy.optimize import newton
import matplotlib.pyplot as plt

facev = 25000 
n = 21
cr = .0675
price1 = 25000*1.23958
price2 = 25000*1.28
C = (cr/2)*facev
AI = 393.75
f = AI/C

def bond_ytm(price):
 dt = range(1,21)
 ytm_func = lambda(x): sum([C/((1 + x/2)**(t-f)) for t in dt])  + facev/((1 + x/2)**(n-f))  - price - AI
 return newton(ytm_func, .038) 
 
def graphIt(price):
 x = np.arange(0.0, .2, 0.001)
 dt = range(1,21)
 ytm_func = sum([C/((1 + x/2)**(t-f)) for t in dt])  + facev/((1 + x/2)**(n-f))  - price - AI
 plt.plot(x, ytm_func)
 plt.grid(True)
 plt.savefig("test.png")
 plt.show()
 return
 

if __name__ == "__main__":
 YTM = bond_ytm(price1)
 print "Yield to Mature: %f" %YTM
 graphIt(price1)
 YTM1 = bond_ytm(price2)
 print "new Yield to Mature: %f" %YTM1

