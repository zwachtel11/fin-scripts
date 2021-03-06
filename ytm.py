import numpy as np
import sympy as sy
from sympy.solvers import solve
from scipy.optimize import fsolve
from scipy.optimize import newton


facev = 100000.00 
dur = 30.00
cr = .08
price1 = 106000.00
C = (cr/2)*facev

def bond_ytm(price):
 dt = range(1,60)
 ytm_func = lambda(x): sum([C/((1 + x/2)**(t)) for t in dt])  + facev/((1 + x/2)**60)  - price
 return newton(ytm_func, .05)

def getPrice(YTM):
 dt = range(1,60)
 price = sum([C/((1 + YTM/2)**(t)) for t in dt])  + facev/((1 + YTM/2)**60)
 return price
 

if __name__ == "__main__":
 YTM = bond_ytm(price1)
 print "Yield to Mature: %f" %YTM
 newPrice = getPrice(YTM + .02)
 print "new updated price: %f" %newPrice
 
