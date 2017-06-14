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

 if random.random() > p:
  return C + .02
 
 else:
  return C-.01

def graphIt2(x, p):
 return sum([setC(t, p)/((1 + x/2)**(t)) for t in range(0,20)]) + facev/((1 + x/2)**(20)) - price

def duration(ytm, p):
 PV = 0
 weighted = 0
 totalPV = 0
 for t in range (1, 20):
    PV = PV + setC(t, p)/((1 + ytm/2)**t)
    weighted = weighted + (PV)* t/float(2)
    totalPV = totalPV + PV
    
 return weighted/totalPV


z = np.arange(0.00, .2, .001)
ps = .66
durs = []
for k in range (0, 10000):
 slope, intercept = np.polyfit(graphIt2(z, ps), z, 1)
 durs.append(duration(intercept, ps))

mean = sum(durs)/ len(durs)

plt.hist(durs)
plt.xlabel('Durations')
plt.ylabel('Percents of Result, 10000 Runs')
print mean
#plt.plot(pa, y)
plt.grid(True)
plt.savefig('exob2.png')
plt.show()

