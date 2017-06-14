import numpy as np
from numpy.linalg import inv

af = np.array([[.3, .01], [.01, .40]])
rf = np.array([.05, .05])
r = np.array([.1, .2])

invaf = inv(af)
             
one_fund1 = invaf.dot(rf - r)

new = 1/float(one_fund1[0] + one_fund1[1])

newnew = new * one_fund1

print newnew

