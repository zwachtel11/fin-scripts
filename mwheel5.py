import numpy as np
from numpy.linalg import inv
from sympy.solvers import solve
from sympy import Symbol

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

a = np.array([[2.250, -1.000, -1.500, 0.500, 1.000], [-1.000, .889, -.6667, -.3333, 1.000], [-1.500, -.666, 5.000, 0.000, 1.000], [.5000, -.333, 0.000, 0.000, 0.000], [1.000, 1.000, 1.000, 0.000, 0.000]])

x = np.array([0, 0, 0, .25,1.000])


inva = inv(a)

ans =  inva.dot(x)

af = np.array([[2.250, -1.000, -1.500], [-1.000, .889, -.6667], [-1.500, -.666, 5.000]])

invaf = inv(af)

rf = np.array([.1,.1,.1])
r = np.array([.5, -.3333, 0.00])

one_fund = invaf.dot(rf - r)

normalize = 1/float(one_fund[0] + one_fund[1] + one_fund[2])

w = normalize * one_fund

mean = 0
for j in range(0, 2):
    mean = mean + w[j]*abs(r[j])

alp = Symbol('alp')
answ = solve(.1*alp + (1-alp)*mean - .25, alp) 

print answ
