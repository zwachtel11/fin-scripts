import numpy as np
from numpy.linalg import inv
from sympy.solvers import solve
from sympy import Symbol

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

a = np.array([[.2239, .1547, .1825, 0.62, 1.000], [.1547, .2684, .2631, -1.36, 1.000], [.1825, .2631, .4035, -.36, 1.000], [.62, -1.36, -.36, 0.000, 0.000], [1.000, 1.000, 1.000, 0.000, 0.000]])

x = np.array([0, 0, 0, 1.5, 1.000])

inva = inv(a)

ans =  inva.dot(x)

print ans

af = np.array([[.2239, .1547, .1825], [.1547, .2684, .2631], [.1825, .2631, .4035]])

invaf = inv(af)

rf = np.array([.1,.1,.1])
r = np.array([.62, -1.36, -.36])

one_fund = invaf.dot(rf - r)

normalize = 1/float(one_fund[0] + one_fund[1] + one_fund[2])

w = normalize * one_fund

mean = 0
for j in range(0, 2):
    mean = mean + w[j]*r[j]

alp = Symbol('alp')
answ = solve(.1*alp + (1-alp)*mean - 1.5, alp) 

print answ


