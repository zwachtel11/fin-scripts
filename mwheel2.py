import numpy as np
from numpy.linalg import inv

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

a = np.array([[2.250, -1.000, -1.500, 0.000, 0.500, 1.000], [-1.000, .889, -.6667, 0.000, -.3333, 1.000], [-1.500, -.666, 5.000, 0.000, 0.000, 1.000], [0,0,0,0,.1,1], [.5000, -.333, 0.000, .100, 0.000, 0.000], [1.000, 1.000, 1.000, 1.000, 0.000, 0.000]])

x = np.array([0, 0, 0, 0, .25, 1.000])


inva = inv(a)

ans =  inva.dot(x)


print ans


