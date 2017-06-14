import numpy as np
from numpy.linalg import inv

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

a = np.array([[.2239, .1547, .1825, 0.62, 1.000], [.1547, .2684, .2631, -1.36, 1.000], [.1825, .2631, .4035, -.36, 1.000], [.62, -1.36, -.36, 0.000, 0.000], [1.000, 1.000, 1.000, 0.000, 0.000]])

x = np.array([0, 0, 0, 1.5, 1.000])


inva = inv(a)

ans =  inva.dot(x)


print ans


