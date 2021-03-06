import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

a = np.array([[2.250, -1.000, -1.500, 0.500, 1.000], [-1.000, .889, -.6667, -.3333, 1.000], [-1.500, -.666, 5.000, 0.000, 1.000], [.5000, -.333, 0.000, 0.000, 0.000], [1.000, 1.000, 1.000, 0.000, 0.000]])

inva = inv(a)
ans = []
for i in range (0, 100):
    alp = i/float(100)
    x = np.array([0, 0, 0, (.25 -.1*alp), (1.000 - alp)])
    ans.append(inva.dot(x)[3])


s = np.arange(0.00, 1.00, 0.01)

plt.plot(s, ans)
plt.show()

