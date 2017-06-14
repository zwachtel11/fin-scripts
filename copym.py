#!/usr/bin/python
from math import pow
from numpy import array, insert, irr, reshape

ror = 1.1
a = array([-6000, -8000, -8000, -8000, -8000])
b = array([-30000, -2000, -2000, -2000, -2000])
diff = b - a

diff = insert(diff, 5 , 10000)

print irr(diff)



