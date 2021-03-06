#!/usr/bin/python
from math import pow

ror = 1.1
pvx = 0.00
pvy = 0.00
x = [-1, 3, 2]
y = [-2, 4]
for index in range(len(x)):
    pvx = pvx + x[index]/pow(ror, index)
for i in range(len(y)):
    pvy = pvy + y[i]/pow(ror,i)

print "PV for x :%f" % pvx
print "PV for y :%f" % pvy

pvxcont = (1/(1 - (1/pow(ror, len(x)))))*pvx
pvycont = (1/(1 - (1/pow(ror, len(y)))))*pvy

print "PV for x cont :%f" % pvxcont
print "PV for y cont:%f" % pvycont

