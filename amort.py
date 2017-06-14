from math import pow

pv = 25000
n = 7*12
r= .07 / 12

A = (r*pv) / (1 - (1 / pow(1+r,n)))


print "Monthly Payment: %f" %A
