#!/usr/bin/python
from math import pow

start = 1000000
rentrate= .05
ror = 1.075
pv = 0
for num in range (1, 11):
    if num == 4 or num == 8:
        pv = pv + (start - 500000)/pow(ror, num)
    elif num == 10:
        pv = pv + (start + 15000000)/pow(ror, num)
    else:
        pv = pv + start/pow(ror, num)
    start = start*rentrate + start

print pv

    
