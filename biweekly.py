from math import pow

pv = 100000
n = 30*12
r= .1 / 12

A = (r*pv) / (1 - (1 / pow(1+r,n)))
totalI = A*n - pv

nb = 30*26
rb = .1/26

Ab =  (rb*pv) / (1 - (1 / pow(1+rb,nb)))
totalIb = Ab*n - pv

print "Monthly Payment: %f Total Interest: %f" %(A, totalI)
print "BiWeekly Payment: %f Total Interest: %f" %(Ab, totalIb)


