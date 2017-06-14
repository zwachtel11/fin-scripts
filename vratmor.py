from math import pow

pv = 100000
n = 30
ir = .08

A = (ir*pv) / (1 - (1/pow(1.08,n)))

print "Payment per year orginal rate: %f" %A

fiveYears = (pv/30)*5
balance = pv - fiveYears

print "balence after five years: %f" %balance

nr = .09

B = (nr*pv) / (1 - (1/pow(1.09,25)))

print "new payment per year : %f" %B

newtotal = B * 25 + A*5
orgtotal = A * 30
timeToPay = (newtotal - orgtotal) / A
term =  timeToPay + 30

print "new total: %f" %newtotal
print "old total: %f" %orgtotal
print "new term: %f" %term
