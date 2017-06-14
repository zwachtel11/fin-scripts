import numpy as np

finalpricel = []
finalbidl = []

for j in xrange(0,10000):

    ASK = np.random.uniform(100,200,200)
    BUY = np.random.uniform(100,200,1) 
    sorted = np.sort(ASK)

    count = 0
    min = 10000000
    for i in sorted:
        if BUY > i:
            count += 1
        else:
            if (i < min):
                min = i

    finalbid = 10050 - count*100
    finalprice = (min + BUY)/2
    if (finalprice < 200):
        finalpricel.append(finalprice)
        finalbidl.append(finalbid)

print np.mean(finalpricel)
print np.mean(finalbidl)
