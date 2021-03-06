import matplotlib.pyplot as plt
delt = .005
rA = 1.5 - 1
rB = 1.333 - 1
rC = 5 - 1
sA2 = 2.25
sB2 = .8977
sC2 = 5
sAB = -1	
sAC = -1.5
sBC = -.67
times = 200

rP = []
sigP = []

for wA in range (0,200):
    for wB in range(0, 200-wA):
        wC = 1 - wA/float(200) - wB/float(200)
        rP.append((wA/float(200))*rA + (wB/float(200))*rB + wC*rC)
        sigP.append((((wA/float(200))**2)*sA2 + ((wB/float(200))**2)*sB2 + (wC**2)*sC2 + 2*(wA/float(200))*(wB/float(200))*sAB + 2*(wA/float(200))*wC*sAC + 2*(wB/float(200))*wC*sBC)**(.5))
    
plt.plot(sigP, rP)
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.savefig('wheel.png')
plt.show()
