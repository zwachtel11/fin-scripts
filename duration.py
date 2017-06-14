
PV = 0
weighted = 0
totalPV = 0

for i in range (1, 20):
    PV = PV + .04/(1.05**i)
    weighted = weighted + (PV)* i/float(2)
    totalPV = totalPV + PV


duration = weighted/totalPV

print duration
