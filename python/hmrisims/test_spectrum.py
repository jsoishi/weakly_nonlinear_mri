from find_hmri_crit_sparse import hmri_eigenproblem

nr = 64

## Rainer's test params
R1 = 1
R2 = 2
Omega1 = 1
Omega2 = 0.27
Pm=0.0001
xi = 4.
#Co = 1.1868223458100633

Ha=16.25082255
Re=1491.8 #1491.70343814
k_opt=2.32594957

Rm = Re*Pm
#beta = Re*Rm/(Omega2*Ha**2)
beta = (2*Re*Rm)/Ha**2
print("beta = {}".format(beta))
insulate = True

EP = hmri_eigenproblem(R1, R2, Omega1, Omega2, beta, xi, Pm, Rm, k_opt, insulate, nr, sparse=True)

EP.solve()
EP.reject_spurious()
EP.spectrum(spectype='good')
