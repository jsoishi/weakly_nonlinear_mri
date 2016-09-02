import time
import numpy as np
from mpi4py import MPI
from scipy import optimize as opt
CW = MPI.COMM_WORLD

from dedalus import public as de
from eigentools import Eigenproblem

import logging
logger = logging.getLogger(__name__)

R1 = 1
R2 = 3
Omega1 = 1
Omega2 = 0.12087
beta = 41.2
xi=0
#Pm = 1.6E-3#1.60000000e-05#Pm = 1.60000000e-06#1.60000000e-04
Pm = 4.0E-6
#Rm=3.34
Q=0.905

print("MRI parameters: Pm = {}, beta = {}, Omega2/Omega1 = {}, R1/R2 = {}".format(Pm, beta, Omega2/Omega1, R1/R2))

c1 = (Omega2*R2**2 - Omega1*R1**2)/(R2**2 - R1**2)
c2 = (R1**2*R2**2*(Omega1 - Omega2))/(R2**2 - R1**2)

gridnum = 100
r_basis = de.Chebyshev('r', gridnum, interval=(R1, R2))
domain = de.Domain([r_basis], np.complex128, comm=MPI.COMM_SELF)
print("running at gridnum", gridnum)

lv1 = de.EVP(domain, ['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],'iRm')

lv1.parameters['Pm'] = Pm
lv1.parameters['Q'] = Q 
lv1.parameters['beta'] = beta
lv1.parameters['c1'] = c1
lv1.parameters['c2'] = c2

lv1.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
lv1.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
lv1.substitutions['twooverbeta'] = '(2.0/beta)'

lv1.substitutions['psivisc'] = '(2*r**2*Q**2*psir - 2*r**3*Q**2*psirr + r**3*Q**4*psi + r**3*dr(psirrr) - 3*psir + 3*r*psirr - 2*r**2*psirrr)'
lv1.substitutions['uvisc'] = '(-r**3*Q**2*u + r**3*dr(ur) + r**2*ur - r*u)'
lv1.substitutions['Avisc'] = '(r*dr(Ar) - r*Q**2*A - Ar)' 
lv1.substitutions['Bvisc'] = '(-r**3*Q**2*B + r**3*dr(Br) + r**2*Br - r*B)'

lv1.add_equation("-r**2*2*ru0*1j*Q*u + r**3*twooverbeta*1j*Q**3*A + twooverbeta*r**2*1j*Q*Ar - twooverbeta*r**3*1j*Q*dr(Ar) - iRm*Pm*psivisc = 0") #+ twooverbeta*r**2*2*xi*1j*Q*B = 0") #corrected on whiteboard 5/6
lv1.add_equation("1j*Q*ru0*psi + 1j*Q*rrdu0*psi - 1j*Q*r**3*twooverbeta*B - iRm*Pm*uvisc = 0") 
lv1.add_equation("-r*1j*Q*psi - iRm*Avisc = 0")
lv1.add_equation("ru0*1j*Q*A - r**3*1j*Q*u - 1j*Q*rrdu0*A - iRm*Bvisc = 0") #- 2*xi*1j*Q*psi = 0") 

lv1.add_equation("dr(psi) - psir = 0")
lv1.add_equation("dr(psir) - psirr = 0")
lv1.add_equation("dr(psirr) - psirrr = 0")
lv1.add_equation("dr(u) - ur = 0")
lv1.add_equation("dr(A) - Ar = 0")
lv1.add_equation("dr(B) - Br = 0")

lv1.add_bc('left(u) = 0')
lv1.add_bc('right(u) = 0')
lv1.add_bc('left(psi) = 0')
lv1.add_bc('right(psi) = 0')
lv1.add_bc('left(psir) = 0')
lv1.add_bc('right(psir) = 0')
lv1.add_bc('left(A) = 0')
lv1.add_bc('right(A) = 0')
lv1.add_bc('left(B + r*Br) = 0')
lv1.add_bc('right(B + r*Br) = 0') # axial component of current = 0


# create an Eigenproblem object
EP = Eigenproblem(lv1)

# create a shim function to translate (x, y) to the parameters for the eigenvalue problem:
def shim(kz):
    gr = EP.growth_rate({"Q":kz})
    print(1./gr[0])
    return 1./gr[0]


sol = opt.minimize_scalar(shim, bracket=[0.80, 1.00],options={'xtol':1e-4})
print(repr(sol.x))

