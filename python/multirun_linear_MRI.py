import numpy as np
import matplotlib.pyplot as plt
from dedalus2.tools.config import config
config['logging']['stdout_level'] = 'critical'
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue
import pylab

import sys

Q = float(sys.argv[2])
Rm = float(sys.argv[1])
name = sys.argv[0]

#Q = 0.75
#Rm = 4.8775

#print("%s: Q = %10.5f; Rm = %10.5f" % (name, Q, Rm))



# Solve eigenvalues of linearized MRI set up. Will minimize to find k_z = Q and Rm = Rm_crit, where s ~ 0. 
# This takes as inputs guesses for the critical values Q and Rm.


lv1 = ParsedProblem(['x'],
                      field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                      param_names=['Q', 'iR', 'iRm', 'q', 'B0', 'beta', 'Co'])
                  
x_basis = Chebyshev(32)
domain = Domain([x_basis])

#Rm is an input parameter
iRm = 1./Rm

#Parameter values from Umurhan+:
Pm = 0.0001 #0.001 #Pm = Rm/R
R = Rm/Pm
iR = 1./R
q = 3/2.
Co = 0.08
B0 = 1.#np.sqrt(Co)
beta = 2./Co

#print('Pm', Pm, 'Re', R, 'beta', beta)

#multiply by -1j and add dt's:
#lv1.add_equation("-dt(psixx) - Q**2*dt(psi) - iR*(dx(psixxx) + 2*psixx*Q**2 + Q**4*psi) + 2*1j*Q*u + B0*ifourpi*1j*Q*(-dx(Ax) - Q**2*A) = 0")
#lv1.add_equation("1j*Q*(2 - q)*psi - iR*(-dx(ux) - Q**2*u) - B0*ifourpi*1j*Q*B + dt(u) = 0")
#lv1.add_equation("-1j*Q*B0*psi - iR*(-dx(Ax) - Q**2*A) + dt(A) = 0")
#lv1.add_equation("-1j*Q*B0*u + 1j*Q*q*A - iRm*(-dx(Bx) - Q**2*B) + dt(B) = 0")

#In terms of beta
#lv1.add_equation("dt(psixx) - Q**2*dt(psi) - iR*dx(psixxx) + 2*iR*Q**2*dx(psix) - iR*Q**4*psi - 2*1j*Q*u - (2/beta)*B0*1j*Q*dx(Ax) + (2/beta)*B0*Q**3*1j*A = 0")
#lv1.add_equation("dt(u) - iR*dx(ux) + iR*Q**2*u + (2-q)*1j*Q*psi - (2/beta)*B0*1j*Q*B = 0")
#lv1.add_equation("dt(A) - iRm*dx(Ax) + iRm*Q**2*A - B0*1j*Q*psi = 0")
#lv1.add_equation("dt(B) - iRm*dx(Bx) + iRm*Q**2*B - B0*1j*Q*u + q*1j*Q*A = 0")

#In terms of Co ....multiplied dt terms by -1j
lv1.add_equation("-1j*dt(psixx) - -1j*Q**2*dt(psi) - iR*dx(psixxx) + 2*iR*Q**2*psixx - iR*Q**4*psi - 2*1j*Q*u - Co*B0*1j*Q*dx(Ax) + Co*B0*Q**3*1j*A = 0")
lv1.add_equation("-1j*dt(u) - iR*dx(ux) + iR*Q**2*u + (2-q)*1j*Q*psi - Co*B0*1j*Q*B = 0") 
lv1.add_equation("-1j*dt(A) - iRm*dx(Ax) + iRm*Q**2*A - B0*1j*Q*psi = 0") 
lv1.add_equation("-1j*dt(B) - iRm*dx(Bx) + iRm*Q**2*B - B0*1j*Q*u + q*1j*Q*A = 0")

lv1.add_equation("dx(psi) - psix = 0")
lv1.add_equation("dx(psix) - psixx = 0")
lv1.add_equation("dx(psixx) - psixxx = 0")
lv1.add_equation("dx(u) - ux = 0")
lv1.add_equation("dx(A) - Ax = 0")
lv1.add_equation("dx(B) - Bx = 0")

#Boundary conditions
lv1.add_left_bc("u = 0")
lv1.add_right_bc("u = 0")
lv1.add_left_bc("psi = 0")
lv1.add_right_bc("psi = 0")
lv1.add_left_bc("A = 0")
lv1.add_right_bc("A = 0")
lv1.add_left_bc("psix = 0")
lv1.add_right_bc("psix = 0")
lv1.add_left_bc("Bx = 0")
lv1.add_right_bc("Bx = 0")

#Parameters
lv1.parameters['Q'] = Q
lv1.parameters['iR'] = iR
lv1.parameters['iRm'] = iRm
lv1.parameters['q'] = q
lv1.parameters['B0'] = B0
lv1.parameters['Co'] = Co
lv1.parameters['beta'] = beta

lv1.expand(domain)
LEV = LinearEigenvalue(lv1,domain)
LEV.solve(LEV.pencils[0])

#Find the eigenvalue that is closest to zero.
evals = LEV.eigenvalues
indx = np.arange(len(evals))
e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]

val = evals[e0]

if len(val) < 1:
    val = [99]

print(val[0])
