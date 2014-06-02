import numpy as np
import matplotlib.pyplot as plt
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue
lv1 = ParsedProblem(['x'],
                      field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'psixxxx', 'ux', 'Ax', 'Bx'],
                      param_names=['Q', 'iR', 'iRm', 'q', 'ifourpi'])
                      
x_basis = Chebyshev(128)#(256)
domain = Domain([x_basis])#,grid_dtype=np.float64)

#Solve equations for fixed z eigenvalue: V+ = V+ e^(iQz)
#Parameter values from Umurhan+:
Rm = 4.9
iRm = 1./Rm
Pm = 0.001 #Pm = Rm/R
R = Rm/Pm
iR = 1./R
Q = 0.75
q = 3/2.
ifourpi = 1./(4*np.pi)

#Define higher order derivatives of x
#These are the equations....
# iR*dx(psixxx) - (2-q)*1j*Q*u - 1j*Q*A - Q*Q*iR*2*dx(psix) + Q**4*iR*psi = 0
# iR*dx(ux) - 2*1j*Q*psi - 1j*Q*B - iR*Q**2*u = 0
# iRm*dx(Ax) - ifourpi*1j*Q*dx(psix) + q*1j*Q*B - Q*Q*iRm*A + 1j*Q**3*ifourpi*psi = 0
# iRm*dx(Bx) - ifourpi*1j*Q*u - Q*Q*iRm*B = 0

#multiply by -1j and add dt's:
lv1.add_equation("-1j*iR*dx(psixxx) - -1j*(2-q)*1j*Q*u - -1j*1j*Q*A - -1j*Q*Q*iR*2*dx(psix) + -1j*Q**4*iR*psi + dt(psi) = 0")
lv1.add_equation("-1j*iR*dx(ux) - -1j*2*1j*Q*psi - -1j*1j*Q*B - -1j*iR*Q**2*u + dt(u) = 0")
lv1.add_equation("-1j*iRm*dx(Ax) - -1j*ifourpi*1j*Q*dx(psix) + -1j*q*1j*Q*B - -1j*Q*Q*iRm*A + -1j*1j*Q**3*ifourpi*psi + dt(A) = 0")
lv1.add_equation("-1j*iRm*dx(Bx) - -1j*ifourpi*1j*Q*u - -1j*Q*Q*iRm*B + dt(B) = 0")

lv1.add_equation("dx(psi) - psix = 0")
lv1.add_equation("dx(psix) - psixx = 0")
lv1.add_equation("dx(psixx) - psixxx = 0")
lv1.add_equation("dx(psixxx) - psixxxx = 0")
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
lv1.parameters['ifourpi'] = ifourpi

lv1.expand(domain)
LEV = LinearEigenvalue(lv1,domain)
LEV.solve(LEV.pencils[0])

#Find the eigenvalue that is closest to zero. This should be the adjoint homogenous solution.
evals = LEV.eigenvalues
indx = np.arange(len(evals))
e0 = indx[evals == np.min(evals[evals >= 0])]
print(e0)

#Plot
x = domain.grid(0)
LEV.set_state(e0[0])

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot(x, LEV.state['psi']['g'])
ax1.set_title(r"$\psi$")
ax1 = fig.add_subplot(222)
ax1.plot(x, LEV.state['u']['g'])
ax1.set_title("u")
ax1 = fig.add_subplot(223)
ax1.plot(x, LEV.state['A']['g'])
ax1.set_title("A")
ax1 = fig.add_subplot(224)
ax1.plot(x, LEV.state['B']['g'])
ax1.set_title("B")
