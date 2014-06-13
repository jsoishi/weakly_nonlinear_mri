import numpy as np
import matplotlib.pyplot as plt
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue
import pylab
lv1 = ParsedProblem(['x'],
                      field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                      param_names=['Q', 'iR', 'iRm', 'q', 'Co'])
                      
x_basis = Chebyshev(64)#(256)
domain = Domain([x_basis])#,grid_dtype=np.float64)

#Solve equations for fixed z eigenvalue: V+ = V+ e^(iQz)
#Parameter values from Umurhan+:
Pm = 0.001 #Pm = Rm/R
q = 3/2.
Co = 0.08

#Parameter values found by solving the linear MRI to find most unstable mode
#Rm = 4.8775
#Rm = 4.89
Rm = 4.877
iRm = 1./Rm
Q = 0.75

R = Rm/Pm
iR = 1./R

#Define higher order derivatives of x
#These are the equations....
# iR*dx(psixxx) - (2-q)*1j*Q*u - 1j*Q*A - Q*Q*iR*2*dx(psix) + Q**4*iR*psi = 0
# iR*dx(ux) - 2*1j*Q*psi - 1j*Q*B - iR*Q**2*u = 0
# iRm*dx(Ax) - ifourpi*1j*Q*dx(psix) + q*1j*Q*B - Q*Q*iRm*A + 1j*Q**3*ifourpi*psi = 0
# iRm*dx(Bx) - ifourpi*1j*Q*u - Q*Q*iRm*B = 0

#multiply by -1j and add dt's:
#lv1.add_equation("-1j*iR*dx(psixxx) - -1j*(2-q)*1j*Q*u - -1j*1j*Q*A - -1j*Q*Q*iR*2*psixx + -1j*Q**4*iR*psi + dt(psi) = 0")
#lv1.add_equation("-1j*iR*dx(ux) - -1j*2*1j*Q*psi - -1j*1j*Q*B - -1j*iR*Q**2*u + dt(u) = 0")
#lv1.add_equation("-1j*iRm*dx(Ax) - -1j*Co*1j*Q*psixx + -1j*q*1j*Q*B - -1j*Q*Q*iRm*A + -1j*1j*Q**3*Co*psi + dt(A) = 0")
#lv1.add_equation("-1j*iRm*dx(Bx) - -1j*Co*1j*Q*u - -1j*Q*Q*iRm*B + dt(B) = 0")

#Proper equations? (pmmp)
#lv1.add_equation("1j*dt(psi) + -1j*Q*A - 1j*Q*(2-q)*u + iR*Q**4*psi - 2*iR*Q**2*psixx + iR*dx(psixxx) = 0")
#lv1.add_equation("1j*dt(u) + -1j*Q*B - 1j*2*Q*psi - iR*Q**2*u + iR*dx(ux) = 0")
#lv1.add_equation("1j*dt(A) + -iRm*Q**2*A + iRm*dx(Ax) + 1j*Q*q*B + 1j*Co*Q**3*psi - 1j*Co*Q*psixx = 0")
#lv1.add_equation("1j*dt(B) + -iRm*Q**2*B + iRm*dx(Bx) - 1j*Co*Q*u = 0")

#Equations without the sign change on the dz terms:
lv1.add_equation("1j*dt(psi) + 1j*Q*A + 1j*Q*(2-q)*u + iR*Q**4*psi - 2*iR*Q**2*psixx + iR*dx(psixxx) = 0")
lv1.add_equation("1j*dt(u) + 1j*Q*B + 2*1j*Q*psi - iR*Q**2*u + iR*dx(ux) = 0")
lv1.add_equation("1j*dt(A) + -iRm*Q**2*A + iRm*dx(Ax) - 1j*Q*q*B - 1j*Co*Q**3*psi + 1j*Co*Q*psixx = 0")
lv1.add_equation("1j*dt(B) + -iRm*Q**2*B + iRm*dx(Bx) + 1j*Co*Q*u = 0")

#without iQ's
#lv1.add_equation("1j*dt(psi) + A + iR*dx(psixxx) + iR*2*psixx + iR*psi + (2-q)*u = 0")
#lv1.add_equation("1j*dt(u) + B + 2*psi + iR*dx(ux) + iR*u = 0")
#lv1.add_equation("1j*dt(A) + iRm*dx(Ax) + iRm*A - q*B + Co*dx(psix) + Co*psi = 0")
#lv1.add_equation("1j*dt(B) + iRm*dx(Bx) + iRm*B + Co*u = 0")

#With U = [0, 0, 0, B]^T
#lv1.add_equation("1j*dt(psi) + 1j*Q*A + 1j*Q*(2-q)*u + iR*Q**4*psi - iR*2*Q**2*psixx + iR*dx(psixxx) = 0")
#lv1.add_equation("1j*dt(u) + 1j*Q*B + 1j*2*Q*psi - iR*Q**2*u + iR*dx(ux) + B = 0")
#lv1.add_equation("1j*dt(A) + -iRm*Q**2*A + iRm*dx(Ax) - 1j*Q*q*B - 1j*Co*Q**3*psi + 1j*Co*Q*psixx - q*B = 0")
#lv1.add_equation("1j*dt(B) + -iRm*Q**2*B + iRm*dx(Bx) + 1j*Co*Q*u + iRm*dx(Bx) + iRm*B = 0")

#NON-ADJOINT LV=0
#lv1.add_equation("1j*dt(psi) - 1j*Co*Q**3*A + 1j*Co*Q*dx(Ax) + 2*1j*Q*u + iR*Q**4*psi - iR*2*Q**2*psixx + iR*dx(psixxx) = 0")
#lv1.add_equation("1j*dt(u) + 1j*B*Co*Q + 1j*Q*(2 - q)*psi - iR*Q**2*u + iR*dx(ux) = 0")
#lv1.add_equation("1j*dt(A) - iRm*Q**2*A + iRm*dx(Ax) + 1j*Q*psi = 0")
#lv1.add_equation("1j*dt(B) - 1j*Q*q*A -iRm*Q**2*B + iRm*dx(Bx) + 1j*Q*u = 0")

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
lv1.parameters['Co'] = Co

lv1.expand(domain)
LEV = LinearEigenvalue(lv1,domain)
LEV.solve(LEV.pencils[0])

#Find the eigenvalue that is closest to zero. This should be the adjoint homogenous solution.
evals = LEV.eigenvalues
indx = np.arange(len(evals))
e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
print(evals[e0])

#Plot
x = domain.grid(0)
LEV.set_state(e0[0])

#L = LEV.eigenvalue_pencil.L.todense()
#b = np.zeros_like(LEV.eigenvectors[0])

#lsolve = np.linalg.solve(L, b)
#LEV.set_state(lsolve)

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot(x, LEV.state['psi']['g'].imag, color="red")
#ax1.plot(x, LEV.state['psi']['g'].real, color="black")

ax1.set_title(r"Im($\psi^\dagger$)")
ax1 = fig.add_subplot(222)
#ax1.plot(x, LEV.state['u']['g'].imag, color="black")
ax1.plot(x, LEV.state['u']['g'].real, color="red")
ax1.set_title("Re($u^\dagger$)")
ax1 = fig.add_subplot(223)
#ax1.plot(x, LEV.state['A']['g'].imag, color="black")
ax1.plot(x, LEV.state['A']['g'].real, color="red")
ax1.set_title("Re($A^\dagger$)")
ax1 = fig.add_subplot(224)
ax1.plot(x, LEV.state['B']['g'].imag, color="red")
#ax1.plot(x, LEV.state['B']['g'].real, color="black")
ax1.set_title("Im($B^\dagger$)")
fig.savefig("ah1.png")


"""
for i in range(len(evals)):
    LEV.set_state(i)

    ax1 = fig.add_subplot(221)
    ax1.plot(x, LEV.state['psi']['g'].imag)
    ax1.set_title(r"$\psi$")
    ax1 = fig.add_subplot(222)
    ax1.plot(x, LEV.state['u']['g'].real)
    ax1.set_title("u")
    ax1 = fig.add_subplot(223)
    ax1.plot(x, LEV.state['A']['g'].real)
    ax1.set_title("A")
    ax1 = fig.add_subplot(224)
    ax1.plot(x, LEV.state['B']['g'].imag)
    ax1.set_title("B")

    pylab.savefig('AH_'+ '%03d' % i +'.png')

    fig.clf()
"""