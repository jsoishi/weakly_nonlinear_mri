import numpy as np
import matplotlib.pyplot as plt
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue
from scipy.linalg import eig, norm
import pylab
import copy

def dothis():

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

#Correct equations
lv1.add_equation("-1j*Q**2*dt(psi) + 1j*dt(psixx) + 1j*Q*A + 1j*(q - 2)*Q*u + iR*Q**4*psi - 2*iR*Q**2*psixx + iR*dx(psixxx) = 0")
lv1.add_equation("1j*dt(u) + 1j*Q*B + 2*1j*Q*psi - iR*Q**2*u + iR*dx(ux) = 0")
lv1.add_equation("1j*dt(A) - iRm*Q**2*A + iRm*dx(Ax) - 1j*Q*q*B - 1j*Co*Q**3*psi + 1j*Co*Q*dx(psix) = 0")
lv1.add_equation("1j*dt(B) - iRm*Q**2*B + iRm*dx(Bx) + 1j*Co*Q*u = 0")

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
#M = LEV.eigenvalue_pencil.M.todense()
#b = np.zeros_like(LEV.eigenvectors[0])
#lsolve = np.linalg.solve(L, b)


#Plotting --------------------------------------------

norm1 = -0.9/np.min(LEV.state['u']['g'].imag) #norm(LEV.eigenvectors)

#Currently normalized so that ee = 1
#ee = np.abs(np.real(LEV.eigenvectors)) + np.abs(np.imag(LEV.eigenvectors))

fig = plt.figure()
ax1 = fig.add_subplot(221)
#ax1.plot(x, LEV.state['psi']['g'].imag*norm1, color="red")
ax1.plot(x, LEV.state['psi']['g'].real*norm1, color="black")
ax1.set_title(r"Im($\psi^\dagger$)")

ax2 = fig.add_subplot(222)
ax2.plot(x, LEV.state['u']['g'].imag*norm1, color="black")
#ax2.plot(x, LEV.state['u']['g'].real*norm1, color="red")
ax2.set_title("Re($u^\dagger$)")

ax3 = fig.add_subplot(223)
ax3.plot(x, LEV.state['A']['g'].imag*norm1, color="black")
#ax3.plot(x, LEV.state['A']['g'].real*norm1, color="red")
ax3.set_title("Re($A^\dagger$)")

ax4 = fig.add_subplot(224)
#ax4.plot(x, LEV.state['B']['g'].imag*norm1, color="red")
ax4.plot(x, LEV.state['B']['g'].real*norm1, color="black")
ax4.set_title("Im($B^\dagger$)")
#fig.savefig("ah1.png")


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