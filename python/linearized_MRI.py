import numpy as np
import matplotlib.pyplot as plt
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue
import pylab

# Solve eigenvalues of linearized MRI set up. Will minimize to find k_z = Q and Rm = Rm_crit, where s ~ 0. 
# This takes as inputs guesses for the critical values Q and Rm.

def linear_MRI(Q, Rm):

    lv1 = ParsedProblem(['x'],
                          field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                          param_names=['Q', 'iR', 'iRm', 'q', 'ifourpi', 'Co'])
                      
    x_basis = Chebyshev(64)
    domain = Domain([x_basis])

    #Rm is an input parameter
    iRm = 1./Rm
    
    #Parameter values from Umurhan+:
    Pm = 0.001 #Pm = Rm/R
    R = Rm/Pm
    iR = 1./R
    q = 3/2.
    
    ifourpi = 0.08#1./(4*np.pi)
    Co = 0.08

    #multiply by -1j and add dt's:
    #lv1.add_equation("-dt(psixx) - Q**2*dt(psi) - iR*(dx(psixxx) + 2*psixx*Q**2 + Q**4*psi) + 2*1j*Q*u + ifourpi*1j*Q*(-dx(Ax) - Q**2*A) = 0")
    #lv1.add_equation("1j*Q*(2 - q)*psi - iR*(-dx(ux) - Q**2*u) - ifourpi*1j*Q*B + dt(u) = 0")
    #lv1.add_equation("-1j*Q*psi - iR*(-dx(Ax) - Q**2*A) + dt(A) = 0")
    #lv1.add_equation("-1j*Q*u + 1j*Q*q*A - iRm*(-dx(Bx) - Q**2*B) + dt(B) = 0")
    
    
    #NON-ADJOINT LV=0
    #lv1.add_equation("1j*dt(psi) - 1j*Co*Q**3*A + 1j*Co*Q*dx(Ax) + 2*1j*Q*u + iR*Q**4*psi - iR*2*Q**2*psixx + iR*dx(psixxx) = 0")
    #lv1.add_equation("1j*dt(psixx) - 1j*Q**2*dt(psi) - 1j*Co*Q**3*A + 1j*Co*Q*dx(Ax) + 2*1j*Q*u + iR*Q**4*psi - iR*2*Q**2*psixx + iR*dx(psixxx) = 0")
    #lv1.add_equation("1j*dt(u) + 1j*B*Co*Q - 1j*Q*(2 - q)*psi - iR*Q**2*u + iR*dx(ux) = 0")
    #lv1.add_equation("1j*dt(A) - iRm*Q**2*A + iRm*dx(Ax) + 1j*Q*psi = 0")
    #lv1.add_equation("1j*dt(B) - 1j*Q*q*A -iRm*Q**2*B + iRm*dx(Bx) + 1j*Q*u = 0")
    
    #In terms of Co ....multiplied dt terms by -1j
    #lv1.add_equation("-1j*dt(psixx) - -1j*Q**2*dt(psi) - iR*dx(psixxx) + 2*iR*Q**2*psixx - iR*Q**4*psi - 2*1j*Q*u - Co*1j*Q*dx(Ax) + Co*Q**3*1j*A = 0")
    #lv1.add_equation("-1j*dt(u) - iR*dx(ux) + iR*Q**2*u + (2-q)*1j*Q*psi - Co*1j*Q*B = 0") 
    #lv1.add_equation("-1j*dt(A) - iRm*dx(Ax) + iRm*Q**2*A - 1j*Q*psi = 0") 
    #lv1.add_equation("-1j*dt(B) - iRm*dx(Bx) + iRm*Q**2*B - 1j*Q*u + q*1j*Q*A = 0")

    lv1.add_equation("1j*dt(psixx) - 1j*Q**2*dt(psi) - 1j*Co*Q**3*A + 1j*Co*Q*dx(Ax) + 2*1j*Q*u + iR*Q**4*psi - iR*2*Q**2*psixx + iR*dx(psixxx) = 0")
    lv1.add_equation("1j*dt(u) + 1j*Co*B + 1j*Q*(q-2)*psi - iR*Q**2*u + iR*dx(ux) = 0")
    lv1.add_equation("1j*dt(A) - iRm*Q**2*A + iRm*dx(Ax) + 1j*Q*psi = 0")
    lv1.add_equation("1j*dt(B) - 1j*Q*q*A - iRm*Q**2*B + iRm*dx(Bx) + 1j*Q*u = 0")

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
    lv1.parameters['ifourpi'] = ifourpi
    lv1.parameters['Co'] = Co

    lv1.expand(domain)
    LEV = LinearEigenvalue(lv1,domain)
    LEV.solve(LEV.pencils[0])

    #Find the eigenvalue that is closest to zero.
    evals = LEV.eigenvalues
    indx = np.arange(len(evals))
    e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
    
    val = evals[e0]
    
    #return val[0]
    LEV.set_state(e0[0])
    x = domain.grid(0)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(x, LEV.state['psi']['g'].imag)
    ax1.plot(x, LEV.state['psi']['g'].real)
    ax1.set_title(r"Im($\psi_{11}$)")
    #ax1.set_ylim(-0.4, 0.1)

    ax2 = fig.add_subplot(222)
    ax2.plot(x, LEV.state['u']['g'].real)
    ax2.plot(x, LEV.state['u']['g'].imag)
    ax2.set_title("Re($u_{11}$)")
    #ax2.set_ylim(-0.5, 1.5)

    ax3 = fig.add_subplot(223)
    ax3.plot(x, LEV.state['A']['g'].real)
    ax3.plot(x, LEV.state['A']['g'].imag)
    ax3.set_title("Re($A_{11}$)")
    #ax3.set_ylim(-0.2, 0.6)

    ax4 = fig.add_subplot(224)
    ax4.plot(x, LEV.state['B']['g'].imag)
    ax4.plot(x, LEV.state['B']['g'].real)
    ax4.set_title("Im($B_{11}$)")
    #ax4.set_ylim(-2.3, -1.8)


def paramsearch():

    #Qsearch = np.arange(0.748, 0.76, 0.001)
    #Rmsearch = np.arange(4.88, 5.0, 0.001)
    #Qsearch = np.arange(0.745, 0.755, 0.001)
    #Rmsearch = np.arange(4.85, 4.95, 0.01)
    #Qsearch = np.arange(0.745, 0.755, 0.001)
    #Rmsearch = np.arange(5.01, 5.06, 0.01)
    Qsearch = np.arange(0.5, 0.9, 0.05)
    Rmsearch = np.arange(3, 6, 0.5)
    
    esearch = np.zeros((len(Qsearch), len(Rmsearch)), np.complex128)
    Rms = np.zeros((len(Qsearch), len(Rmsearch)), np.float_)
    Qs = np.zeros((len(Qsearch), len(Rmsearch)), np.float_)
    
    print(esearch.shape)
    
    for i in range(len(Qsearch)):
       for j in range(len(Rmsearch)):
           
           #e = linear_MRI(Qsearch[i], Rmsearch[j])
           #print(e)
           esearch[i, j] = linear_MRI(Qsearch[i], Rmsearch[j])
           Qs[i, j] = Qsearch[i]
           Rms[i, j] = Rmsearch[j]
    
    return esearch, Qs, Rms
    
#esearch, Qs, Rms = paramsearch()

#np.save("esearch_4.npy", esearch)
#np.save("Qsearch_4.npy", Qs)
#np.save("Rmsearch_4.npy", Rms)

