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
                          param_names=['Q', 'iR', 'iRm', 'q', 'ifourpi'])
                      
    x_basis = Chebyshev(16)
    domain = Domain([x_basis])

    #Rm is an input parameter
    iRm = 1./Rm
    
    #Parameter values from Umurhan+:
    Pm = 0.001 #Pm = Rm/R
    R = Rm/Pm
    iR = 1./R
    q = 3/2.
    
    ifourpi = 1./(4*np.pi)

    #multiply by -1j and add dt's:
    lv1.add_equation("-dt(dx(psix)) - Q**2*dt(psi) - iR*(dx(psixxx) + 2*dx(psix)*Q**2 + Q**4*psi) + 2*1j*Q*u + ifourpi*1j*Q*(-dx(Ax) - Q**2*A) = 0")
    lv1.add_equation("1j*Q*(2 - q)*psi - iR*(-dx(ux) - Q**2*u) - ifourpi*1j*Q*B + dt(u) = 0")
    lv1.add_equation("-1j*Q*psi - iR*(-dx(Ax) - Q**2*A) + dt(A) = 0")
    lv1.add_equation("-1j*Q*u + 1j*Q*q*A - iRm*(-dx(Bx) - Q**2*B) + dt(B) = 0")

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

    lv1.expand(domain)
    LEV = LinearEigenvalue(lv1,domain)
    LEV.solve(LEV.pencils[0])

    #Find the eigenvalue that is closest to zero.
    evals = LEV.eigenvalues
    indx = np.arange(len(evals))
    e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
    
    val = evals[e0]
    
    return val[0]


def paramsearch():

    #Qsearch = np.arange(0.748, 0.76, 0.001)
    #Rmsearch = np.arange(4.88, 5.0, 0.001)
    #Qsearch = np.arange(0.745, 0.755, 0.001)
    #Rmsearch = np.arange(4.85, 4.95, 0.01)
    Qsearch = np.arange(0.755, 0.765, 0.001)
    Rmsearch = np.arange(4.95, 5.01, 0.01)
    
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
    
esearch, Qs, Rms = paramsearch()

np.save("esearch_2.npy", esearch)
np.save("Qsearch_2.npy", Qs)
np.save("Rmsearch_2.npy", Rms)

