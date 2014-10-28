from multiprocessing import Pool
import itertools
import numpy as np
import matplotlib.pyplot as plt
from dedalus2.tools.config import config
config['logging']['stdout_level'] = 'critical'
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue
import pylab
import pickle

import sys

def run_mri_solve_2(Q, Pm, Rm, q, B0, Co, lv1=None, LEV=None):
    output = "Hello. Parameter Q = %10.5e" % Q
    return output

def run_mri_solve(Q, Pm, Rm, q, Co):

    # Rm is an input parameter
    iRm = 1./Rm
    R = Rm/Pm
    iR = 1./R

    #Q = float(sys.argv[2])
    #Rm = float(sys.argv[1])
    #name = sys.argv[0]
    #print("%s: Q = %10.5f; Rm = %10.5f" % (name, Q, Rm))

    # Solve eigenvalues of linearized MRI set up. Will minimize to find k_z = Q and Rm = Rm_crit, where s ~ 0. 
    # This takes as inputs guesses for the critical values Q and Rm.
    lv1 = ParsedProblem(['x'],
                          field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                          param_names=['Q', 'iR', 'iRm', 'q', 'Co'])

    x_basis = Chebyshev(64)
    domain = Domain([x_basis])

    # In terms of Co ....multiplied dt terms by -1j
    lv1.add_equation("-1j*dt(psixx) - -1j*Q**2*dt(psi) - iR*dx(psixxx) + 2*iR*Q**2*psixx - iR*Q**4*psi - 2*1j*Q*u - Co*1j*Q*dx(Ax) + Co*Q**3*1j*A = 0")
    lv1.add_equation("-1j*dt(u) - iR*dx(ux) + iR*Q**2*u + (2-q)*1j*Q*psi - Co*1j*Q*B = 0") 
    lv1.add_equation("-1j*dt(A) - iRm*dx(Ax) + iRm*Q**2*A - 1j*Q*psi = 0") 
    lv1.add_equation("-1j*dt(B) - iRm*dx(Bx) + iRm*Q**2*B - 1j*Q*u + q*1j*Q*A = 0")

    lv1.add_equation("dx(psi) - psix = 0")
    lv1.add_equation("dx(psix) - psixx = 0")
    lv1.add_equation("dx(psixx) - psixxx = 0")
    lv1.add_equation("dx(u) - ux = 0")
    lv1.add_equation("dx(A) - Ax = 0")
    lv1.add_equation("dx(B) - Bx = 0")

    # Boundary conditions
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


    # Parameters
    lv1.parameters['Q'] = Q
    lv1.parameters['iR'] = iR
    lv1.parameters['iRm'] = iRm
    lv1.parameters['q'] = q
    lv1.parameters['Co'] = Co

    lv1.expand(domain)
    LEV = LinearEigenvalue(lv1,domain)
    LEV.solve(LEV.pencils[0])

    # Find the eigenvalue that is closest to zero.
    evals = LEV.eigenvalues
    indx = np.arange(len(evals))
    e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]

    val = evals[e0]
    return val[0]

if __name__ == '__main__':

    # Parameter values from Umurhan+:
    #Q = 0.75
    #Rm = 4.8775
    Pm = 0.0001 #Pm = Rm/R
    q = 3/2.
    Co = 0.08

    Qsearch = np.arange(0.05, 10, 0.05)
    Rmsearch = np.arange(0.05, 8, 0.05)
    
    # Search all combinations of Qsearch and Rmsearch 
    QRm = np.array(list(itertools.product(Qsearch, Rmsearch)))
    Qs = QRm[:, 0]
    Rms = QRm[:, 1]

    with Pool(processes=15) as pool:
       result = pool.starmap_async(run_mri_solve, (zip(Qs, itertools.repeat(Pm), Rms, itertools.repeat(q), itertools.repeat(Co))))
       print(result.get(timeout=75))
     
    results = result.get()  
    pickle.dump(results, open("multirun/Pm_"+str(Pm)+"_Q_"+str(Qsearch[0])+"_Rm_"+str(Rmsearch[0])+".p", "wb"))



