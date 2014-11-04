from multiprocessing import Pool
import multiprocessing as mp
import itertools
import numpy as np
import matplotlib.pyplot as plt
from dedalus2.tools.config import config
config['logging']['stdout_level'] = 'critical'
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue
import pylab
import pickle
import time

import sys

def run_mri_solve_2(Q, Pm, Rm, q, B0, Co, lv1=None, LEV=None):
    output = "Hello. Parameter Q = %10.5e" % Q
    return output

def Pmrun(Pm, q, Co, dQ, dRm, Qsearch, Rmsearch):       

    print("Pm = %10.5e" % Pm)
  
    # Search all combinations of Qsearch and Rmsearch 
    QRm = np.array(list(itertools.product(Qsearch, Rmsearch)))
    Qs = QRm[:, 0]
    Rms = QRm[:, 1]

    start_time = time.time()

    params = (zip(Qs, itertools.repeat(Pm), Rms, itertools.repeat(q), itertools.repeat(Co), np.arange(len(Qs))))
    print("Processing %10.5e parameter combinations" % len(Qs))

    with Pool(processes=16) as pool:
        result_pool = [pool.apply_async(run_mri_solve, args) for args in params]

        results = []
        for result in result_pool:
            try:
                results.append(result.get(100))
            except mp.context.TimeoutError:
                print("TimeoutError encountered. Continuing..") 

    result_dict = {}    
    for x in results:
        result_dict[x[0]] = x[1]
    
    print(result_dict)
    print("test: ", result_dict[0])


    pickle.dump(result_dict, open("multirun/Pm_"+str(Pm)+"_Q_"+str(Qsearch[0])+"_dQ_"+str(dQ)+"_Rm_"+str(Rmsearch[0])+"_dRm_"+str(dRm)+".p", "wb"))

    print("process took %10.5e seconds" % (time.time() - start_time))

def Pmrun_hmri(Pm, q, beta, dQ, dRm, Qsearch, Rmsearch, xi, x0): 
    Co = 2.0/beta      

    print("Pm = %10.5e" % Pm)
    print("beta = %10.5e" % beta)
    print("this is the hmri")
  
    # Search all combinations of Qsearch and Rmsearch 
    QRm = np.array(list(itertools.product(Qsearch, Rmsearch)))
    Qs = QRm[:, 0]
    Rms = QRm[:, 1]

    start_time = time.time()

    params = (zip(Qs, itertools.repeat(Pm), Rms, itertools.repeat(q), itertools.repeat(Co), np.arange(len(Qs)), itertools.repeat(xi), itertools.repeat(x0)))
    print("Processing %10.5e parameter combinations" % len(Qs))

    with Pool(processes=16) as pool:
        result_pool = [pool.apply_async(run_hmri_solve, args) for args in params]

        results = []
        for result in result_pool:
            try:
                results.append(result.get(100))
            except mp.context.TimeoutError:
                print("TimeoutError encountered. Continuing..") 

    result_dict = {}    
    for x in results:
        result_dict[x[0]] = x[1]
    
    print(result_dict)
    print("test: ", result_dict[0])


    pickle.dump(result_dict, open("multirun/hmri/Pm_"+str(Pm)+"_beta_"+str(beta)+"_Q_"+str(Qsearch[0])+"_dQ_"+str(dQ)+"_Rm_"+str(Rmsearch[0])+"_dRm_"+str(dRm)+".p", "wb"))

    print("process took %10.5e seconds" % (time.time() - start_time))
    
def Betarun(Pm, q, beta, dQ, dRm, Qsearch, Rmsearch):   
    Co = 2.0/beta   

    print("Pm = %10.5e" % Pm)
    print("beta = %10.5e" % beta)
  
    # Search all combinations of Qsearch and Rmsearch 
    QRm = np.array(list(itertools.product(Qsearch, Rmsearch)))
    Qs = QRm[:, 0]
    Rms = QRm[:, 1]

    start_time = time.time()

    params = (zip(Qs, itertools.repeat(Pm), Rms, itertools.repeat(q), itertools.repeat(Co), np.arange(len(Qs))))
    print("Processing %10.5e parameter combinations" % len(Qs))

    with Pool(processes=16) as pool:
        result_pool = [pool.apply_async(run_mri_solve, args) for args in params]

        results = []
        for result in result_pool:
            try:
                results.append(result.get(100))
            except mp.context.TimeoutError:
                print("TimeoutError encountered. Continuing..") 

    result_dict = {}    
    for x in results:
        result_dict[x[0]] = x[1]
    
    print(result_dict)
    print("test: ", result_dict[0])


    pickle.dump(result_dict, open("multirun/beta_"+str(beta)+"_Pm_"+str(Pm)+"_Q_"+str(Qsearch[0])+"_dQ_"+str(dQ)+"_Rm_"+str(Rmsearch[0])+"_dRm_"+str(dRm)+".p", "wb"))

    print("process took %10.5e seconds" % (time.time() - start_time))
        

def run_mri_solve(Q, Pm, Rm, q, Co, run_id):

    try:

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
        return (run_id, val[0])
        
    except np.linalg.LinAlgError:
        return (run_id, np.nan)
    
def run_hmri_solve(Q, Pm, Rm, q, beta, run_id):

    try:

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
                              param_names=['Q', 'iR', 'iRm', 'q', 'beta', 'xi', 'x0'])

        x_basis = Chebyshev(64)
        domain = Domain([x_basis])
        
         # linear hMRI equations in terms of beta ....multiplied dt terms by -1j
        lv1.add_equation("-1j*dt(psixx) - -1j*Q**2*dt(psi) - iR*dx(psixxx) + 2*iR*Q**2*psixx - iR*Q**4*psi - 2*1j*Q*u - (2/beta)*1j*Q*dx(Ax) + (2/beta)*Q**3*1j*A - xi*x0*(1/x**2)*1j*Q*B + xi*x0*(1/x)*1j*Q*Bx = 0")
        lv1.add_equation("-1j*dt(u) - iR*dx(ux) + iR*Q**2*u + (2-q)*1j*Q*psi - (2/beta)*1j*Q*B + xi*(x0/x**2)*1j*Q*A = 0") 
        lv1.add_equation("-1j*dt(A) - iRm*dx(Ax) + iRm*Q**2*A - 1j*Q*psi = 0") 
        lv1.add_equation("-1j*dt(B) - iRm*dx(Bx) + iRm*Q**2*B - 1j*Q*u + q*1j*Q*A - xi*(x0/x**2)*1j*Q*psi = 0")

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
        lv1.parameters['beta'] = beta
        lv1.parameters['xi'] = xi
        lv1.parameters['x0'] = x0

        lv1.expand(domain)
        LEV = LinearEigenvalue(lv1,domain)
        LEV.solve(LEV.pencils[0])

        # Find the eigenvalue that is closest to zero.
        evals = LEV.eigenvalues
        indx = np.arange(len(evals))
        e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]

        val = evals[e0]
        return (run_id, val[0])
        
    except np.linalg.LinAlgError:
        return (run_id, np.nan)

if __name__ == '__main__':
    # beta = 2/Co. Co = 2/beta.

    # Parameter values from Umurhan+:
    #Q = 0.75
    #Rm = 4.8775
    #Co = 0.08
    
    Pm = 1E-3 #Pm = Rm/R
    q = 3/2.
    beta = 25#0.250
    xi = 1.0
    x0 = 1.0

    dQ = 0.05
    dRm = 0.05
    
    #Qsearch = np.arange(0.2, 1.2, dQ)
    #Rmsearch = np.arange(5.0, 6.1, dRm)
    Qsearch = np.arange(0.2, 1.5, dQ)
    Rmsearch = np.arange(0.5, 8, dRm)
    
    Pmrun_hmri(Pm, q, beta, dQ, dRm, Qsearch, Rmsearch, xi, x0)
    
    """
    Betarun(Pm, q, beta, dQ, dRm, Qsearch, Rmsearch)

    beta = 0.025
    Betarun(Pm, q, beta, dQ, dRm, Qsearch, Rmsearch)
    
    beta = 0.0025
    Betarun(Pm, q, beta, dQ, dRm, Qsearch, Rmsearch)
    """


