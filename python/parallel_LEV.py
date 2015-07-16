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

gridnum = 128
print("running at gridnum", gridnum)
x_basis = Chebyshev(gridnum)
domain = Domain([x_basis], grid_dtype=np.complex128)

# Second basis for checking eigenvalues
x_basis192 = Chebyshev(192)
domain192 = Domain([x_basis192], grid_dtype = np.complex128)


def Pmrun(Pm, q, Co, dQ, dRm, Qsearch, Rmsearch):       

    print("Pm = %10.5e" % Pm)
  
    # Search all combinations of Qsearch and Rmsearch 
    QRm = np.array(list(itertools.product(Qsearch, Rmsearch)))
    Qs = QRm[:, 0]
    Rms = QRm[:, 1]

    start_time = time.time()

    params = (zip(Qs, itertools.repeat(Pm), Rms, itertools.repeat(q), itertools.repeat(Co), np.arange(len(Qs))))
    print("Processing %10.5e parameter combinations" % len(Qs))

    with Pool(processes = 19) as pool:
        result_pool = [pool.apply_async(run_mri_solve, args) for args in params]

        results = []
        for result in result_pool:
            try:
                results.append(result.get(100))
                print("Results appended.") 
            except mp.context.TimeoutError:
                print("TimeoutError encountered. Continuing..") 

    result_dict = {}    
    for x in results:
        result_dict[x[0]] = x[1]
    
    result_dict["Qsearch"] = Qsearch
    result_dict["Rmsearch"] = Rmsearch
    result_dict["Pm"] = Pm
    result_dict["q"] = q
    result_dict["beta"] = 2.0/Co
    result_dict["dQ"] = dQ
    result_dict["dRm"] = dRm
    
    print(result_dict)

    pickle.dump(result_dict, open("multirun/gridnum_"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Qsearch[0])+"_dQ_"+str(dQ)+"_Rm_"+str(Rmsearch[0])+"_dRm_"+str(dRm)+".p", "wb"))

    print("process took %10.5e seconds" % (time.time() - start_time))
    
def solve_LEV(problem):

    """
    Solves the linear eigenvalue problem for a ParsedProblem object.
    """
    
    problem.expand(domain)
    LEV = LinearEigenvalue(problem, domain)
    LEV.solve(LEV.pencils[0])
    
    return LEV
        
def solve_LEV_secondgrid(problem):

    """
    Solves the linear eigenvalue problem for a ParsedProblem object.
    Uses gridnum = 192 domain. For use in discarding spurious eigenvalues.
    """
    
    problem.expand(domain192)
    LEV = LinearEigenvalue(problem, domain192)
    LEV.solve(LEV.pencils[0])
    
    return LEV
    

def discard_spurious_eigenvalues(problem):
    
    """
    Solves the linear eigenvalue problem for two different resolutions.
    Returns trustworthy eigenvalues using nearest delta, from Boyd chapter 7.
    """

    # Solve the linear eigenvalue problem at two different resolutions.
    LEV1 = solve_LEV(problem)
    LEV2 = solve_LEV_secondgrid(problem)
    
    # Eigenvalues returned by dedalus must be multiplied by -1
    lambda1 = -LEV1.eigenvalues
    lambda2 = -LEV2.eigenvalues
    
    # Make sure argsort treats complex infs correctly
    lambda1[np.where(np.isnan(lambda1) == True)] = None
    lambda2[np.where(np.isnan(lambda2) == True)] = None
    lambda1[np.where(np.isinf(lambda1) == True)] = None
    lambda2[np.where(np.isinf(lambda2) == True)] = None
    
    # Sort lambda1 and lambda2 by real parts
    lambda1_indx = np.argsort(lambda1.real)
    lambda1_sorted = lambda1[lambda1_indx]
    lambda2_indx = np.argsort(lambda2.real)
    lambda2_sorted = lambda2[lambda2_indx]
    
    # Reverse engineer correct indices to make unsorted list from sorted
    reverse_lambda1_indx = sorted(range(len(lambda1_indx)), key=lambda1_indx.__getitem__)
    reverse_lambda2_indx = sorted(range(len(lambda2_indx)), key=lambda2_indx.__getitem__)
    
    # Compute sigmas from lower resolution run (gridnum = N1)
    sigmas = np.zeros(len(lambda1_sorted))
    sigmas[0] = np.abs(lambda1_sorted[0] - lambda1_sorted[1])
    sigmas[1:-1] = [0.5*(np.abs(lambda1_sorted[j] - lambda1_sorted[j - 1]) + np.abs(lambda1_sorted[j + 1] - lambda1_sorted[j])) for j in range(1, len(lambda1_sorted) - 1)]
    sigmas[-1] = np.abs(lambda1_sorted[-2] - lambda1_sorted[-1])
    
    # Nearest delta
    delta_near = [np.nanmin(np.abs(lambda1_sorted[j] - lambda2_sorted)) for j in range(len(lambda1_sorted))]/sigmas
    
    # Discard eigenvalues with 1/delta_near < 10^6
    delta_near_unsorted = delta_near[reverse_lambda1_indx]
    lambda1[np.where((1.0/delta_near_unsorted) < 1E6)] = None
    lambda1[np.where(np.isnan(1.0/delta_near_unsorted) == True)] = None
    
    return lambda1, LEV1

def get_largest_eigenvalue_index(LEV, goodevals = None):
        
    """
    Return index of largest eigenvalue. Can be positive or negative.
    """
    if goodevals == None:
        evals = LEV.eigenvalues
    else:
        evals = goodevals
        
    indx = np.arange(len(evals))
    largest_eval_indx = indx[evals == np.nanmax(evals)]
    
    return largest_eval_indx[0]
    
def get_largest_real_eigenvalue_index(LEV, goodevals = None):
        
    """
    Return index of largest eigenvalue. Can be positive or negative.
    """
    if goodevals == None:
        evals = LEV.eigenvalues
    else:
        evals = goodevals
        
    indx = np.arange(len(evals))
    largest_eval_indx = indx[evals.real == np.nanmax(evals.real)]
    
    return largest_eval_indx

def run_mri_solve(Q, Pm, Rm, q, Co, run_id):

    try:

        # Rm is an input parameter
        iRm = 1./Rm
        R = Rm/Pm
        iR = 1./R

        # Solve eigenvalues of linearized MRI set up. Will minimize to find k_z = Q and Rm = Rm_crit, where s ~ 0. 
        # This takes as inputs guesses for the critical values Q and Rm.
        lv1 = ParsedProblem(['x'],
                              field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                              param_names=['Q', 'iR', 'iRm', 'q', 'Co'])

        x_basis = Chebyshev(gridnum)
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
       
        # Discard spurious eigenvalues
        goodevals, LEV = discard_spurious_eigenvalues(lv1)
        
        # Find the largest eigenvalue (fastest growing mode).
        largest_eval_indx = get_largest_real_eigenvalue_index(LEV, goodevals = goodevals)
        
        val = goodevals[largest_eval_indx]
        return ((Rm, Q), val[0])
        
    except np.linalg.LinAlgError:
        return ((Rm, Q), np.nan + np.nan*1j)
        
if __name__ == '__main__':

    Pm = 1.0E-3
    q = 1.5
    beta = 25.0
    Co = 0.08

    dQ = 0.001
    dRm = 0.001
    #dQ = 0.01
    #dRm = 0.01
    
    Qsearch = np.arange(0.74, 0.76, dQ)
    Rmsearch = np.arange(4.87, 4.91, dRm)
    Pmrun(Pm, q, Co, dQ, dRm, Qsearch, Rmsearch)
    