from multiprocessing import Pool
import multiprocessing as mp
import itertools
import numpy as np
import matplotlib.pyplot as plt
#config['logging']['stdout_level'] = 'critical'
import dedalus.public as de
import pylab
import pickle
import time

nr1 = 64#256#512
r1 = de.Chebyshev('r', nr1, interval=(80, 120))
d1 = de.Domain([r1])

nr2 = 128#512#768
r2 = de.Chebyshev('r', nr2, interval=(80, 120))
d2 = de.Domain([r2])


print("grid number {}, spurious eigenvalue check at {}".format(nr1, nr2))

def Pmrun(Pm, q, Co, dQ, dRm, Qsearch, Rmsearch):       

    print("Pm = %10.5e" % Pm)
  
    # Search all combinations of Qsearch and Rmsearch 
    QRm = np.array(list(itertools.product(Qsearch, Rmsearch)))
    Qs = QRm[:, 0]
    Rms = QRm[:, 1]
    
    outfn = "multirun/widegap_gridnum_"+str(nr1)+"_grid2_"+str(nr2)+"_Pm_"+str(Pm)+"_Q_"+str(Qsearch[0])+"_dQ_"+str(dQ)+"_Rm_"+str(Rmsearch[0])+"_dRm_"+str(dRm)+"_allgoodeigs.p"

    start_time = time.time()

    params = (zip(Qs, itertools.repeat(Pm), Rms, itertools.repeat(q), itertools.repeat(Co), np.arange(len(Qs))))
    print("Processing %10.5e parameter combinations" % len(Qs))

    with Pool(processes = 19) as pool:
        result_pool = [pool.apply_async(run_mri_solve, args) for args in params]

        results = []
        for result in result_pool:
            try:
                results.append(result.get(500))
                print("Results appended.") 
            except mp.context.TimeoutError:
                print("TimeoutError encountered. Continuing..") 

    result_dict = {}    
    for x in results:
        result_dict[x[0]] = (x[1], x[2])
    
    result_dict["Qsearch"] = Qsearch
    result_dict["Rmsearch"] = Rmsearch
    result_dict["Pm"] = Pm
    result_dict["q"] = q
    result_dict["beta"] = 2.0/Co
    result_dict["dQ"] = dQ
    result_dict["dRm"] = dRm
    
    print(result_dict)

    pickle.dump(result_dict, open(outfn, "wb"))

    print("process took %10.5e seconds" % (time.time() - start_time))

    
def discard_spurious_eigenvalues(lambda1, lambda2):

    """
    lambda1 :: eigenvalues from low res run
    lambda2 :: eigenvalues from high res run
    
    Solves the linear eigenvalue problem for two different resolutions.
    Returns trustworthy eigenvalues using nearest delta, from Boyd chapter 7.
    """

    # Reverse engineer correct indices to make unsorted list from sorted
    reverse_lambda1_indx = np.arange(len(lambda1)) 
    reverse_lambda2_indx = np.arange(len(lambda2))

    lambda1_and_indx = np.asarray(list(zip(lambda1, reverse_lambda1_indx)))
    lambda2_and_indx = np.asarray(list(zip(lambda2, reverse_lambda2_indx)))
    
    # remove nans
    lambda1_and_indx = lambda1_and_indx[np.isfinite(lambda1)]
    lambda2_and_indx = lambda2_and_indx[np.isfinite(lambda2)]

    # Sort lambda1 and lambda2 by real parts
    lambda1_and_indx = lambda1_and_indx[np.argsort(lambda1_and_indx[:, 0].real)]
    lambda2_and_indx = lambda2_and_indx[np.argsort(lambda2_and_indx[:, 0].real)]
    
    lambda1_sorted = lambda1_and_indx[:, 0]
    lambda2_sorted = lambda2_and_indx[:, 0]

    # Compute sigmas from lower resolution run (gridnum = N1)
    sigmas = np.zeros(len(lambda1_sorted))
    sigmas[0] = np.abs(lambda1_sorted[0] - lambda1_sorted[1])
    sigmas[1:-1] = [0.5*(np.abs(lambda1_sorted[j] - lambda1_sorted[j - 1]) + np.abs(lambda1_sorted[j + 1] - lambda1_sorted[j])) for j in range(1, len(lambda1_sorted) - 1)]
    sigmas[-1] = np.abs(lambda1_sorted[-2] - lambda1_sorted[-1])

    if not (np.isfinite(sigmas)).all():
        print("WARNING: at least one eigenvalue spacings (sigmas) is non-finite (np.inf or np.nan)!")

    # Nearest delta
    delta_near = np.array([np.nanmin(np.abs(lambda1_sorted[j] - lambda2_sorted)/sigmas[j]) for j in range(len(lambda1_sorted))])

    # Discard eigenvalues with 1/delta_near < 10^6
    lambda1_and_indx = lambda1_and_indx[np.where((1.0/delta_near) > 1E6)]
    #print(lambda1_and_indx)
    
    lambda1 = lambda1_and_indx[:, 0]
    indx = lambda1_and_indx[:, 1]
    
    return lambda1, indx

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

def run_mri_solve(Q, Pm, Rm, q, Co, run_id, all_mode=False):
    """Solve the numerical eigenvalue problem for a single value of parameters.

    inputs:
    Q: k_z
    Pm: Magnetic Prandtl Number
    Rm: Magnetic Reynolds Number
    q: shear parameter (=3/2 for keplerian flow)
    run_id: ?
    all_mode (bool): if true, return all the good eigenvalues. if false,
    return only the eigenvalue with the largest real part (which could be
    less than zero, if the parameters specified are stable).

    """
    try:

        # Rm is an input parameter
        iRm = 1./Rm
        R = Rm/Pm
        iR = 1./R

        # widegap order epsilon
        widegap1 = de.EVP(d1,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],'sigma')
        widegap2 = de.EVP(d2,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],'sigma')
        
        # Add equations
        for widegap in [widegap1, widegap2]:
            widegap.parameters['Q'] = Q
            widegap.parameters['iR'] = iR
            widegap.parameters['iRm'] = iRm
            widegap.parameters['q'] = q
            widegap.parameters['beta'] = beta
        
            widegap.add_equation("sigma*(-1*Q**2*r**3*psi + r**3*psirr - r**2*psir) - 3*1j*Q*r**(4 - q)*u - iR*r**3*Q**4*psi + (2/beta)*1j*Q**3*r**3*A + iR*2*Q**2*r**3*psirr - iR*2*Q**2*r**2*psir - (2/beta)*1j*Q*r**3*dr(Ar) + (2/beta)*1j*Q*r**2*Ar - iR*r**3*dr(psirrr) + 2*iR*r**2*psirrr - iR*3*r*psirr + iR*3*psir = 0")
            widegap.add_equation("sigma*(r**4*u) - 1j*Q*q*r**(3 - q)*psi + 4*1j*Q*r**(3 - q)*psi + iR*r**4*Q**2*u - (2/beta)*1j*Q*r**4*B - iR*r**4*dr(ur) - iR*r**3*ur + iR*r**3*u = 0")
            widegap.add_equation("sigma*(r**4*A) + iRm*r**4*Q**2*A - 1j*Q*r**4*psi - iRm*r**4*dr(Ar) + iRm*r**3*Ar = 0")
            widegap.add_equation("sigma*(r**4*B) + 1j*Q*q*r**(3 - q)*A - 2*1j*Q*r**(3 - q)*A + iRm*r**4*Q**2*B - 1j*Q*r**4*u - iRm*r**4*dr(Br) - iRm*r**3*Br + iRm*r**2*B = 0")

            widegap.add_equation("dr(psi) - psir = 0")
            widegap.add_equation("dr(psir) - psirr = 0")
            widegap.add_equation("dr(psirr) - psirrr = 0")
            widegap.add_equation("dr(u) - ur = 0")
            widegap.add_equation("dr(A) - Ar = 0")
            widegap.add_equation("dr(B) - Br = 0")

            widegap.add_bc('left(u) = 0')
            widegap.add_bc('right(u) = 0')
            widegap.add_bc('left(psi) = 0')
            widegap.add_bc('right(psi) = 0')
            widegap.add_bc('left(A) = 0')
            widegap.add_bc('right(A) = 0')
            widegap.add_bc('left(psi + r*psir) = 0')
            widegap.add_bc('right(psi + r*psir) = 0')
            widegap.add_bc('left(B + r*Br) = 0')
            widegap.add_bc('right(B + r*Br) = 0')
        
        solver1 = widegap1.build_solver()
        solver2 = widegap2.build_solver()
        
        solver1.solve(solver1.pencils[0])
        solver2.solve(solver2.pencils[0])
        
        # Discard spurious eigenvalues
        ev1 = solver1.eigenvalues
        ev2 = solver2.eigenvalues
        goodeigs, goodeigs_indices = discard_spurious_eigenvalues(ev1, ev2)

        #goodeigs_index = np.where(goodeigs.real == np.nanmax(goodeigs.real))[0][0]
        goodeigs_index = np.nanargmax(goodeigs.real)
        print(goodeigs_index, int(goodeigs_indices[goodeigs_index]))
        marginal_mode_index = int(goodeigs_indices[goodeigs_index])
        
        # Largest real-part eigenvalue
        critical_eval = goodeigs[goodeigs_index]

        if all_mode:
            return goodeigs
        else:
            return ((Rm, Q), critical_eval, (goodeigs))
        
    except np.linalg.LinAlgError:
        return ((Rm, Q), np.nan + np.nan*1j, ([np.nan + np.nan*1j]))
        
if __name__ == '__main__':

    Pm = 1.0E-3
    #Pm = 5.0E-3
    #Pm = 1.0E-2
    #Pm = 2.0E-4
    #Pm = 3.0E-4
    #Pm = 5.0E-4
    #Pm = 9.0E-5
    #Pm = 5.0E-5
    #Pm = 2.0E-3
    #Pm = 5.0E-6
    #Pm = 7.0E-4
    #Pm = 7.0E-3
    
    q = 1.5
    beta = 25.0
    Co = 0.08

    #dQ = 0.001
    #dRm = 0.001
    dQ = 0.1
    dRm = 0.1
    
    Qsearch = np.arange(0.2, 1.4, dQ)
    #Qsearch = np.arange(0.74, 0.76, dQ)
    #Rmsearch = np.arange(4.87, 4.91, dRm) # Great for <1E-2
    #Rmsearch = np.arange(4.91, 4.95, dRm)
    Rmsearch = np.arange(4.0, 6.0, dRm)
    Pmrun(Pm, q, Co, dQ, dRm, Qsearch, Rmsearch)
    
