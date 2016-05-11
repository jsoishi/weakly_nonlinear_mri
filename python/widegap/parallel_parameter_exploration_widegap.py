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

nr1 = 128 #64#256#512
r1 = de.Chebyshev('r', nr1, interval=(5, 15))
d1 = de.Domain([r1])

nr2 = 192 #128#512#768
r2 = de.Chebyshev('r', nr2, interval=(5, 15))
d2 = de.Domain([r2])


print("grid number {}, spurious eigenvalue check at {}".format(nr1, nr2))

def Pmrun(Pm, c1, c2, beta, dQ, dRm, Qsearch, Rmsearch):       

    print("Pm = %10.5e" % Pm)
  
    # Search all combinations of Qsearch and Rmsearch 
    QRm = np.array(list(itertools.product(Qsearch, Rmsearch)))
    Qs = QRm[:, 0]
    Rms = QRm[:, 1]
    
    root = "/home/sec2170/weakly_nonlinear_mri/weakly_nonlinear_mri/python/"
    outfn = root + "multirun/widegap_gridnum_"+str(nr1)+"_grid2_"+str(nr2)+"_Pm_"+str(Pm)+"_Q_"+str(Qsearch[0])+"_dQ_"+str(dQ)+"_Rm_"+str(Rmsearch[0])+"_dRm_"+str(dRm)+"_fixr_allgoodeigs.p"

    start_time = time.time()

    params = (zip(Qs, itertools.repeat(Pm), Rms, itertools.repeat(c1), itertools.repeat(c2), itertools.repeat(beta), np.arange(len(Qs))))
    print("Processing %10.5e parameter combinations" % len(Qs))

    with Pool(processes = 18) as pool:
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
    result_dict["c1"] = c1
    result_dict["c2"] = c2
    result_dict["beta"] = beta
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

def run_mri_solve(Q, Pm, Rm, c1, c2, beta, run_id, all_mode=False):
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
        print(Q, Pm, Rm, c1, c2, beta, run_id)
        
        # Rm is an input parameter
        iRm = 1./Rm
        R = Rm/Pm
        iR = 1./R

        # widegap order epsilon
        widegap1 = de.EVP(d1,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],'sigma')
        widegap2 = de.EVP(d2,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],'sigma')
        
        # Add equations
        for widegap in [widegap1, widegap2]:
            widegap.parameters['k'] = Q
            widegap.parameters['iR'] = iR
            widegap.parameters['iRm'] = iRm
            widegap.parameters['c1'] = c1
            widegap.parameters['c2'] = c2
            widegap.parameters['beta'] = beta
            widegap.parameters['B0'] = 1

            widegap.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
            widegap.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
            widegap.substitutions['twooverbeta'] = '(2.0/beta)'
            widegap.substitutions['psivisc'] = '(2*r**2*k**2*psir - 2*r**3*k**2*psirr + r**3*k**4*psi + r**3*dr(psirrr) - 3*psir + 3*r*psirr - 2*r**2*psirrr)'
            widegap.substitutions['uvisc'] = '(-r**3*k**2*u + r**3*dr(ur) + r**2*ur - r*u)'
            widegap.substitutions['Avisc'] = '(r*dr(Ar) - r*k**2*A - Ar)' # checked on whiteboard 5/6
            widegap.substitutions['Bvisc'] = '(-r**3*k**2*B + r**3*dr(Br) + r**2*Br - r*B)'
            
            widegap.add_equation("sigma*(-r**3*k**2*psi + r**3*psirr - r**2*psir) - r**2*2*ru0*1j*k*u + r**3*twooverbeta*B0*1j*k**3*A + twooverbeta*B0*r**2*1j*k*Ar - twooverbeta*r**3*B0*1j*k*dr(Ar) - iR*psivisc = 0") #corrected on whiteboard 5/6
            widegap.add_equation("sigma*r**3*u + 1j*k*ru0*psi + 1j*k*rrdu0*psi - 1j*k*r**3*twooverbeta*B0*B - iR*uvisc = 0") # correct 5/5
            widegap.add_equation("sigma*r*A - r*B0*1j*k*psi - iRm*Avisc = 0")
            widegap.add_equation("sigma*r**3*B + ru0*1j*k*A - r**3*B0*1j*k*u - 1j*k*rrdu0*A - iRm*Bvisc = 0") # correct 5/5
    
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
            widegap.add_bc('left(psir) = 0')
            widegap.add_bc('right(psir) = 0')
            widegap.add_bc('left(B + r*Br) = 0')
            widegap.add_bc('right(B + r*Br) = 0') # axial component of current = 0
        
        solver1 = widegap1.build_solver()
        solver2 = widegap2.build_solver()
        
        solver1.solve(solver1.pencils[0])
        solver2.solve(solver2.pencils[0])
        
        # Discard spurious eigenvalues
        ev1 = solver1.eigenvalues
        ev2 = solver2.eigenvalues
        print(len(ev1), len(ev2))
        goodeigs, goodeigs_indices = discard_spurious_eigenvalues(ev1, ev2)

        try:
            goodeigs_index = np.nanargmax(goodeigs.real)
            print(goodeigs_index, int(goodeigs_indices[goodeigs_index]))
            marginal_mode_index = int(goodeigs_indices[goodeigs_index])
        
            # Largest real-part eigenvalue
            critical_eval = goodeigs[goodeigs_index]
        except ValueError:
            critical_eval = np.nan + np.nan*1j
            goodeigs = [np.nan + np.nan*1j]

        if all_mode:
            return goodeigs
        else:
            return ((Rm, Q), critical_eval, (goodeigs))
        
    except np.linalg.LinAlgError:
        return ((Rm, Q), np.nan + np.nan*1j, ([np.nan + np.nan*1j]))
        
if __name__ == '__main__':

    """
    Pm = 1.0E-3    
    q = 1.5
    beta = 25.0
    Co = 0.08
    """
    
    # Parameters approximating Umurhan+ 2007
    Pm = 1.0E-3
    beta = 25.0
    R1 = 5
    R2 = 15
    Omega1 = 313.55
    Omega2 = 67.0631
    
    # Umurhan+ "thin gap"
    #R1 = 9.5
    #R2 = 10.5
    #Omega2 = 270.25
    
    # Parameters approximating Goodman & Ji 2001    
    #Pm = 1.6E-6
    #beta = 0.43783886002604167#25.0
    #R1 = 5
    #R2 = 15
    #Omega1 = 314
    #Omega2 = 37.9

    c1 = (Omega2*R2**2 - Omega1*R1**2)/(R2**2 - R1**2)
    c2 = (R1**2*R2**2*(Omega1 - Omega2))/(R2**2 - R1**2)
    
    zeta_mean = 2*(R2**2*Omega2 - R1**2*Omega1)/((R2**2 - R1**2)*np.sqrt(Omega1*Omega2))
    
    
    print("mean zeta is {}, meaning q = 2 - zeta = {}".format(zeta_mean, 2 - zeta_mean))
    
    # critical parameters found in Goodman & Ji 2001 - search around these
    #Rm = 4.052031250000001
    #Q = np.pi/10 

    dQ = 0.05
    dRm = 0.05
    
    Qsearch = np.arange(0.6, 0.9, dQ)
    #Qsearch = np.arange(0.2, 0.6, dQ)
    #Qsearch = np.arange(-0.2, 0.2, dQ)
    Rmsearch = np.arange(4.6, 5.1, dRm)
    #Rmsearch = np.arange(5.1, 6.0, dRm)
    
    #Qsearch = np.arange(0.2, 0.4, dQ)
    #Qsearch = np.arange(0.74, 0.76, dQ)
    #Rmsearch = np.arange(4.87, 4.91, dRm) # Great for <1E-2
    #Rmsearch = np.arange(4.91, 4.95, dRm)
    #Rmsearch = np.arange(3.2, 4.8, dRm)
    Pmrun(Pm, c1, c2, beta, dQ, dRm, Qsearch, Rmsearch)
    
