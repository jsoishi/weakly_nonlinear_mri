import time
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy import optimize as opt
CW = MPI.COMM_WORLD

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

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

def min_evalue(kz):
    """Calculate smallest e-value for linear growth."""

    # Create bases and domain
    # Use COMM_SELF so keep calculations independent between processes
    nr = 64
    r_basis = de.Chebyshev('r', nr, interval=(5, 15))
    domain1 = de.Domain([r_basis], comm=MPI.COMM_SELF)
    
    nr = 96
    r_basis = de.Chebyshev('r', nr, interval=(5, 15))
    domain2 = de.Domain([r_basis], comm=MPI.COMM_SELF)
    
    # Parameters approximating Umurhan+ 2007
    Pm = 1.0E-3
    beta = 25.0
    R1 = 5
    R2 = 15
    Omega1 = 313.55
    Omega2 = 67.0631
    
    c1 = (Omega2*R2**2 - Omega1*R1**2)/(R2**2 - R1**2)
    c2 = (R1**2*R2**2*(Omega1 - Omega2))/(R2**2 - R1**2)

    # Widegap MRI equations
    widegap1 = de.EVP(domain1, variables = ['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'], eigenvalue = 'iRm', tolerance = 1E-10)
    widegap2 = de.EVP(domain2, variables = ['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'], eigenvalue = 'iRm', tolerance = 1E-10)
    
    for widegap in [widegap1, widegap2]:
        widegap.parameters['k'] = kz
        widegap.parameters['c1'] = c1
        widegap.parameters['c2'] = c2
        widegap.parameters['beta'] = beta
        widegap.parameters['Pm'] = Pm
        widegap.parameters['B0'] = 1

        widegap.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
        widegap.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
        widegap.substitutions['twooverbeta'] = '(2.0/beta)'
        widegap.substitutions['psivisc'] = '(2*r**2*k**2*psir - 2*r**3*k**2*psirr + r**3*k**4*psi + r**3*dr(psirrr) - 3*psir + 3*r*psirr - 2*r**2*psirrr)'
        widegap.substitutions['uvisc'] = '(-r**3*k**2*u + r**3*dr(ur) + r**2*ur - r*u)'
        widegap.substitutions['Avisc'] = '(r*dr(Ar) - r*k**2*A - Ar)' # checked on whiteboard 5/6
        widegap.substitutions['Bvisc'] = '(-r**3*k**2*B + r**3*dr(Br) + r**2*Br - r*B)'
    
        # widegap MRI equations with sigma (growth rate) set to zero. Note iR = (Pm/Rm)
        widegap.add_equation("-r**2*2*ru0*1j*k*u + r**3*twooverbeta*B0*1j*k**3*A + twooverbeta*B0*r**2*1j*k*Ar - twooverbeta*r**3*B0*1j*k*dr(Ar) - Pm*iRm*psivisc = 0") #corrected on whiteboard 5/6
        widegap.add_equation("1j*k*ru0*psi + 1j*k*rrdu0*psi - 1j*k*r**3*twooverbeta*B0*B - Pm*iRm*uvisc = 0") # correct 5/5
        widegap.add_equation("-r*B0*1j*k*psi - iRm*Avisc = 0")
        widegap.add_equation("ru0*1j*k*A - r**3*B0*1j*k*u - 1j*k*rrdu0*A - iRm*Bvisc = 0") # correct 5/5

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
        
    # Solve for eigenvalues
    solver1 = widegap1.build_solver()
    solver2 = widegap2.build_solver()
    
    solver1.solve(solver1.pencils[0])
    solver2.solve(solver2.pencils[0])
    
    # Discard spurious eigenvalues
    ev1 = solver1.eigenvalues
    ev2 = solver2.eigenvalues
    goodeigs, goodeigs_indices = discard_spurious_eigenvalues(ev1, ev2)

    # Return smallest real part
    goodeigs_index = np.nanargmin(goodeigs.real)
    marginal_mode_index = int(goodeigs_indices[goodeigs_index])

    # Largest real-part eigenvalue
    critical_eval = goodeigs[goodeigs_index]
    print(critical_eval)
    
    return critical_eval


sol = opt.minimize_scalar(min_evalue, bracket=[0.01, 0.04])
print(repr(sol.x))

