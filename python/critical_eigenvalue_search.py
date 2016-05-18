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

def min_evalue(kz):
    """Calculate smallest e-value for linear growth."""

    # Create bases and domain
    # Use COMM_SELF so keep calculations independent between processes
    nr = 64
    r_basis = de.Chebyshev('r', nr, interval=(5, 15))
    domain = de.Domain([r_basis], comm=MPI.COMM_SELF)
    
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
    widegap = de.EVP(domain, variables = ['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'], eigenvalue = 'iRm', tolerance = 1E-10)
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
    solver = widegap.build_solver()
    solver.solve(solver.pencils[0])

    # Return smallest real part
    ev = solver.eigenvalues
    ev = ev[np.isfinite(ev)]
    i_min = np.argmin(np.abs(ev.real))
    print(repr(ev.real[i_min]))
    return np.min(ev.real[i_min])

def min_evalue2(Rm, kz = None):
    """Calculate smallest e-value for linear growth."""

    # Create bases and domain
    # Use COMM_SELF so keep calculations independent between processes
    nr = 64
    r_basis = de.Chebyshev('r', nr, interval=(5, 15))
    domain = de.Domain([r_basis], comm=MPI.COMM_SELF)
    
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
    widegap = de.EVP(domain, variables = ['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'], eigenvalue = 'sigma', tolerance = 1E-10)
    widegap.parameters['k'] = kz
    widegap.parameters['c1'] = c1
    widegap.parameters['c2'] = c2
    widegap.parameters['beta'] = beta
    widegap.parameters['Pm'] = Pm
    widegap.parameters['iRm'] = 1.0/Rm
    widegap.parameters['B0'] = 1

    widegap.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
    widegap.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
    widegap.substitutions['twooverbeta'] = '(2.0/beta)'
    widegap.substitutions['psivisc'] = '(2*r**2*k**2*psir - 2*r**3*k**2*psirr + r**3*k**4*psi + r**3*dr(psirrr) - 3*psir + 3*r*psirr - 2*r**2*psirrr)'
    widegap.substitutions['uvisc'] = '(-r**3*k**2*u + r**3*dr(ur) + r**2*ur - r*u)'
    widegap.substitutions['Avisc'] = '(r*dr(Ar) - r*k**2*A - Ar)' # checked on whiteboard 5/6
    widegap.substitutions['Bvisc'] = '(-r**3*k**2*B + r**3*dr(Br) + r**2*Br - r*B)'
    
    # widegap MRI equations with sigma (growth rate) set to zero. Note iR = (Pm/Rm)
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
        
    # Solve for eigenvalues
    solver = widegap.build_solver()
    solver.solve(solver.pencils[0])

    # Return smallest real part
    ev = solver.eigenvalues
    ev = ev[np.isfinite(ev)]
    i_min = np.argmin(np.abs(ev.real))
    print(repr(ev.real[i_min]))
    return np.min(ev.real[i_min])

sol = opt.minimize_scalar(min_evalue, bracket=[0.01, 0.04])
print(repr(sol.x))

kz_critical = sol.x

sol2 = opt.minimize_scalar(min_evalue2(args = (kz_critical,)), bracket=[0.83, 0.85])
print(repr(sol2.x))

