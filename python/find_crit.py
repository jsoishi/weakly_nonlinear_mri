import time
import numpy as np
from mpi4py import MPI
from scipy import optimize as opt
CW = MPI.COMM_WORLD

from dedalus import public as de
from eigentools import Eigenproblem

import logging
logger = logging.getLogger(__name__)

def find_crit(domain, Pm, q, beta):
    """find critical parmaeters for the narrow gap MRI

    inputs
    ------
    Pm : magnetic prandtl number
    q  : shear parameter
    res: (optional) resolution 
    """

    Q = 0.75 # initial guess

    mri = de.EVP(domain,['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],'iRm')

    mri.parameters['Q'] = Q
    mri.parameters['Pm'] = Pm
    mri.parameters['q'] = q
    mri.parameters['beta'] = beta

    mri.add_equation("iRm*(-Pm*dx(psixxx) + 2*Pm*Q**2*psixx - Pm*Q**4*psi) - 2*1j*Q*u - (2/beta)*1j*Q*dx(Ax) + (2/beta)*Q**3*1j*A = 0")
    mri.add_equation("iRm*(-Pm*dx(ux) + Pm*Q**2*u) - (q - 2)*1j*Q*psi - (2/beta)*1j*Q*B = 0") 
    mri.add_equation("-iRm*dx(Ax) + iRm*Q**2*A - 1j*Q*psi = 0") 
    mri.add_equation("-iRm*dx(Bx) + iRm*Q**2*B - 1j*Q*u + q*1j*Q*A = 0")

    mri.add_equation("dx(psi) - psix = 0")
    mri.add_equation("dx(psix) - psixx = 0")
    mri.add_equation("dx(psixx) - psixxx = 0")
    mri.add_equation("dx(u) - ux = 0")
    mri.add_equation("dx(A) - Ax = 0")
    mri.add_equation("dx(B) - Bx = 0")

    mri.add_bc("left(u) = 0")
    mri.add_bc("right(u) = 0")
    mri.add_bc("left(psi) = 0")
    mri.add_bc("right(psi) = 0")
    mri.add_bc("left(A) = 0")
    mri.add_bc("right(A) = 0")
    mri.add_bc("left(psix) = 0")
    mri.add_bc("right(psix) = 0")
    mri.add_bc("left(Bx) = 0")
    mri.add_bc("right(Bx) = 0")

    # create an Eigenproblem object
    EP = Eigenproblem(mri)

    # create a shim function to translate (x, y) to the parameters for the eigenvalue problem:
    def shim(kz):
        gr = EP.growth_rate({"Q":kz})
        print(1./gr[0])
        return 1./gr[0]


    sol = opt.minimize_scalar(shim, bracket=[0.74, 0.76],options={'xtol':1e-4,'maxiter':5})
    q_c = sol.x
    Rm_c = sol.fun

    return q_c, Rm_c

if __name__ == "__main__":
    #Pm = 1e-6
    Pm=1.0E-1
    q = 1.5
    beta = 25.0
    
    gridnum = 64 
    x_basis = de.Chebyshev('x',gridnum)
    domain = de.Domain([x_basis], np.complex128, comm=MPI.COMM_SELF)
    print("running at gridnum", gridnum)

    q_c, Rm_c = find_crit(domain, Pm, q, beta)
    print("Re_c = {:10.5e}".format(Rm_c))
    print("q_c = {:10.5e}".format(q_c))
