"""finds the critical Renoylds number and wave number for the
widegap MRI eigenvalue equation.

"""
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import time
import dedalus.public as de
import numpy as np
import matplotlib.pylab as plt

comm = MPI.COMM_WORLD


# Reasonable parameters

# URM07-like
# nr = 100
# R1 = 5
# R2 = 15
# Omega1 = 313.55
# Omega2 = 67.0631
# beta = 25.0
# k = 0.748
# Rm = 4.879
# Pm = 1.0E-3

# Goodman & Ji 2001-like
# nr = 256
#Pm = 1.6E-6
#beta = 0.43783886002604167#25.0
#R1 = 5
#R2 = 15
#Omega1 = 313.55
#Omega2 = 37.9
#k = np.pi/10
#Rm = 4.052


def find_crit(R1, R2, Omega1, Omega2, beta, Pm, Rm_min, Rm_max, k_min, k_max, n_Rm, n_k, nr):
    Rm = Rm_min
    k = k_min
    R = Rm/Pm
    iR = 1.0/R

    c1 = (Omega2*R2**2 - Omega1*R1**2)/(R2**2 - R1**2)
    c2 = (R1**2*R2**2*(Omega1 - Omega2))/(R2**2 - R1**2)

    zeta_mean = 2*(R2**2*Omega2 - R1**2*Omega1)/((R2**2 - R1**2)*np.sqrt(Omega1*Omega2))
    if comm.rank == 0:
        print("mean zeta is {}, meaning q = 2 - zeta = {}".format(zeta_mean, 2 - zeta_mean))

    r = de.Chebyshev('r', nr, interval = (R1, R2))
    d = de.Domain([r],comm=MPI.COMM_SELF)

    widegap = de.EVP(d,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],'sigma')

    widegap.parameters['k'] = k
    widegap.parameters['iR'] = iR
    widegap.parameters['Rm'] = Rm # Rm rather than iRm so search options are more intuitive
    widegap.parameters['c1'] = c1
    widegap.parameters['c2'] = c2
    widegap.parameters['beta'] = beta
    widegap.parameters['B0'] = 1
    widegap.parameters['xi'] = 0

    widegap.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
    widegap.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
    widegap.substitutions['twooverbeta'] = '(2.0/beta)'
    widegap.substitutions['psivisc'] = '(2*r**2*k**2*psir - 2*r**3*k**2*psirr + r**3*k**4*psi + r**3*dr(psirrr) - 3*psir + 3*r*psirr - 2*r**2*psirrr)'
    widegap.substitutions['uvisc'] = '(-r**3*k**2*u + r**3*dr(ur) + r**2*ur - r*u)'
    widegap.substitutions['Avisc'] = '(r*dr(Ar) - r*k**2*A - Ar)' 
    widegap.substitutions['Bvisc'] = '(-r**3*k**2*B + r**3*dr(Br) + r**2*Br - r*B)'

    widegap.add_equation("sigma*(-r**3*k**2*psi + r**3*psirr - r**2*psir) - r**2*2*ru0*1j*k*u + r**3*twooverbeta*B0*1j*k**3*A + twooverbeta*B0*r**2*1j*k*Ar - twooverbeta*r**3*B0*1j*k*dr(Ar) - iR*psivisc + twooverbeta*r**2*2*xi*1j*k*B = 0") #corrected on whiteboard 5/6
    widegap.add_equation("sigma*r**3*u + 1j*k*ru0*psi + 1j*k*rrdu0*psi - 1j*k*r**3*twooverbeta*B0*B - iR*uvisc = 0") 
    widegap.add_equation("sigma*r*A - r*B0*1j*k*psi - (1/Rm)*Avisc = 0")
    widegap.add_equation("sigma*r**3*B + ru0*1j*k*A - r**3*B0*1j*k*u - 1j*k*rrdu0*A - (1/Rm)*Bvisc - 2*xi*1j*k*psi = 0") 

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

    # create an Eigenproblem object
    EP = Eigenproblem(widegap)

    # create a shim function to translate (x, y) to the parameters for the eigenvalue problem:
    def shim(x,y):
        gr, indx = EP.growth_rate({"k":x,"Rm":y})
        return gr

    cf = CriticalFinder(shim, comm)

    # generating the grid is the longest part
    start = time.time()
    cf.grid_generator(Rm_min, Rm_max, k_min, k_max, n_Rm, n_k) 
    end = time.time()
    if comm.rank == 0:
        print("grid generation time: {:10.5f} sec".format(end-start))
        cf.save_grid('../../data/widegap_growth_rates_res{0:d}_Rmmin{1:5.02e}_Rmmax{2:5.02e}_kmin{3:5.02e}_kmax{4:5.02e}_nRm{5:5.02e}_nk{6:5.02e}'.format(nr,Rm_min,Rm_max, k_min, k_max, n_Rm, n_k))

    cf.root_finder()
    crit = cf.crit_finder()

    if comm.rank == 0:
        print("critical wavenumber k = {:10.5f}".format(crit[0]))
        print("critical Rm = {:10.5f}".format(crit[1]))
        title_str = '../../figs/widegap_growth_rates_res{0:d}_Rmmin{1:5.02e}_Rmmax{2:5.02e}_kmin{3:5.02e}_kmax{4:5.02e}_nRm{5:5.02e}_nk{6:5.02e}'.format(nr,Rm_min,Rm_max, k_min, k_max, n_Rm, n_k)
        cf.plot_crit(title = title_str, xlabel = r"$k_z$", ylabel = r"$\mathrm{Rm}$")
