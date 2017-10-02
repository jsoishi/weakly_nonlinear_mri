"""finds the critical Renoylds number and wave number for the
widegap MRI eigenvalue equation.

"""
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import time
import dedalus.public as de
import numpy as np
import matplotlib.pylab as plt
from scipy import special

from mpi4py import MPI
comm = MPI.COMM_WORLD

def hmri_eigenproblem(R1, R2, Omega1, Omega2, beta, xi, Pm, Rm, k, insulate, nr, sparse=True):
    Rm = Rm
    k = k
    R = Rm/Pm
    iR = 1.0/R

    c1 = (Omega2*R2**2 - Omega1*R1**2)/(R2**2 - R1**2)
    c2 = (R1**2*R2**2*(Omega1 - Omega2))/(R2**2 - R1**2)
    
    print("c1 = {}, c2 = {}".format(c1, c2))

    zeta_mean = 2*(R2**2*Omega2 - R1**2*Omega1)/((R2**2 - R1**2)*np.sqrt(Omega1*Omega2))
    if comm.rank == 0:
        print("mean zeta is {}, meaning q = 2 - zeta = {}".format(zeta_mean, 2 - zeta_mean))
        
    if insulate:
        magnetic_bcs = "insulating"
    else:
        magnetic_bcs = "conducting"

    r = de.Chebyshev('r', nr, interval = (R1, R2))
    d = de.Domain([r],comm=MPI.COMM_SELF)
    
    Co = 2.0/beta
    Re = Rm/Pm

    widegap = de.EVP(d,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],'sigma')

    widegap.parameters['k'] = k
    widegap.parameters['Rm'] = Rm # Rm rather than iRm so search options are more intuitive
    #widegap.parameters['iR'] = Pm/widegap.parameters['Rm']#1/Re#iR
    widegap.parameters['Pm'] = Pm
    widegap.parameters['c1'] = c1
    widegap.parameters['c2'] = c2
    widegap.parameters['beta'] = beta
    widegap.parameters['B0'] = 1
    widegap.parameters['xi'] = xi
    if magnetic_bcs == "insulating":
        widegap.parameters['bessel1'] = special.iv(0, k*R1)/special.iv(1, k*R1)
        widegap.parameters['bessel2'] = special.kn(0, k*R2)/special.kn(1, k*R2)

    widegap.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
    widegap.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
    widegap.substitutions['twooverbeta'] = '(2.0/beta)'
    widegap.substitutions['iR'] = '(Pm/Rm)'
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
    widegap.add_bc('left(psir) = 0')
    widegap.add_bc('right(psir) = 0')
    
    if magnetic_bcs == "conducting":
        widegap.add_bc('left(A) = 0')
        widegap.add_bc('right(A) = 0')
        widegap.add_bc('left(B + r*Br) = 0')
        widegap.add_bc('right(B + r*Br) = 0') # axial component of current = 0

    if magnetic_bcs == "insulating":
        #widegap.add_bc('left(dr(r*1j*k*A) - k*r*bessel1*1j*k*A) = 0')
        #widegap.add_bc('right(dr(r*1j*k*A) + k*r*bessel2*1j*k*A) = 0')
        widegap.add_bc('left(dr(A) - k*bessel1*A) = 0')
        widegap.add_bc('right(dr(A) + k*bessel2*A) = 0')
        widegap.add_bc('left(B) = 0')
        widegap.add_bc('right(B) = 0')
            


    return Eigenproblem(widegap, sparse=sparse)

def find_crit(comm, R1, R2, Omega1, Omega2, beta, xi, Pm, Rm_min, Rm_max, k_min, k_max, n_Rm, n_k, nr, insulate, sparse=True):

    EP = hmri_eigenproblem(R1, R2, Omega1, Omega2, beta, xi, Pm, Rm_min, k_min, insulate, nr, sparse=True)
    # create a shim function to translate (x, y) to the parameters for the eigenvalue problem:
    def shim(x,y):
        gr, indx,freq = EP.growth_rate({"k":x,"Rm":y, "bessel1":special.iv(0, x*R1)/special.iv(1, x*R1), "bessel2":special.kn(0, x*R2)/special.kn(1, x*R2)})
        #EP.spectrum(title='../figs/good_eigenvalue_spectrum_k_{}_Rm_{}'.format(k, Rm),spectype='good')
        print("Co = {}, Rm = {}, k = {}, Re = {}, Pm = {}, xi = {}".format(2/EP.EVP.parameters['beta'], EP.EVP.parameters['Rm'], EP.EVP.parameters['k'], EP.EVP.parameters['Pm']/EP.EVP.parameters['Rm'], Pm, EP.EVP.parameters['xi']))
        print("Growth rate: {} freq: {}".format(gr, freq))
        return gr+1j*freq

    cf = CriticalFinder(shim, comm)

    # generating the grid is the longest part
    start = time.time()
    cf.grid_generator(Rm_min, Rm_max, k_min, k_max, n_Rm, n_k) 
    end = time.time()
    if comm.rank == 0:
        print("grid generation time: {:10.5f} sec".format(end-start))
        if xi == 0:
            gridname = '../data/growth_rates_res{0:d}_Pm{1:5.02e}_Rmmin{2:5.02e}_Rmmax{3:5.02e}_kmin{4:5.02e}_kmax{5:5.02e}_nRm{6:5.02e}_nk{7:5.02e}_Omega1_{8:5.02e}_Omega2_{9:5.02e}_R1_{10:5.02e}_R2_{11:5.02e}'.format(nr,Pm, Rm_min,Rm_max, k_min, k_max, n_Rm, n_k, Omega1, Omega2, R1, R2)
        else:
            if magnetic_bcs == "conducting":
                gridname = '../data/hmri_growth_rates_res{0:d}_Pm{1:5.02e}_Rmmin{2:5.02e}_Rmmax{3:5.02e}_kmin{4:5.02e}_kmax{5:5.02e}_nRm{6:5.02e}_nk{7:5.02e}_Omega1_{8:5.02e}_Omega2_{9:5.02e}_R1_{10:5.02e}_R2_{11:5.02e}'.format(nr,Pm, Rm_min,Rm_max, k_min, k_max, n_Rm, n_k, Omega1, Omega2, R1, R2)
            elif magnetic_bcs == "insulating":
                gridname = '../data/hmri_growth_rates_res{0:d}_Pm{1:5.02e}_Rmmin{2:5.02e}_Rmmax{3:5.02e}_kmin{4:5.02e}_kmax{5:5.02e}_nRm{6:5.02e}_nk{7:5.02e}_Omega1_{8:5.02e}_Omega2_{9:5.02e}_R1_{10:5.02e}_R2_{11:5.02e}_insulating'.format(nr,Pm, Rm_min,Rm_max, k_min, k_max, n_Rm, n_k, Omega1, Omega2, R1, R2)
        
        cf.save_grid(gridname)
        
        # plot grid regardless of whether critical value is found
        if xi == 0:
            title_str = '../figs/widegap_growth_rates_Pm{0:.4f}_xi0_res{0:d}_Rmmin{1:5.02e}_Rmmax{2:5.02e}_kmin{3:5.02e}_kmax{4:5.02e}_nRm{5:5.02e}_nk{6:5.02e}'.format(Pm, nr,Rm_min,Rm_max, k_min, k_max, n_Rm, n_k)
            title_str_freq = '../figs/widegap_Im_sigma_Pm{0:.4f}_xi0_res{1:f}_Rmmin{2:5.02e}_Rmmax{3:5.02e}_kmin{4:5.02e}_kmax{5:5.02e}_nRm{6:5.02e}_nk{7:5.02e}_Co{8}'.format(Pm, nr,Rm_min,Rm_max, k_min, k_max, n_Rm, n_k, Co)
        
        else:
            title_str = '../figs/helical_growth_rates_Pm{0:.4f}_xi{1:d}_res{2:f}_Rmmin{3:5.02e}_Rmmax{4:5.02e}_kmin{5:5.02e}_kmax{6:5.02e}_nRm{7:5.02e}_nk{8:5.02e}_Co{9}'.format(Pm, xi, nr,Rm_min,Rm_max, k_min, k_max, n_Rm, n_k, Co)
            title_str_freq = '../figs/helical_Im_sigma_Pm{0:.4f}_xi{1:d}_res{2:f}_Rmmin{3:5.02e}_Rmmax{4:5.02e}_kmin{5:5.02e}_kmax{6:5.02e}_nRm{7:5.02e}_nk{8:5.02e}_Co{9}'.format(Pm, xi, nr,Rm_min,Rm_max, k_min, k_max, n_Rm, n_k, Co)
        cf.plot_crit(title = title_str, xlabel = r"$k_z$", ylabel = r"$\mathrm{Rm}$", plotroots=False)
        cf.plot_crit(title = title_str_freq, xlabel = r"$k_z$", ylabel = r"$\mathrm{Rm}$", plotroots=False, plotfreq=True)

    cf.root_finder()
    crit = cf.crit_finder(find_freq = True)

    if comm.rank == 0:
        print("crit = {}".format(crit))
        print("critical omega = {:10.5f}".format(crit[2]))
        print("critical wavenumber k = {:10.5f}".format(crit[0]))
        print("critical Rm = {:10.5f}".format(crit[1]))

    Q  = crit[0]
    Rmc = crit[1]
    omega = crit[2]
    return Q, Rmc, omega

"""
if __name__ == "__main__":

    #Pm1, beta1: 3.3    99.   2.685373   0.000384   0.006329
    #Pm1, beta0: 3.3    99.   2.688426   0.000572   0.000000
    nr = 64
    R1 = 1
    R2 = 2
    Omega1 = 1
    Omega2 = 0.25
    Ha = 3.3
    Re = 99.
    Pm = 1.0
    xi = 0#1.0
	
    Rm = Pm*Re
    beta = Re*Rm/Ha**2
	
    insulate = True
	
    Rm_min = Rm - 0.5*Rm
    Rm_max = Rm + 0.5*Rm
    print("Rm min is {}, Rm max is {}".format(Rm_min, Rm_max))
	
    k_min = 2.0
    k_max = 3.0
    
    n_Rm = 10
    n_k = 10
	
    Q, Rmc, omega = find_crit(comm, R1, R2, Omega1, Omega2, beta, xi, Pm, Rm_min, Rm_max, k_min, k_max, n_Rm, n_k, nr, insulate)

    print("critical Ha = sqrt(Re*Rm_c/beta) = {}".format(np.sqrt(Re*Rmc/beta)))
"""
