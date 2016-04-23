"""finds the critical Renoylds number and wave number for the
MRI eigenvalue equation.

"""
from dedalus.tools.config import config
#config['linear algebra']['use_baleig'] = 'True'


import matplotlib
matplotlib.use('Agg')
import sys
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import time

import dedalus.public as de
import numpy as np
import matplotlib.pylab as plt

comm = MPI.COMM_WORLD


# Define the MRI problem in Dedalus: 

x = de.Chebyshev('x',96)
d = de.Domain([x],comm=MPI.COMM_SELF)

mri = de.EVP(d,['ux','uy', 'uz', 'Bx', 'By', 'Bz', 'pi', 'uy_x','uz_x', 'By_x','Bz_x'],'sigma')


Rm = 4.879
Pm = 0.001
mri.parameters['q'] = 1.5
mri.parameters['beta'] = 25.0
mri.parameters['iR'] = Pm/Rm
mri.parameters['iRm'] = 1./Rm
mri.parameters['Q'] = 0.748
mri.parameters['B0'] = 1.
mri.parameters['Omega0'] = 1.

mri.add_equation("sigma*ux - 2*Omega0*uy + dx(pi) - B0*1j*Q*Bx*(2./beta) - iR * (-1j*Q*uz_x - Q**2 * ux) = 0")
mri.add_equation("sigma*uy + (2-q)*Omega0*ux - B0*1j*Q*By*(2./beta) - iR * (dx(uy_x) - Q**2 * uy) = 0")
mri.add_equation("sigma*uz + 1j*Q*pi - B0*1j*Q*Bz*(2./beta) - iR * (dx(uz_x) - Q**2 * uz) = 0")
mri.add_equation("sigma*Bx - B0*1j*Q*ux - iRm*(-1j*Q*Bz_x - Q**2*Bx) = 0")
mri.add_equation("sigma*By - B0*1j*Q*uy + q*Omega0*Bx - iRm*(dx(By_x) - Q**2*By) = 0")
mri.add_equation("sigma*Bz - B0*1j*Q*uz - iRm*(dx(Bz_x) - Q**2*Bz) = 0")
mri.add_equation("dx(ux) + 1j*Q*uz = 0")
mri.add_equation("uy_x - dx(uy) = 0")
mri.add_equation("uz_x - dx(uz) = 0")
mri.add_equation("By_x - dx(By) = 0")
mri.add_equation("Bz_x - dx(Bz) = 0")


mri.add_bc("left(ux) = 0")
mri.add_bc("right(ux) = 0")
mri.add_bc("left(uy) = 0")
mri.add_bc("right(uy) = 0")
mri.add_bc("left(uz) = 0")
mri.add_bc("right(uz) = 0")
mri.add_bc("left(Bx) = 0")
mri.add_bc("right(Bx) = 0")
mri.add_bc("left(By_x) = 0")
mri.add_bc("right(By_x) = 0")
# create an Eigenproblem object
EP = Eigenproblem(mri)

# gr = EP.growth_rate({})
# print("MRI corrected growth rate = {0:10.5e}".format(gr))
# EP.spectrum()
# EP.spectrum(spectype='good',title='spectrum_good')
# sys.exit()

# create a shim function to translate (x, y) to the parameters for the eigenvalue problem:
def shim(x,y):
    return EP.growth_rate({"Q":x,"iRm":1/y})

cf = CriticalFinder(shim, comm)

# generating the grid is the longest part
start = time.time()
cf.grid_generator(4.6,5.4,0.2,1.5,10,10)
end = time.time()
print("grid generation time: {:10.5f} sec".format(end-start))

if comm.rank == 0:
    cf.save_grid('mri_primative_growth_rates')

cf.root_finder()
crit = cf.crit_finder()

if comm.rank == 0:
    print("critical wavenumber alpha = {:10.5f}".format(crit[0]))
    print("critical Re = {:10.5f}".format(crit[1]))
    cf.plot_crit()
