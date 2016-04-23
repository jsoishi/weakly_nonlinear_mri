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

mri = de.EVP(d,['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],'sigma')


Rm = 4.879
Pm = 0.001
mri.parameters['q'] = 1.5
mri.parameters['beta'] = 25.0
mri.parameters['iR'] = Pm/Rm
mri.parameters['iRm'] = 1./Rm
mri.parameters['Q'] = 0.748

mri.add_equation("sigma*psixx - Q**2*sigma*psi - iR*dx(psixxx) + 2*iR*Q**2*psixx - iR*Q**4*psi - 2*1j*Q*u - (2/beta)*1j*Q*dx(Ax) + (2/beta)*Q**3*1j*A = 0")
mri.add_equation("sigma*u - iR*dx(ux) + iR*Q**2*u - (q - 2)*1j*Q*psi - (2/beta)*1j*Q*B = 0") 
mri.add_equation("sigma*A - iRm*dx(Ax) + iRm*Q**2*A - 1j*Q*psi = 0") 
mri.add_equation("sigma*B - iRm*dx(Bx) + iRm*Q**2*B - 1j*Q*u + q*1j*Q*A = 0")

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
cf.grid_generator(4.6,5.4,0.1,1.5,10,10)
end = time.time()
print("grid generation time: {:10.5f} sec".format(end-start))

if comm.rank == 0:
    cf.save_grid('mri_growth_rates')

cf.root_finder()
crit = cf.crit_finder()

if comm.rank == 0:
    print("critical wavenumber alpha = {:10.5f}".format(crit[0]))
    print("critical Re = {:10.5f}".format(crit[1]))
    cf.plot_crit()
