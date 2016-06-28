import time
import numpy as np
import h5py
import dedalus.public as de
from mpi4py import MPI
from scipy import optimize as opt
comm = MPI.COMM_WORLD

from allorders_2 import AmplitudeAlpha
from find_crit import find_crit

import logging
logger = logging.getLogger(__name__)

res = 50
x = de.Chebyshev('x',res)
domain = de.Domain([x], np.complex128, comm=MPI.COMM_SELF)
logger.info("running at resolution {}".format(res))

q = 1.5
beta = 25.0
npoints = 4
global_Pm = [5e-4,1e-3,3e-3,5e-3] #np.logspace(-5,-3.5, npoints)

nproc = comm.size
rank = comm.rank

Pm_split = np.array_split(global_Pm, nproc)
local_Pm = Pm_split[rank]

local_size = local_Pm.size 

coeffs = {'a':np.empty(local_size,dtype='complex128'),
          'b': np.empty(local_size,dtype='complex128'),
          'c': np.empty(local_size,dtype='complex128'),
          'ctwiddle': np.empty(local_size,dtype='complex128'),
          'h': np.empty(local_size,dtype='complex128'),
          'Q_c': np.empty(local_size,dtype='complex128'),
          'Rm_c': np.empty(local_size,dtype='complex128')}

global_coeffs = {'a':np.empty(npoints,dtype='complex128'),
                 'b': np.empty(npoints,dtype='complex128'),
                 'c': np.empty(npoints,dtype='complex128'),
                 'ctwiddle': np.empty(npoints,dtype='complex128'),
                 'h': np.empty(npoints,dtype='complex128'),
                 'Q_c': np.empty(npoints,dtype='complex128'),
                 'Rm_c': np.empty(npoints,dtype='complex128')}

for i,Pm in enumerate(local_Pm):
    Q_c, Rm_c = find_crit(domain, Pm, q, beta)
    aa = AmplitudeAlpha(domain, Q = Q_c, Rm = Rm_c, Pm = Pm, q = q, beta = beta)
    coeffs['a'][i] = aa.a
    coeffs['b'][i] = aa.b
    coeffs['c'][i] = aa.c
    coeffs['ctwiddle'][i] = aa.ctwiddle
    coeffs['h'][i] = aa.h
    coeffs['Q_c'][i] = Q_c
    coeffs['Rm_c'][i] = Rm_c

rec_counts = [i.size for i in Pm_split]
displacements = np.cumsum(rec_counts) - rec_counts
if rank == 0:
    logger.info("Rec_counts = {}; displacements = {}".format(rec_counts,displacements))

keys = list(coeffs.keys())
keys.sort()
for k in keys:
    comm.Gatherv(coeffs[k],[global_coeffs[k],rec_counts,displacements,MPI.F_DOUBLE_COMPLEX],root=0)

if rank == 0:
    outfile = h5py.File("../data/pm_sat_coeffs.h5","w")
    for k in keys:
        outfile.create_dataset(k,data=global_coeffs[k])
    outfile.create_dataset('Pm',data=global_Pm)
    outfile.close()
