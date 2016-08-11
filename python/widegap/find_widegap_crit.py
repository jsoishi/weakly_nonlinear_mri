"""Driver to find critical Rm, k_z for widegap MRI 

This will solve critical parameters for n_Pm values of the magnetic
Prandtl number, log spaced between Pm_min and Pm_max. 

For optimal performance, the code should be run on n_Pm * n_k * n_r
processors. It will make n_Pm groups of n_k * n_r processors, each of
which will launch a CriticalFinder instance and collectively find the
critical parameters for that Pm.

Usage:
    find_widegap_crit.py [--R1=<R1> --R2=<R2> --Omega1=<Omega1> --Omega2=<Omega2> --n_Pm=<n_Pm>  --Pm_min=<Pm_min> --Pm_max=<Pm_max> --beta=<beta> --xi=<xi> --Rm_min=<Rm_min> --Rm_max=<Rm_max> --k_min=<k_min> --k_max=<k_max> --n_Rm=<n_Rm> --n_k=<n_k> --n_r=<n_r> --insulate --log]

Options:

    --R1=<R1>                  inner cylinder Radius (in cm) [default: 5.]
    --R2=<R2>                  outer cylinder Radius (in cm) [default: 15.]
    --Omega1=<Omega1>          inner cylinder rotation rate (units of 1/s) [default: 313.55]
    --Omega2=<Omega2>          outer cylinder rotation rate (units of 1/s) [default: 67.0631]
    --n_Pm=<Pm>                number of magnetic Prandtl numbers [default: 20]
    --Pm_min=<Rm_min>          minimum magnetic Prandtl number [default: 1e-4]
    --Pm_max=<Rm_max>          maximum magnetic Prandtl number [default: 3e-3]
    --beta=<beta>              plasma beta [default: 25.]
    --xi=<xi>                  helical base field strength for HMRI [default: 0.]
    --Rm_min=<Rm_min>          minimum magnetic Reynolds number [default: 0.5]
    --Rm_max=<Rm_max>          maximum magnetic Reynolds number [default: 2.0]
    --k_min=<k_min>            minimum z wavenumber [default: 0.001]
    --k_max=<k_max>            maximum z wavenumber [default: 0.2]
    --n_Rm=<n_Rm>              number of points on Rm grid [default: 20]
    --n_k=<n_k>                number of points on k grid [default: 20]
    --n_r=<n_r>                number of points on Chebyshev r grid for eigenvalue solution [default: 100]
    --log                      if true, interpret Pm_min, Pm_max as log values (e.g. -3 for 10^-3)
    --insulate                 if true, insulating boundary conditions
"""
import numpy as np
from mpi4py import MPI
from docopt import docopt
from find_crit import find_crit
import h5py

import logging
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD

# parse arguments
args = docopt(__doc__)

R1 = float(args['--R1'])
R2 = float(args['--R2'])
Omega1 = float(args['--Omega1'])
Omega2 = float(args['--Omega2'])
Pm_min = float(args['--Pm_min'])
Pm_max = float(args['--Pm_max'])
n_Pm = int(args['--n_Pm'])
beta = float(args['--beta'])
xi = float(args['--xi'])
Rm_min = float(args['--Rm_min'])
Rm_max = float(args['--Rm_max'])
k_min = float(args['--k_min'])
k_max = float(args['--k_max'])
n_Rm = int(args['--n_Rm'])
n_k = int(args['--n_k'])
n_r = int(args['--n_r'])
log = args['--log']
insulate = args['--insulate']

if log:
    global_Pm = np.logspace(Pm_min,Pm_max,n_Pm,endpoint=False)
else:
    global_Pm = np.logspace(np.log10(Pm_min),np.log10(Pm_max),n_Pm,endpoint=False)
Pm_split = np.array_split(global_Pm, n_Pm)

color = int(n_Pm * comm.rank/comm.size)
crit_finder_comm = comm.Split(color,comm.rank)
local_Pm = Pm_split[color]

logger.info("Pm: {:d} steps from {:5.2e} to {:5.2e}".format(n_Pm, global_Pm[0], global_Pm[-1]))

wgroup = comm.Get_group()

skip = int(comm.size/n_Pm)
coll_ranks = np.arange(comm.size)[::skip]
coll_group = wgroup.Incl(coll_ranks)
coll_comm = comm.Create_group(coll_group)

local_Qc = np.empty(len(local_Pm))
local_Rmc = np.empty(len(local_Pm))

for i, Pm in enumerate(local_Pm):
    Q_c, Rm_c, gamma = find_crit(crit_finder_comm, R1, R2, Omega1, Omega2, beta, xi, Pm, Rm_min, Rm_max, k_min, k_max, n_Rm, n_k, n_r, insulate)
    local_Qc[i] = Q_c
    local_Rmc[i] = Rm_c

global_Qc = None
global_Rmc = None
if comm.rank == 0:
    global_Qc = np.empty(n_Pm)
    global_Rmc = np.empty(n_Pm)

rec_counts = [i.size for i in Pm_split]
displacements = np.cumsum(rec_counts) - rec_counts

if coll_comm:
    coll_comm.Gatherv(local_Qc,[global_Qc,rec_counts,displacements,MPI.DOUBLE],root=0)
    coll_comm.Gatherv(local_Rmc,[global_Rmc,rec_counts,displacements,MPI.DOUBLE],root=0)

if comm.rank == 0:
    outfile = h5py.File("../../data/widegap_crit_Rm_Q.h5","w")
    outfile.create_dataset('Pm',data=global_Pm)
    outfile.create_dataset('Q_c',data=global_Qc)
    outfile.create_dataset('Rm_c',data=global_Rmc)
    outfile.close()
