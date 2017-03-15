"""
Dedalus script for 2D MRI simulations


Usage:
    MRI_run.py [--Rm=<Rm> --eps=<eps> --Pm=<Pm> --beta=<beta> --qsh=<qsh> --Omega0=<Omega0> --Q=<Q> --restart=<restart_file> --linear --nz=<nz> --nx=<nx> --Lz=<Lz> --stop=<stop> --stop_iter=<stop_iter> --use-CFL --evalue-IC --three-mode]

Options:
    --Rm=<Rm>                  magnetic Reynolds number [default: 4.8775]
    --eps=<eps>                epsilon [default: 0.5]
    --Pm=<Pm>                  magnetic Prandtl Number [default: 1e-3]
    --Q=<Q>                    vertical wavenumber  [default: 0.75]
    --qsh=<qsh>                shear parameter q [default: 1.5]
    --Omega0=<Omega0>          background rotation rate  [default: 1.]
    --beta=<beta>              plasma Beta parateter  [default: 25.]
    --restart=<restart_file>   restart from checkpoint
    --linear                   turn off non-linear terms
    --nz=<nz>                  vertical z (Fourier) resolution [default: 32]
    --nx=<nz>                  horizontal x (Chebyshev) resolution [default: 32]
    --Lz=<Lz>                  vertical length scale in units of 2 pi/Q, Q = critical wavenumber [default: 1]
    --stop=<stop>              stopping time in units of inner cylinder orbits [default: 200]
    --stop_iter=<stop_iter>    stopping iteration [default: inf]
    --use-CFL                  use CFL condition
    --evalue-IC                use linear eigenvalue as initial condition
    --three-mode               use three z modes as initial conditions
"""
import glob
import logging
import os
import re
import sys
import time 

import numpy as np
from docopt import docopt
from mpi4py import MPI
# parameters
filter_frac=0.25 #0.5

# parse arguments
args = docopt(__doc__)

Rm = float(args['--Rm'])
eps = float(args['--eps'])
Pm  = float(args['--Pm'])
beta  = float(args['--beta'])
Q = float(args['--Q'])
qsh = float(args['--qsh'])
Omega0 = float(args['--Omega0'])
nx = int(args['--nx'])
nz = int(args['--nz'])
Lz = int(args['--Lz'])
stop = float(args['--stop'])
stop_iter = args['--stop_iter']
if stop_iter == 'inf':
    stop_iter = np.inf
else:
    stop_iter = int(stop_iter)

restart = args['--restart']
linear = args['--linear']
CFL = args['--use-CFL']
evalue_IC = args['--evalue-IC']
three_mode = args['--three-mode']

# save data in directory named after script
data_dir = "scratch/" + sys.argv[0].split('.py')[0]
data_dir += "_Rm{0:5.02e}_eps{1:5.02e}_Pm{2:5.02e}_beta{3:5.02e}_Q{4:5.02e}_qsh{5:5.02e}_Omega{6:5.02e}_nx{7:d}_nz{8:d}_Lz{9:d}".format(Rm, eps, Pm, beta, Q, qsh, Omega0, nx, nz, Lz)

if linear:
    data_dir += "_linear"
if CFL:
    data_dir += "_CFL"
else:
    dt = 1e-4
    data_dir += "_dt{:5.02e}".format(dt)
if evalue_IC:
    data_dir += "_evalueIC"
    if three_mode:
        data_dir += "_threeMode"

if restart:
    restart_dirs = glob.glob(data_dir+"restart*")
    if restart_dirs:
        restart_dirs.sort()
        last = int(re.search("_restart(\d+)", restart_dirs[-1]).group(1))
        data_dir += "_restart{}".format(last+1)
    else:
        if os.path.exists(data_dir):
            data_dir += "_restart1"

from dedalus.tools.config import config

config['logging']['filename'] = os.path.join(data_dir,'dedalus_log')
config['logging']['file_level'] = 'DEBUG'

import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
logger = logging.getLogger(__name__)

from equations import MRI_equations
from filter_field import filter_field, smooth_filter_field
# configure MRI equations
MRI = MRI_equations(nx=nx, nz=nz, linear=linear)
MRI.set_parameters(Rm, Pm, eps, Omega0, qsh, beta, Q, Lz)
MRI.set_IVP_problem()
MRI.set_BC()
problem = MRI.problem

if MRI.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

        # write any hg diffs to a text file
        if MRI.hg_diff:
            diff_filename = os.path.join(data_dir,'diff.txt')
            with open(diff_filename,'w') as file:
                file.write(MRI.hg_diff)

logger.info("saving run in: {}".format(data_dir))

ts = de.timesteppers.RK443
solver= problem.build_solver(ts)

for k,v in problem.parameters.items():
    logger.info("problem parameter {}: {}".format(k,v))

if restart is None:
    A0 = 1e-3 # initial amplitude
    if evalue_IC:
        comm = MRI.domain.distributor.comm
        rank = comm.Get_rank()
        slices = MRI.domain.dist.grid_layout.slices(scales=(1,1))
        # solve linear eigenvalue problem for initial conditions only on rank 0
        if rank == 0:
            from allorders_2 import OrderE
            x = de.Chebyshev('x', nx, interval=[-1., 1.])
            lev_domain = de.Domain([x,],comm=MPI.COMM_SELF)
            lev = OrderE(lev_domain, Q=MRI.Q, Rm=MRI.Rm, Pm=MRI.Pm, q=MRI.q, beta=MRI.beta)
            attr_list = ["_".join([re.match("([^x]+)(x+$)",v).group(1),re.match("([^x]+)(x+$)",v).group(2)]) if re.match("([^x]+)(x+$)",v) else v for v in lev.lv1.variables]
            translation_table = dict(zip(MRI.variables,attr_list))
            growth_rate = np.max(lev.EP.evalues_good.real)
            logger.info("Expected growth rate: {0:5.2e}".format(growth_rate))
        for var in solver.state.field_names:
            if rank == 0:
                lev_field = getattr(lev,translation_table[var])
                data = lev_field['g']
            else:
                data = np.empty(nx,dtype=np.complex128)
            comm.Bcast(data,root=0)
            if three_mode:
                zfunction = np.exp(1j*Q*MRI.domain.grid(0)) + np.exp(1j*Q/2*MRI.domain.grid(0)) + np.exp(1j*3/2*Q*MRI.domain.grid(0))
            else:
                zfunction = np.exp(1j*Q*MRI.domain.grid(0))
            total_data = A0 * (data*zfunction).real
            solver.state[var]['g'] = total_data[slices]
    else:
        # Random perturbations, need to initialize globally
        gshape = MRI.domain.dist.grid_layout.global_shape(scales=MRI.domain.dealias)
        slices = MRI.domain.dist.grid_layout.slices(scales=MRI.domain.dealias)
        rand = np.random.RandomState(seed=42)
        noise = rand.standard_normal(gshape)[slices]
        
        # ICs
        psi = solver.state['psi']
        psi_x = solver.state['psi_x']
        psi_xx = solver.state['psi_xx']
        psi_xxx = solver.state['psi_xxx']
        psi.set_scales(MRI.domain.dealias, keep_data=False)
        x = MRI.domain.grid(-1,scales=MRI.domain.dealias)
        psi['g'] = A0 * noise * np.cos(np.pi*x/2.)
        if filter_frac != 1.: 
            #filter_field(psi,frac=filter_frac)
            smooth_filter_field(psi,frac=filter_frac)
        else:
            logger.warn("No filtering applied to ICs! This is probably bad!")

        psi.differentiate('x',out=psi_x)
        psi_x.differentiate('x',out=psi_xx)
        psi_xx.differentiate('x',out=psi_xxx)
else:
    logger.info("restarting from {}".format(restart))
    solver.load_state(restart,-1)

period = 2*np.pi/Omega0

solver.stop_sim_time = stop*period
solver.stop_wall_time = np.inf
solver.stop_iteration = stop_iter

analysis_tasks = MRI.initialize_output(solver, data_dir, period)

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("psi_x**2 + (dz(psi))**2 + u**2", name='Ekin')

if CFL:
    CFL = flow_tools.CFL(solver, initial_dt=1e-3, cadence=5, safety=0.3,
                         max_change=1.5, min_change=0.5)
    CFL.add_velocities(('dz(psi)', '-psi_x'))
    CFL.add_velocities(('dz(A)', '-A_x'))
    dt = CFL.compute_dt()

# Main loop
start_time = time.time()

while solver.ok:
    solver.step(dt)
    if (solver.iteration-1) % 10 == 0:
        logger.info('Iteration: %i, Time (in orbits): %e, dt: %e' %(solver.iteration, solver.sim_time/period, dt))
        logger.info('Max E_kin = %5.3e' %flow.max('Ekin'))
    if CFL:
        dt = CFL.compute_dt()


end_time = time.time()

# Print statistics
logger.info('Total wall time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
logger.info('Average timestep: %f' %(solver.sim_time/solver.iteration))

logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)

