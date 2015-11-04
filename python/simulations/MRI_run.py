"""
Dedalus script for 2D MRI simulations


Usage:
    MRI_run.py [--Rm=<Rm> --eps=<eps> --Pm=<Pm> --beta=<beta> --qsh=<qsh> --Omega0=<Omega0> --Q=<Q> --restart=<restart_file> --linear --nz=<nz>] 

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
"""
import logging
import os
import sys
import time 

import numpy as np
from docopt import docopt

# parse arguments
args = docopt(__doc__)

Rm = float(args['--Rm'])
eps = float(args['--eps'])
Pm  = float(args['--Pm'])
beta  = float(args['--beta'])
Q = float(args['--Q'])
qsh = float(args['--qsh'])
Omega0 = float(args['--Omega0'])
nz = int(args['--nz'])

nx = nz

restart = args['--restart']
linear = args['--linear']

# save data in directory named after script
data_dir = "scratch/" + sys.argv[0].split('.py')[0]
data_dir += "_Rm{0:5.02e}_eps{1:5.02e}_Pm{2:5.02e}_beta{3:5.02e}_Q{4:5.02e}_qsh{5:5.02e}_Omega{6:5.02e}_nz{7:d}/".format(Rm, eps, Pm, beta, Q, qsh, Omega0, nz)

from dedalus.tools.config import config

config['logging']['filename'] = os.path.join(data_dir,'dedalus_log')
config['logging']['file_level'] = 'DEBUG'

import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
logger = logging.getLogger(__name__)

# use checkpointing if available
try:
    from dedalus.extras.checkpointing import Checkpoint
    do_checkpointing=True
except ImportError:
    logger.warn("No Checkpointing module. Setting checkpointing to false.")
    do_checkpointing=False

from equations import MRI_equations
# configure MRI equations
MRI = MRI_equations(nx=nx, nz=nz, linear=linear)
MRI.set_parameters(Rm, Pm, eps, Omega0, qsh, beta, Q)
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

if do_checkpointing:
    checkpoint = Checkpoint(data_dir)
    checkpoint.set_checkpoint(solver, wall_dt=1800)

if restart is None:
    # Random perturbations, need to initialize globally
    gshape = MRI.domain.dist.grid_layout.global_shape(scales=MRI.domain.dealias)
    slices = MRI.domain.dist.grid_layout.slices(scales=MRI.domain.dealias)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    A0 = 1e-6

    # ICs
    psi = solver.state['psi']
    psi.set_scales(MRI.domain.dealias, keep_data=False)
    x = MRI.domain.grid(-1,scales=MRI.domain.dealias)
    psi['g'] = A0 * noise * np.sin(np.pi*x)
else:
    logger.info("restarting from {}".format(restart))
    checkpoint.restart(restart, solver)

period = 2*np.pi/Omega0

solver.stop_sim_time = 12.5*period
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

output_time_cadence = 0.1*period
analysis_tasks = MRI.initialize_output(solver, data_dir, iter=10)#sim_dt=output_time_cadence)

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("psi_x**2 + (dz(psi))**2 + u**2", name='Ekin')

#CFL = flow_tools.CFL(solver, initial_dt=1e-3, cadence=5, safety=0.3,
#                     max_change=1.5, min_change=0.5)
#CFL.add_velocities(('dz(psi)', '-psi_x'))

#dt = CFL.compute_dt()
dt = 1e-2

# Main loop
start_time = time.time()

while solver.ok:
    solver.step(dt)
    if (solver.iteration-1) % 10 == 0:
        logger.info('Iteration: %i, Time (in orbits): %e, dt: %e' %(solver.iteration, solver.sim_time/period, dt))
        logger.info('Max E_kin = %5.3e' %flow.max('Ekin'))
    #dt = CFL.compute_dt()


end_time = time.time()

# Print statistics
logger.info('Total time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
logger.info('Average timestep: %f' %(solver.sim_time/solver.iteration))

logger.info('beginning join operation')
if do_checkpointing:
    logger.info(data_dir+'/checkpoint/')
    post.merge_analysis(data_dir+'/checkpoint/')

for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)

