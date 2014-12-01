import numpy as np
import time
import os
import sys
#import checkpointing

import logging
logger = logging.getLogger(__name__)

import dedalus2.public as de
from dedalus2.tools import post
from dedalus2.extras import flow_tools

initial_time = time.time()
Ampl0 = 0.07
epsilon = 0.5
Q = 0.75 # wavenumber
Lx = 1.
Lz = 2*np.pi/Q
nx = 32 # 256
nz = nx
x_basis = de.Chebyshev(nx)
z_basis = de.Fourier(nz)
domain = de.Domain([z_basis,x_basis])

mri = de.ParsedProblem(['z','x'],
                       field_names=['psi','psi_x','psi_xx','psi_xxx'],
                       param_names=['Re','Rm','B0','Omega0','q','beta'])

# Parameters
Pm = 0.001
mri.parameters['B0'] = 1. - epsilon**2
mri.parameters['Rm'] = 4.8775
mri.parameters['Re'] = mri.parameters['Rm']/Pm
mri.parameters['q'] = 1.5
mri.parameters['beta'] = 25.0
mri.parameters['Omega0'] = 1.

#streamfunction
mri.add_equation("dt(dx(psi_x)) + dz(dz(dt(psi))) - (dx(psi_xxx) + dz(dz(dz(dz(psi)))))/Re = 0")

# first-order scheme definitions
mri.add_equation("psi_x - dx(psi) = 0")
mri.add_equation("psi_xx - dx(psi_x) = 0")
mri.add_equation("psi_xxx - dx(psi_xx) = 0")

# four boundary conditions
mri.add_left_bc("psi = 0")#,condition="dz == 0")
mri.add_right_bc("psi = 0")#,condition="dz == 0")
mri.add_left_bc("psi_x = 0")
mri.add_right_bc("psi_x = 0")

# prepare
mri.expand(domain)

# time stepper
ts = de.timesteppers.RK443

# build solver
solver = de.solvers.IVP(mri, domain, ts)
solver.stop_sim_time = np.inf
solver.stop_iteration = 2 #np.inf
solver.stop_wall_time = 24*3600. # run for 24 hours

# initial conditions
psi = solver.state['psi']

# noise
x = domain.grid(1)
z = domain.grid(0)
psi['g'] = Ampl0 * np.sin(np.pi*x/Lx)*np.random.rand(*psi['g'].shape)

# analysis
data_dir = "data"
output_time_cadence= 0.01
output_iter_cadence = 1
analysis_slice = solver.evaluator.add_file_handler(os.path.join(data_dir,"slices"), iter=output_iter_cadence, sim_dt=output_time_cadence, parallel=False)

analysis_slice.add_task("psi", name="psi")

dt = 0.001
start_time = time.time()
while solver.ok:
    logger.info("iteration = %i" %solver.iteration)
    solver.step(dt)
end_time = time.time()
    
if (domain.distributor.rank==0):

    N_TOTAL_CPU = domain.distributor.comm_world.size
    
    # Print statistics
    print('-' * 40)
    total_time = end_time-initial_time
    main_loop_time = end_time - start_time
    startup_time = start_time-initial_time
    print('  startup time:', startup_time)
    print('main loop time:', main_loop_time)
    print('    total time:', total_time)
    print('Iterations:', solver.iteration)
    print('Average timestep:', solver.sim_time / solver.iteration)
    print('scaling:',
          ' {:d} {:d} {:d} {:d} {:d} {:d}'.format(N_TOTAL_CPU, 0, N_TOTAL_CPU,nx, 0, nz),
          ' {:8.3g} {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(startup_time,
                                                            main_loop_time, 
                                                            main_loop_time/solver.iteration, 
                                                            main_loop_time/solver.iteration/(nx*nz), 
                                                            N_TOTAL_CPU*main_loop_time/solver.iteration/(nx*nz)))
    print('-' * 40)