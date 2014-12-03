import numpy as np
import time
import os
import sys
import pickle
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
Lx = 2.
Lz = 2*np.pi/Q
nx = 32 # 256
nz = nx
x_basis = de.Chebyshev(nx)
z_basis = de.Fourier(nz,interval=(0,Lz))
domain = de.Domain([z_basis,x_basis])

mri = de.ParsedProblem(['z','x'],
                       field_names=['psi','psi_x','psi_xx','psi_xxx', 'u', 'u_x', 'A', 'A_x', 'b', 'b_x'],
                       param_names=['Re','Rm','B0','Omega0','q','beta'])
# Parameters
Pm = 0.001
mri.parameters['B0'] = 1. - epsilon**2
mri.parameters['Rm'] = 10 #4.8775
mri.parameters['Re'] = mri.parameters['Rm']/Pm
mri.parameters['q'] = 1.5
mri.parameters['beta'] = 25.0
mri.parameters['Omega0'] = 1.

#streamfunction
mri.add_equation("dt(dx(psi_x)) + dz(dz(dt(psi))) - 2*Omega0*dz(u) - (dx(psi_xxx) + dz(dz(dz(dz(psi)))))/Re - 2*B0/beta*(dz(dx(A_x)) + dz(dz(dz(A)))) = 2/beta*((dx(dx(A_x)) + dz(dz(A_x))) * dz(A) - (dz(dx(A_x)) + dz(dz(dz(A))))*A_x) - ((psi_xxx + dz(dz(A_x))) * dz(psi) - (dz(psi_xx) + dz(dz(dz(psi))))*psi_x)")

#u (y-velocity)
mri.add_equation("dt(u) + (2-q)*Omega0*dz(psi) - 2*B0/beta * dz(b) - (dx(u_x) + dz(dz(u)))/Re = 2./beta*(dz(A) * b_x - A_x * dz(b))")

#vector potential
mri.add_equation("dt(A) - B0 * dz(psi) - (dx(A_x) + dz(dz(A)))/Rm = dz(A) * psi_x - A_x * dz(psi)")

#b (y-field)
mri.add_equation("dt(b) - B0*dz(u) + q*Omega0 * dz(A) - (dx(b_x) + dz(dz(b)))/Rm = dz(A) * u_x - A_x * dz(u) - (b_x*dz(psi) - dz(b)*psi_x)")

# first-order scheme definitions
mri.add_equation("psi_x - dx(psi) = 0")
mri.add_equation("psi_xx - dx(psi_x) = 0")
mri.add_equation("psi_xxx - dx(psi_xx) = 0")
mri.add_equation("u_x - dx(u) = 0")
mri.add_equation("A_x - dx(A) = 0")
mri.add_equation("b_x - dx(b) = 0")

# ten boundary conditions
mri.add_left_bc("psi = 0")#,condition="dz == 0")
mri.add_right_bc("psi = 0")#,condition="dz == 0")
mri.add_left_bc("psi_x = 0")
mri.add_right_bc("psi_x = 0")
mri.add_left_bc("u = 0")
mri.add_right_bc("u = 0")
#mri.add_left_bc("dz(A) = 0", condition="dz != 0")
mri.add_left_bc("A = 0")#, condition="dz == 0")
mri.add_right_bc("A = 0")
mri.add_left_bc("dx(b) = 0")
mri.add_right_bc("dx(b) = 0")

# prepare
mri.expand(domain)

# time stepper
ts = de.timesteppers.RK443

t_orb = 2*np.pi/mri.parameters['Omega0']

# build solver
solver = de.solvers.IVP(mri, domain, ts)
solver.stop_sim_time = 3*t_orb
solver.stop_iteration = np.inf
solver.stop_wall_time = 24*3600. # run for 24 hours

# initial conditions
psi = solver.state['psi']
psi_x =  solver.state['psi_x']
psi_xx = solver.state['psi_xx']
psi_xxx = solver.state['psi_xxx']
u_x = solver.state['u_x'] 
A_x = solver.state['A_x']
b_x = solver.state['b_x']
u = solver.state['u']
b = solver.state['b']
A = solver.state['A']

# noise
# x = domain.grid(1)
# z = domain.grid(0)
# psi['g'] = 1e-4 * Ampl0 * np.cos(np.pi*x/Lx)*np.sin(2*np.pi*z/Lz)*np.random.rand(*psi['g'].shape)

# load in evalues
ic_filename = os.path.expanduser("/home/jsoishi/Downloads/coeffs_gridnum32_Pm_0.001_Q_0.75_Rm_4.8775_q_1.5_beta_25.0.p")
data = pickle.load(open(ic_filename,"rb"))

local_layout = domain.dist.grid_layout.slices
psi['g'] = data['Psi first order'][local_layout].real
u['g'] = data['u_y first order'][local_layout].real
A['g'] = data['A first order'][local_layout].real
b['g'] = data['B_y first order'][local_layout].real

# set up first order scheme
u.differentiate(1,u_x)
psi.differentiate(1,psi_x)
psi_x.differentiate(1,psi_xx)
psi_xx.differentiate(1,psi_xxx)
A.differentiate(1,A_x)
b.differentiate(1,b_x)

# analysis
data_dir = "data"
output_time_cadence= t_orb/2.
output_iter_cadence = 1000
ts_time_cadence = t_orb/100.

# slices
analysis_slice = solver.evaluator.add_file_handler(os.path.join(data_dir,"slices"), iter=output_iter_cadence, sim_dt=output_time_cadence, parallel=False)
analysis_slice.add_task("psi", name="psi")
analysis_slice.add_task("u", name="u")
analysis_slice.add_task("A", name="A")
analysis_slice.add_task("b", name="b")

# time series
analysis_ts = solver.evaluator.add_file_handler(os.path.join(data_dir,"time_series"), sim_dt=ts_time_cadence, parallel=False)
analysis_ts.add_task("Integrate(0.5 * (u*u + psi_x*psi_x + dz(psi)*dz(psi)))", name="total kinetic energy")
analysis_ts.add_task("Integrate(0.5 * (u*u))", name="toridal kinetic energy")
analysis_ts.add_task("Integrate(0.5 * (b*b + A_x*A_x + dz(A)*dz(A)))", name="total magnetic energy")
analysis_ts.add_task("Integrate(0.5 * (b*b))", name="toridal magnetic energy")

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
