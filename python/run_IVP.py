import dedalus.public as de
import numpy as np
from mpi4py import MPI
import h5py
import random
from dedalus.public import *

class Coefficients():
    def __init__(self, index, pmcoeffs_fn):
        self.index = index
        self.pmcoeffs_fn = pmcoeffs_fn
        
        pmcoeffs = h5py.File(self.pmcoeffs_fn,'r')
        self.Pm = pmcoeffs['Pm'].value[self.index]
        self.a = pmcoeffs['a'].value[self.index]
        self.b = pmcoeffs['b'].value[self.index]
        self.c = pmcoeffs['c'].value[self.index]
        self.h = pmcoeffs['h'].value[self.index]

# not saved so we'll just take value for Pm=1E-3
#Q = 0.01269
#Omega1 = 313.55
Q = 0.7470

gridnum = 128
lambda_crit = 2*np.pi/Q

pmcoeffs_fn = '/Users/susanclark/weakly_nonlinear_mri/data/pm_sat_coeffs.h5'

# Coefficients object contains coefficients
#coeff = Coefficients(15, pmcoeffs_fn)
#coeff = Coefficients(0, pmcoeffs_fn)
#coeff = Coefficients(19, pmcoeffs_fn)
coeff = Coefficients(5, pmcoeffs_fn)

print("saturation amplitude = {}".format(np.sqrt(coeff.b/coeff.c)))

# Actually solve the IVP
#Absolute = operators.Absolute
Absolute = de.operators.UnaryGridFunction
                   
num_critical_wavelengths = 20
Z_basis = de.Fourier('Z', gridnum, interval=(-(num_critical_wavelengths/2)*lambda_crit, (num_critical_wavelengths/2)*lambda_crit), dealias=3/2)
Zdomain = de.Domain([Z_basis], grid_dtype=np.complex128)
problem = de.IVP(Zdomain, variables=['alpha', 'alphaZ'], ncc_cutoff=1e-10)

problem.parameters['ac'] = -coeff.a/coeff.c
problem.parameters['bc'] = coeff.b/coeff.c
problem.parameters['hc'] = -coeff.h/coeff.c

print("ac = {}, bc = {}, hc = {}".format(-coeff.a/coeff.c, coeff.b/coeff.c, -coeff.h/coeff.c))

problem.add_equation("-ac*dt(alpha) + bc*alpha + hc*dZ(alphaZ) = alpha*abs(alpha**2)") 
problem.add_equation("alphaZ - dZ(alpha) = 0")

#dt = max_dt = 1.
dt = 2E-4
#period = 2*np.pi/Omega1

ts = de.timesteppers.RK443
IVP = problem.build_solver(ts)
IVP.stop_sim_time = np.inf#15.*period
IVP.stop_wall_time = np.inf
IVP.stop_iteration = 50000

# reference local grid and state fields
Z = Zdomain.grid(0)
alpha = IVP.state['alpha']
alphaZ = IVP.state['alphaZ']

# initial conditions ... plus noise!
noiselvl = 1E-3 #1E-15
noise = np.array([random.uniform(-noiselvl, noiselvl) for _ in range(len(Z))])
mean_init = 0
alpha['g'] = mean_init + noise#1.0E-5 + noise
alpha['c'][gridnum/2:] = 0 

# try subtracting mean IC value
alpha['g'] = alpha['g'] - np.nanmean(alpha['g'])

alpha.differentiate(0, out=alphaZ)

# Setup storage
alpha_list = [np.copy(alpha['g'])]
t_list = [IVP.sim_time]

# Main loop
dt = 2e-2#2e-3
while IVP.ok:
    IVP.step(dt)
    if IVP.iteration % 20 == 0:
        alpha_list.append(np.copy(alpha['g']))
        t_list.append(IVP.sim_time)
        
# Convert storage to arrays
alpha_array = np.array(alpha_list)
t_array = np.array(t_list)

saturation_amplitude = alpha_array[-1, 0]
print(saturation_amplitude)


f = h5py.File("/Users/susanclark/weakly_nonlinear_mri/data/Pm_"+str(coeff.Pm)+"_thingap_GLE_IVP_gridnum"+str(gridnum)+"_init_"+str(mean_init)+"_noiselvl"+str(noiselvl)+".hdf5", "w")
alpharr = f.create_dataset("alpha_array", data=alpha_array)
tarr = f.create_dataset("t_array", data=t_array)
f.attrs["Q"] = Q
f.attrs["lambda_crit"] = lambda_crit
f.attrs["gridnum"] = gridnum
f.attrs["num_lambda_crit"] = num_critical_wavelengths
f.attrs["mean_init"] = mean_init
f.attrs["dt"] = dt
f.attrs["Pm"] = coeff.Pm
f.attrs["a"] = coeff.a
f.attrs["b"] = coeff.b
f.attrs["c"] = coeff.c
f.attrs["h"] = coeff.h
f.close()

