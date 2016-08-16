import dedalus.public as de
import numpy as np
from mpi4py import MPI
import h5py
import random
from dedalus.public import *

class Coefficients():
    def __init__(self, coeffs_fn):
        self.coeffs_fn = coeffs_fn
        
        coeffs = h5py.File(self.coeffs_fn,'r')
        self.Pm = coeffs.attrs['Pm']
        self.a = coeffs.attrs['a']
        self.b = coeffs.attrs['b']
        self.c = coeffs.attrs['c']
        self.h = coeffs.attrs['h']

        self.Q = coeffs.attrs['Q']
         
# data created in run_widegap_amplitude.py
fn_root = "/Users/susanclark/weakly_nonlinear_mri/data/"
#fn = "widegap_amplitude_parameters_Q_{:03.2f}_Rm_{:04.4f}_Pm_{:.2e}_Omega1_{:05.2f}_Omega2_{:05.2f}_beta_{:.2f}_xi_{:.2f}_gridnum_{}_Anorm".format(Q, Rm, Pm, Omega1, Omega2, beta, xi, gridnum)
fn = "widegap_amplitude_parameters_Q_2.33_Rm_0.0015_Pm_1.00e-06_Omega1_01.00_Omega2_00.27_beta_0.02_xi_4.00_gridnum_128_conducting_False_norm_False"
coeff_fn = fn_root + fn + ".h5"

# Read coefficients from file
coeff = Coefficients(coeff_fn)

# For IVP
Q = coeff.Q
gridnum = 512
lambda_crit = 2*np.pi/Q

#print("HACK: all coeffs -> real")
#coeff.a = coeff.a.real
#coeff.b = coeff.b.real
#coeff.c = coeff.c.real
#coeff.h = coeff.h.real

#print("HACK: setting new coeffs")
#coeff.a = 1+2.1485568148251073e-26j
#coeff.b = 0.045368169457467356+2.4922212429290374e-13j#0.0007066860686026265+8.853530842684617e-15j
#coeff.c = 0.24796998124059338+2.259874552160415e-12j
#coeff.h = 2.7433491997827075+1.2795201244316272e-09j

print("HACK: testing coeffs from http://pauli.uni-muenster.de/tp/fileadmin/lehre/NumMethoden/SoSe10/Skript/GLE.pdf")
coeff.a = 1.0 + 0j
coeff.b = 1.0 + 0j
coeff.c = 1.0 - 1.0j
coeff.h = 1.0 + 2.0j

print("ac = {}, bc = {}, hc = {}".format(coeff.a/coeff.c, coeff.b/coeff.c, coeff.h/coeff.c)) 
print("saturation amplitude = {}".format(np.sqrt(coeff.b/coeff.c)))

# Actually solve the IVP
                   
num_critical_wavelengths = 20
Z_basis = de.Fourier('Z', gridnum, interval=(-(num_critical_wavelengths/2)*lambda_crit, (num_critical_wavelengths/2)*lambda_crit), dealias=3/2)
Zdomain = de.Domain([Z_basis], grid_dtype=np.complex128)
problem = de.IVP(Zdomain, variables=['alpha', 'alphaZ'], ncc_cutoff=1e-10)

problem.parameters['ac'] = coeff.a/coeff.c
problem.parameters['bc'] = coeff.b/coeff.c
problem.parameters['hc'] = coeff.h/coeff.c

# a dt alpha + b alpha + h dZ^2 alpha + c alpha|alpha|^2 = 0
# a dt alpha = b alpha + h dZ^2 alpha - c alpha|alpha|^2 (b -> -b, h -> -h) <- standard GLE structure
# -a dt alpha + b alpha + h dZ^2 alpha = c alpha |alpha|^2

problem.add_equation("-ac*dt(alpha) + bc*alpha + hc*dZ(alphaZ) = alpha*abs(alpha**2)") 
problem.add_equation("alphaZ - dZ(alpha) = 0")

ts = de.timesteppers.RK443
IVP = problem.build_solver(ts)
IVP.stop_sim_time = np.inf#15.*period
IVP.stop_wall_time = np.inf
tlen = 5000#0
IVP.stop_iteration = tlen

# reference local grid and state fields
Z = Zdomain.grid(0)
alpha = IVP.state['alpha']
alphaZ = IVP.state['alphaZ']

# initial conditions ... plus noise!
noiselvl = 1.0E-1#1E-3#15
noise = np.array([random.uniform(-noiselvl, noiselvl) for _ in range(len(Z))])
mean_init = 1#0#0.1
alpha['g'] = mean_init + noise#1.0E-5 + noise
alpha['c'][gridnum/2:] = 0 

# try subtracting mean IC value
#alpha['g'] = alpha['g'] - np.nanmean(alpha['g'])

print("should saturate in t ~ {}".format((1/coeff.b)*np.log(np.sqrt(-coeff.b/coeff.c))/mean_init))

alpha.differentiate(0, out=alphaZ)

# Setup storage
alpha_list = [np.copy(alpha['g'])]
t_list = [IVP.sim_time]

# Main loop
dt = 5e-2#2e-3
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

out_root = "/Users/susanclark/weakly_nonlinear_mri/data/"
f = h5py.File(out_root + "IVP_" + fn + "_mean_init_{:.1e}_noiselvl_{:.1e}_tlen_{}.h5".format(mean_init, noiselvl, tlen), "w")
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

