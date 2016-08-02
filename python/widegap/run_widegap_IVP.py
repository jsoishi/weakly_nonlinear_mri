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

# UMR07-like parameters

Q = 0.01269
#Rm = 0.67355
Rm = 0.8403
Pm = 1.0e-3
beta = 25.0
Omega1 = 313.55
Omega2 = 56.43
xi = 0
R1 = 5
R2 = 15
gridnum = 128

# Hollerbach & Rudiger
"""
Pm = 1.0E-6
Re = 1521
Rm = Pm*Re
xi = 4
Q = 2.33
Ha = 16.3
beta = (2*Re*Rm)/(Ha**2)
R1 = 1.
R2 = 2.
mu_omega = 0.27
Omega1 = 313.55
Omega2 = Omega1*mu_omega
"""

# data created in run_widegap_amplitude.py
fn_root = "/Users/susanclark/weakly_nonlinear_mri/data/"
fn = "widegap_amplitude_parameters_Q_{:03.2f}_Rm_{:04.4f}_Pm_{:.2e}_Omega1_{:05.2f}_Omega2_{:05.2f}_beta_{:.2f}_xi_{:.2f}_gridnum_{}_intnormu".format(Q, Rm, Pm, Omega1, Omega2, beta, xi, gridnum)
coeff_fn = fn_root + fn + ".h5"

# For IVP
gridnum = 50
lambda_crit = 2*np.pi/Q

# Read coefficients from file
coeff = Coefficients(coeff_fn)

#print("HACK: all coeffs -> real")
#coeff.a = coeff.a.real
#coeff.b = coeff.b.real
#coeff.c = coeff.c.real
#coeff.h = coeff.h.real

#print("HACK: setting new coeffs")
#coeff.a = 1.0000000000000002+1.6817162881677001e-26j
#coeff.b = 0.00056776945676967957+1.5421420067772087e-14j
#coeff.c = 0.008129082765040497+6.182564137121617e-06j#0.008129112543701307+6.182564137121872e-06j#0.020810184143773249-1.3877885448830667e-05j
#coeff.h = 0.021707778758499826+3.9384254266509086e-11j

print("hack: changing sign of b, c coeffs")
coeff.b = -coeff.b
coeff.c = -coeff.c

print("ac = {}, bc = {}, hc = {}".format(-coeff.a/coeff.c, -coeff.b/coeff.c, -coeff.h/coeff.c)) 
print("saturation amplitude = {}".format(np.sqrt(-coeff.b/coeff.c)))

# Actually solve the IVP
                   
num_critical_wavelengths = 20
Z_basis = de.Fourier('Z', gridnum, interval=(-(num_critical_wavelengths/2)*lambda_crit, (num_critical_wavelengths/2)*lambda_crit), dealias=3/2)
Zdomain = de.Domain([Z_basis], grid_dtype=np.complex128)
problem = de.IVP(Zdomain, variables=['alpha', 'alphaZ'], ncc_cutoff=1e-10)

problem.parameters['ac'] = -coeff.a/coeff.c
problem.parameters['bc'] = -coeff.b/coeff.c
problem.parameters['hc'] = -coeff.h/coeff.c

# a dt alpha + b alpha + h dZ^2 alpha + c alpha|alpha|^2 = 0
# a dt alpha = b alpha + h dZ^2 alpha - c alpha|alpha|^2 (b -> -b, h -> -h) <- standard GLE structure
# -a dt alpha + b alpha + h dZ^2 alpha = c alpha |alpha|^2

problem.add_equation("ac*dt(alpha) + bc*alpha + hc*dZ(alphaZ) = alpha*abs(alpha**2)") 
problem.add_equation("alphaZ - dZ(alpha) = 0")

ts = de.timesteppers.RK443
IVP = problem.build_solver(ts)
IVP.stop_sim_time = np.inf#15.*period
IVP.stop_wall_time = np.inf
IVP.stop_iteration = 50000#0#0

# reference local grid and state fields
Z = Zdomain.grid(0)
alpha = IVP.state['alpha']
alphaZ = IVP.state['alphaZ']

# initial conditions ... plus noise!
noiselvl = 1E-3#15
noise = np.array([random.uniform(-noiselvl, noiselvl) for _ in range(len(Z))])
mean_init = 0.1
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
dt = 2e-1#2e-3
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
f = h5py.File(out_root + "IVP_" + fn + "mean_init_{:.1e}_noiselvl_{:.1e}.h5".format(mean_init, noiselvl), "w")
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

