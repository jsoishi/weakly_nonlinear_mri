import numpy as np
import matplotlib.pyplot as plt
import h5py 
from matplotlib import pyplot, lines
from scipy import interpolate, optimize
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import rc
import dedalus.public as de
import copy
rc('text', usetex=True)

#file_root = "/home/joishi/hg-projects/weakly_nonlinear_mri/data/"
file_root = "/Users/susanclark/weakly_nonlinear_MRI/data/"
fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8790_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128"
#fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8738_Pm_1.00e-04_q_1.5_beta_25.00_gridnum_256_Anorm"
#fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8790_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_256_Anorm"
#fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8790_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_256_Anorm"

#fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.9000_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_256_Anorm"
#fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.9000_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128_Anorm"

#fn = "thingap_amplitude_parameters_Q_0.76_Rm_4.9213_Pm_1.00e-02_q_1.5_beta_25.00_gridnum_128_Anorm"
#fn = "thingap_amplitude_parameters_Q_0.84_Rm_5.3502_Pm_1.00e-01_q_1.5_beta_25.00_gridnum_128_Anorm"
#fn = "thingap_amplitude_parameters_Q_0.76_Rm_4.9213_Pm_1.00e-02_q_1.5_beta_25.00_gridnum_128_Anorm"
fn = "thingap_amplitude_parameters_Q_0.76_Rm_4.9213_Pm_1.00e-02_q_1.5_beta_25.00_gridnum_128_Anorm"

fn = "thingap_amplitude_parameters_Q_0.76_Rm_4.9213_Pm_1.00e-02_q_1.5_beta_25.00_gridnum_128_norm_False" # non normalized

fn = "thingap_amplitude_parameters_Q_0.76_Rm_4.9213_Pm_1.00e-02_q_1.5_beta_25.00_gridnum_128_norm_True_normconst2"

obj = h5py.File(file_root + fn + ".h5", "r")

Q = obj.attrs['Q']
b = obj.attrs['b']
c = obj.attrs['c']
#c = 113.75214354998781
x = obj['x'].value
xgrid = x

# epsilon (small parameter)
eps = 0.5

constant_alpha = True 
if constant_alpha:
    # saturation amplitude -- for now just constant, coefficient-determined
    satamp = np.sqrt(obj.attrs['b']/obj.attrs['c']) #1
    print('satamp is sqrt {}/{} = {}'.format(b, c, satamp))
    # satamp = np.sqrt(obj.attrs['b']/(obj.attrs['c']/obj.attrs['normconst']**2)) #test
    # satamp = satamp*0 + (0.27929596842805482-4.6781424718396889e-14j)
    satamp = satamp*0 +  0.23643163#0.23645746
    satamp_Z = 0.0 + 0j
    print('saturation amplitude = {}'.format(satamp))
    ivporconst = "const"
else:
    print('saturation amplitude = {} but using IVP data'.format(np.sqrt(obj.attrs['b']/obj.attrs['c'])))
    #fn = "IVP_thingap_amplitude_parameters_Q_0.75_Rm_4.9000_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128_Anorm_init_0_noiselvl0.001_eps_0.5_gridnum_1024.hdf5"
    fn = "IVP_thingap_amplitude_parameters_Q_0.75_Rm_4.9000_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128_Anorm_init_0_noiselvl0.001_eps_0.5_gridnum_1024_nodealias.hdf5"
    fn = "IVP_thingap_amplitude_parameters_Q_0.75_Rm_4.9000_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128_Anorm_init_0_noiselvl0.001_dt_0.2_nlambdacrit_2_eps_0.5_gridnum_1024_nodealias.hdf5"
    fn = "IVP_thingap_amplitude_parameters_Q_0.76_Rm_4.9213_Pm_1.00e-02_q_1.5_beta_25.00_gridnum_128_Anorm_init_0_noiselvl0.001_dt_0.2_nlambdacrit_2_eps_0.5_gridnum_1024_nodealias.hdf5"
    fn = "IVP_thingap_amplitude_parameters_Q_0.76_Rm_4.9213_Pm_1.00e-02_q_1.5_beta_25.00_gridnum_128_Anorm_init_0_noiselvl0.001_dt_0.002_nlambdacrit_2_eps_0.5_gridnum_1024_nodealias2.hdf5"
    fn = "IVP_thingap_amplitude_parameters_Q_0.76_Rm_4.9213_Pm_1.00e-02_q_1.5_beta_25.00_gridnum_128_Anorm_init_0_noiselvl0.01_dt_0.002_nlambdacrit_20_eps_0.5_gridnum_1024_nodealias.hdf5"
    ivpdata = h5py.File(file_root + fn, "r")
    alpha_array = ivpdata["alpha_array"].value
    alphaZ_array = ivpdata["alphaZ_array"].value
    final_alpha = alpha_array[-1, :]
    final_alphaZ = alphaZ_array[-1, :]
    ivp_lambda_crit = ivpdata.attrs['lambda_crit']
    ivp_gridnum = ivpdata.attrs['gridnum']
    ivp_num_lambda_crit = ivpdata.attrs['num_lambda_crit']
    #Z_array = np.linspace(-ivp_num_lambda_crit*ivp_lambda_crit, ivp_num_lambda_crit*ivp_lambda_crit, alpha_array.shape[1])
    satamp = final_alpha.reshape(ivpdata.attrs['gridnum'], 1)
    satamp_Z = final_alphaZ.reshape(ivpdata.attrs['gridnum'], 1)
    ivporconst = "IVP"
 
 
# to avoid all ambiguity in z, take it from the sims.
sim_fn_root = "../data/simulations/"
simname = "MRI_run_Rm4.92e+00_eps5.00e-01_Pm1.00e-02_beta2.50e+01_Q7.60e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz1024_Lz2_CFL_evalueIC"
simfn = sim_fn_root + simname + "_plotfields.h5"
simdata = h5py.File(simfn, "r")
simz = simdata['z'].value
 
beta = obj.attrs['beta']

# create z grid
#nz = obj.attrs['gridnum']
nz = 1024#2048
nLz = 1#2 # number of critical wavelengths
Lz = 2*np.pi/Q
#z = np.linspace(0, nLz*Lz, nz, endpoint=False)
z = copy.copy(simz)
zz = z.reshape(nz, 1)

dz = z[1] - z[0]

# fit center of channel u
#In [197]: np.where(simB == np.nanmax(simB))
#Out[197]: (array([128, 640]), array([64, 64]))
simB = simdata['B'].value
#zcenterindx = np.int(nz/8.)
#xcenterindx = np.int(len(x)/2.)
zcenterindx = 640
xcenterindx = 64
zcenter = z[zcenterindx]
xcenter = x[xcenterindx]
simBcenter = simB[zcenterindx, xcenterindx]
print('found simBcenter = {} at {}, {}'.format(simBcenter, zcenterindx, xcenterindx))

eiqz = np.exp(1j*Q*zz)
ei2qz = np.exp(2*1j*Q*zz)
ei0qz = np.exp(0*1j*Q*zz)
eiqz_z = 1j*Q*np.exp(1j*Q*zz)
ei2qz_z = 2*1j*Q*np.exp(2*1j*Q*zz)

eiqzstar = np.exp(-1j*Q*zz)
eiqzstar_z = -1j*Q*np.exp(-1j*Q*zz)
ei2qzstar = np.exp(-2*1j*Q*zz)
ei2qzstar_z = -2*1j*Q*np.exp(-2*1j*Q*zz)
ei0qzstar = np.exp(-0*1j*Q*zz)

# u21 star
psi21_star = obj['psi21'].value.conj()
u21_star = obj['u21'].value.conj()
A21_star = obj['A21'].value.conj()
B21_star = obj['B21'].value.conj() 

#print('x gridnum is {}'.format(gridnum))
#x_basis = de.Chebyshev('x',gridnum)
#d1d = de.Domain([x_basis], np.complex128, comm=MPI.COMM_SELF)

# test all z height
simsat_root1_all = np.zeros(nz, np.complex128)
simsat_root2_all = np.zeros(nz, np.complex128)
for i, _zindx in enumerate(range(nz)):
    B1_center = eps*obj['B11'].value[xcenterindx]*eiqz[_zindx] + eps*obj['B11_star'].value[xcenterindx]*eiqzstar[_zindx]
    B2_center = eps**2*obj['B22'].value[xcenterindx]*ei2qz[_zindx] + eps**2*obj['B22_star'].value[xcenterindx]*ei2qzstar[_zindx] + eps**2*obj['B20'].value[xcenterindx]*ei0qz[_zindx] + eps**2*obj['B20_star'].value[xcenterindx]*ei0qzstar[_zindx]

    simroot1 = (-B1_center + np.sqrt(B1_center**2 + 4*B2_center*simBcenter))/(2.0*B2_center)
    simroot2 = (-B1_center - np.sqrt(B1_center**2 + 4*B2_center*simBcenter))/(2.0*B2_center)
    simsat_root1_all[i] = simroot1[0]
    simsat_root2_all[i] = simroot2[0]

print('avg roots over all z: {} and {}'.format(np.nanmean(simsat_root1_all), np.nanmean(simsat_root1_all)))

#In [198]: np.where(Vboth_B == np.nanmax(Vboth_B))
#Out[198]: (array([644]), array([63]))
xcenterindx_wnl = 63
zcenterindx_wnl = 644

B1_center = eps*obj['B11'].value[xcenterindx_wnl]*eiqz[zcenterindx_wnl] + eps*obj['B11_star'].value[xcenterindx_wnl]*eiqzstar[zcenterindx_wnl]
B2_center = eps**2*obj['B22'].value[xcenterindx_wnl]*ei2qz[zcenterindx_wnl] + eps**2*obj['B22_star'].value[xcenterindx_wnl]*ei2qzstar[zcenterindx_wnl] + eps**2*obj['B20'].value[xcenterindx_wnl]*ei0qz[zcenterindx_wnl] + eps**2*obj['B20_star'].value[xcenterindx_wnl]*ei0qzstar[zcenterindx_wnl]

print('satamp*u1center + satamp**2*u2_center should = simucenter.')
simsat_root1 = (-B1_center + np.sqrt(B1_center**2 + 4*B2_center*simBcenter))/(2.0*B2_center)
simsat_root2 = (-B1_center - np.sqrt(B1_center**2 + 4*B2_center*simBcenter))/(2.0*B2_center)
print('satamp = {} or {}'.format(simsat_root1, simsat_root2))

print('test: {} = {}'.format(simsat_root1*B1_center + simsat_root1**2*B2_center, simBcenter))


# two-dimensional u and Bstructure
V1_u = eps*satamp*obj['u11'].value*eiqz + eps*satamp.conj()*obj['u11_star'].value*eiqzstar
V1_B = eps*satamp*obj['B11'].value*eiqz + eps*satamp.conj()*obj['B11_star'].value*eiqzstar

V2_u = eps**2*satamp**2*obj['u22'].value*ei2qz + eps**2*satamp.conj()**2*obj['u22_star'].value*ei2qzstar + eps**2*(np.abs(satamp))**2*obj['u20'].value*ei0qz + eps**2*(np.abs(satamp.conj()))**2*obj['u20_star'].value*ei0qzstar #+ eps**2*(satamp_Z*obj['u21'].value*eiqz + satamp_Z.conj()*u21_star*eiqzstar)
V2_B = eps**2*satamp**2*obj['B22'].value*ei2qz + eps**2*satamp.conj()**2*obj['B22_star'].value*ei2qzstar + eps**2*(np.abs(satamp))**2*obj['B20'].value*ei0qz + eps**2*(np.abs(satamp.conj()))**2*obj['B20_star'].value*ei0qzstar #+ eps**2*(satamp_Z*obj['B21'].value*eiqz + satamp_Z.conj()*B21_star*eiqzstar)

#V2_B = eps**2*satamp**2*obj['B22'].value*ei2qz + eps**2*(np.abs(satamp))**2*obj['B20'].value*ei0qz

Vboth_B = V1_B + V2_B
Vboth_u = V1_u + V2_u

# shear parameter
q = obj.attrs['q']

#base_flow_rdim = rgrid*c1 + c2/rgrid
#base_flow = (base_flow_rdim*ei0qz)/((R1 + R2)/2)
base_flow= -q*xgrid
base_flow_2d = base_flow*ei0qz

#norm_base_flow = base_flow/np.nanmax(base_flow)

# psi22_star, a22_star not stored
psi22_star = obj['psi22'].value.conj()
psi22_star_x = obj['psi22_x'].value.conj()

A22_star = obj['A22'].value.conj()
A22_star_x = obj['A22_x'].value.conj()

psi21_star_x = obj['psi21_x'].value.conj()
A21_star = obj['A21'].value.conj()
A21_star_x = obj['A21_x'].value.conj()

# Vboth Psi, A
V1_Psi = eps*satamp*obj['psi11'].value*eiqz + eps*satamp.conj()*obj['psi11_star'].value*eiqzstar
V2_Psi = eps**2*satamp**2*obj['psi22'].value*ei2qz + eps**2*satamp.conj()**2*psi22_star*ei2qzstar + eps**2*np.abs(satamp)**2*(obj['psi20'].value*ei0qz + obj['psi20_star'].value*ei0qzstar) 
if constant_alpha is False:
    V2_Psi += eps**2*(satamp_Z*obj['psi21'].value*eiqz + satamp_Z.conj()*psi21_star*eiqzstar)
Vboth_Psi = V1_Psi + V2_Psi

V1_A = eps*satamp*obj['A11'].value*eiqz + eps*satamp.conj()*obj['A11_star'].value*eiqzstar
V2_A = eps**2*satamp**2*obj['A22'].value*ei2qz + eps**2*satamp.conj()**2*A22_star*ei2qzstar + eps**2*np.abs(satamp)**2*(obj['A20'].value*ei0qz + obj['A20_star'].value*ei0qzstar) 
if constant_alpha is False:
    V2_A += eps**2*(satamp_Z*obj['A21'].value*eiqz + satamp_Z.conj()*A21_star*eiqzstar)
Vboth_A = V1_A + V2_A

# first order perturbations
V1_ur1 = eps*satamp*obj['psi11'].value*eiqz_z + eps*satamp.conj()*obj['psi11_star'].value*eiqzstar_z
V1_uz1 = -eps*satamp*obj['psi11_x'].value*eiqz - eps*satamp.conj()*obj['psi11_star_x'].value*eiqzstar
V1_Br1 = eps*satamp*obj['A11'].value*eiqz_z + eps*satamp.conj()*obj['A11_star'].value*eiqzstar_z
V1_Bz1 = -eps*satamp*obj['A11_x'].value*eiqz - eps*satamp.conj()*obj['A11_star_x'].value*eiqzstar

print('CHECK: using satamp {}'.format(satamp))

# second order perturbations
V2_ur1 = eps**2*satamp**2*obj['psi22'].value*ei2qz_z + eps**2*satamp.conj()**2*psi22_star*ei2qzstar_z 
if constant_alpha is False:
    V2_ur1 += eps**2*(satamp_Z*obj['psi21'].value*eiqz_z + satamp_Z.conj()*psi21_star*eiqzstar_z)
V2_uz1 = -eps**2*satamp**2*obj['psi22_x'].value*ei2qz - eps**2*satamp.conj()**2*psi22_star_x*ei2qzstar - eps**2*np.abs(satamp)**2*(obj['psi20_x'].value*ei0qz + obj['psi20_star_x'].value*ei0qzstar) 
if constant_alpha is False:
    V2_uz1 -= eps**2*(satamp_Z*obj['psi21_x'].value*eiqz + satamp_Z.conj()*psi21_star_x*eiqzstar)
V2_Br1 = eps**2*satamp**2*obj['A22'].value*ei2qz_z + eps**2*satamp.conj()**2*A22_star*ei2qzstar_z 
if constant_alpha is False:
    V2_Br1 += eps**2*(satamp_Z*obj['A21'].value*eiqz_z + satamp_Z.conj()*A21_star*eiqzstar_z)
V2_Bz1 = -eps**2*satamp**2*obj['A22_x'].value*ei2qz -eps**2*satamp.conj()**2*A22_star_x*ei2qzstar - eps**2*np.abs(satamp)**2*(obj['A20_x'].value*ei0qz + obj['A20_star_x'].value*ei0qzstar) 
if constant_alpha is False:
    V2_Bz1 -= eps**2*(satamp_Z*obj['A21_x'].value*eiqz + satamp_Z.conj()*A21_star_x*eiqzstar)

# Combined perturbations
Vboth_ur1 = V1_ur1 + V2_ur1
Vboth_uz1 = V1_uz1 + V2_uz1
Vboth_Br1 = V1_Br1 + V2_Br1
Vboth_Bz1 = V1_Bz1 + V2_Bz1

Bzinitial_z0 = np.zeros(obj.attrs['gridnum']) + 1.0 # B0 = 1
Bzinitial_2D = Bzinitial_z0*ei0qz

# Define 2D fields
d2D = de.Domain([de.Fourier('z',nz,interval=[0,nLz*Lz]),de.Chebyshev('x',obj.attrs['gridnum'],interval=[-1,1])],grid_dtype='complex128')

# saturated uphi and Bz include background fields
sat_uphi_field = d2D.new_field()
sat_uphi_field['g'] = base_flow_2d + Vboth_u
sat_Bz_field = d2D.new_field()
sat_Bz_field['g'] = Bzinitial_2D + Vboth_Bz1

# total perturbation-only 
Vboth_ur_field = d2D.new_field()
Vboth_ur_field['g'] = Vboth_ur1
Vboth_uz_field = d2D.new_field()
Vboth_uz_field['g'] = Vboth_uz1
Vboth_Bphi_field = d2D.new_field()
Vboth_Bphi_field['g'] = Vboth_B
Vboth_Br_field = d2D.new_field()
Vboth_Br_field['g'] = Vboth_Br1
Vboth_uphi_field = d2D.new_field()
Vboth_uphi_field['g'] = Vboth_u
Vboth_Bz_field = d2D.new_field()
Vboth_Bz_field['g'] = Vboth_Bz1

Vboth_A_field = d2D.new_field()
Vboth_A_field['g'] = Vboth_A
Vboth_Psi_field = d2D.new_field()
Vboth_Psi_field['g'] = Vboth_Psi

Re = obj.attrs['Rm']/obj.attrs['Pm']
diff_tens_u = ((1.0/Re)*(Vboth_ur_field.differentiate('x')**2 + Vboth_uphi_field.differentiate('x')**2 + Vboth_uz_field.differentiate('x')**2 + Vboth_ur_field.differentiate('z')**2 + Vboth_uphi_field.differentiate('z')**2 + Vboth_uz_field.differentiate('z')**2)).evaluate()

diff_tens_B = ((1.0/obj.attrs['Rm'])*(Vboth_Br_field.differentiate('x')**2 + Vboth_Bphi_field.differentiate('x')**2 + Vboth_Bz_field.differentiate('x')**2
                                      + Vboth_Br_field.differentiate('z')**2 + Vboth_Bphi_field.differentiate('z')**2 + Vboth_Bz_field.differentiate('z')**2)).evaluate()

e_stress_production = (Vboth_ur_field['g'] * Vboth_uphi_field['g'] - 2*(Vboth_Br_field['g'] * Vboth_Bphi_field['g'])/beta)

etot = d2D.new_field()
etot['g'] = q*e_stress_production - diff_tens_u['g'] - 2*diff_tens_B['g']/beta
dEdt = etot.integrate('x').integrate('z')['g'][0,0].real
print("test energy saturation: dE/dt = {:10.5e}".format(dEdt))

# compute reynolds stress
Reynolds_stress = Vboth_ur_field['g'] * Vboth_uphi_field['g']
TR = d2D.new_field()
TR['g'] = Reynolds_stress

# compute maxwell stress
Maxwell_stress = -(2/obj.attrs['beta'])*Vboth_Br_field['g']*Vboth_Bphi_field['g']
TM = d2D.new_field()
TM['g'] = Maxwell_stress

Ttot = d2D.new_field()
Ttot['g'] = Reynolds_stress + Maxwell_stress

# J dot is avg stress over domain
Jdot_R = TR.integrate('x').integrate('z')/(nLz*Lz*2)
Jdot_R = Jdot_R.evaluate()['g'][0][0]
Jdot_M = TM.integrate('x').integrate('z')/(nLz*Lz*2)
Jdot_M = Jdot_M.evaluate()['g'][0][0]
Jdot_tot = Ttot.integrate('x').integrate('z')/(nLz*Lz*2)
Jdot_tot = Jdot_tot.evaluate()['g'][0][0]

print('Jdot reynolds: {}, magnetic: {}, total: {}'.format(Jdot_R, Jdot_M, Jdot_tot))

print('checking that {} is equal to {}'.format(Jdot_tot, Jdot_M + Jdot_R))

# compute total energy (volume averaged)
KE = 0.5*(Vboth_ur_field['g']**2 + Vboth_uphi_field['g']**2 + Vboth_uz_field['g']**2)
KEfield = d2D.new_field()
KEfield['g'] = KE
KE_int = KEfield.integrate('x').integrate('z')/(nLz*Lz*2)
KE_int = KE_int.evaluate()
print('avg integrated kinetic energy: {}'.format(KE_int['g'][0][0]))  

BE = 0.5*(2/beta)*(Vboth_Br_field['g']**2 + Vboth_Bphi_field['g']**2 + Vboth_Bz_field['g']**2)
BEfield = d2D.new_field()
BEfield['g'] = BE
BE_int = BEfield.integrate('x').integrate('z')/(nLz*Lz*2)
BE_int = BE_int.evaluate()
print('avg integrated magnetic energy: {}'.format(BE_int['g'][0][0])) 

TE_int = KE_int['g'][0][0] + BE_int['g'][0][0]
print('avg integrated total energy: {}'.format(TE_int)) 


# only in the bulk
"""
# define bulk as -0.5 <= x <= 0.5
bulk_left_edge  = -0.5
bulk_right_edge = 0.5
Lx_bulk = bulk_right_edge - bulk_left_edge

KE_anti = KEfield.antidifferentiate(d2D.bases[-1],("left", 0))
KE_anti_zint = KE_anti.integrate('z')
avg_KE_bulk = (KE_anti_zint.interpolate(x=bulk_right_edge) - KE_anti_zint.interpolate(x=bulk_left_edge)).evaluate()
avg_KE_bulk = avg_KE_bulk['g'][0,0]/(Lx_bulk*nLz*Lz)

BE_anti = BEfield.antidifferentiate(d2D.bases[-1],("left", 0))
BE_anti_zint = BE_anti.integrate('z')
avg_BE_bulk = (BE_anti_zint.interpolate(x=bulk_right_edge) - BE_anti_zint.interpolate(x=bulk_left_edge)).evaluate()
avg_BE_bulk = avg_BE_bulk['g'][0,0]/(Lx_bulk*nLz*Lz)

TR_anti = TR.antidifferentiate(d2D.bases[-1],("left", 0))
TR_anti_zint = TR_anti.integrate('z')
avg_TR_bulk = (TR_anti_zint.interpolate(x=bulk_right_edge) - TR_anti_zint.interpolate(x=bulk_left_edge)).evaluate()
avg_TR_bulk = avg_TR_bulk['g'][0,0]/(Lx_bulk*nLz*Lz)

TM_anti = TM.antidifferentiate(d2D.bases[-1],("left", 0))
TM_anti_zint = TM_anti.integrate('z')
avg_TM_bulk = (TM_anti_zint.interpolate(x=bulk_right_edge) - TM_anti_zint.interpolate(x=bulk_left_edge)).evaluate()
avg_TM_bulk = avg_TM_bulk['g'][0,0]/(Lx_bulk*nLz*Lz)

print('avg_KE_bulk = {}'.format(avg_KE_bulk))
print('avg_BE_bulk = {}'.format(avg_BE_bulk))
print('avg_TR_bulk = {}'.format(avg_TR_bulk))
print('avg_TM_bulk = {}'.format(avg_TM_bulk))
"""

# vertical averages
base_flow_zavg = np.mean(base_flow_2d, axis=0)
uphifinal_zavg = np.mean(sat_uphi_field['g'], axis=0)
Bzinitial_zavg = np.mean(Bzinitial_2D, axis=0)
Bzfinal_zavg = np.mean(sat_Bz_field['g'], axis=0)
nabla_u_zavg = np.mean(diff_tens_u['g'], axis=0)
nabla_B_zavg = np.mean(diff_tens_B['g'], axis=0)
    
#J\left(\Psi, u_{y}\right) + (2 - q) \Omega_0 \partial_z \Psi \ - \frac{2}{\beta}B_0\partial_z B_{y} \, - \, \frac{2}{\beta} J\left(A, B_{y}\right) \, - \, \frac{1}{\reye} \nabla^2 u_{y} = 0

# Steady state u terms 
# J (Psi, u)
JPsiu = (Vboth_ur_field*Vboth_uphi_field.differentiate('x') + Vboth_uz_field*Vboth_uphi_field.differentiate('z')).evaluate()

# J (A, B)
JAB = (-(2.0/beta)*(Vboth_Br_field*Vboth_Bphi_field.differentiate('x') + Vboth_Bz_field*Vboth_Bphi_field.differentiate('z'))).evaluate()

nablasqu = (-(1.0/Re)*(Vboth_uphi_field.differentiate('x').differentiate('x') + Vboth_uphi_field.differentiate('z').differentiate('z'))).evaluate()

shearu = ((2 - q)*Vboth_ur_field).evaluate()

dzBphi = (-(2.0/beta)*Vboth_Bphi_field.differentiate('z')).evaluate()


JPsiu_zavg = np.mean(JPsiu['g'], axis=0)
JAB_zavg = np.mean(JAB['g'], axis=0)
nablasqu_zavg = np.mean(nablasqu['g'], axis=0)
shearu_zavg = np.mean(shearu['g'], axis=0)
dzBphi_zavg = np.mean(dzBphi['g'], axis=0)

# to test if they sum to zero....
all2D = JPsiu['g'] + JAB['g'] + nablasqu['g'] + shearu['g'] + dzBphi['g']

# J (A, Psi)
JAPsi = (-(-Vboth_Br_field*Vboth_uz_field + Vboth_Bz_field*Vboth_ur_field)).evaluate()

nablasqB = (Vboth_Bphi_field.differentiate('x').differentiate('x') + Vboth_Bphi_field.differentiate('z').differentiate('z')).evaluate()

nablasqA = (Vboth_Br_field.differentiate('z') - Vboth_Bz_field.differentiate('x')).evaluate()
nablasqA2 = (Vboth_A_field.differentiate('x').differentiate('x') + Vboth_A_field.differentiate('z').differentiate('z')).evaluate()

nablasqAterm = (-(1.0/obj.attrs['Rm'])*nablasqA).evaluate()
nablasqAterm2 = (-(1.0/obj.attrs['Rm'])*nablasqA2).evaluate()
print("Rm:", obj.attrs['Rm'])
print("Q:", obj.attrs['Q'])

Aterm1 = (-Vboth_ur_field).evaluate()
Aterm2 = (-Vboth_Psi_field.differentiate('z')).evaluate()

JAPsi_zavg = np.mean(JAPsi['g'], axis=0)
nablasqAterm_zavg = np.mean(nablasqAterm['g'], axis=0)
Aterm1_zavg = np.mean(Aterm1['g'], axis=0)

# partial_x of terms in the steady state A equation to get steady state of B_z
JAPsi_dx = JAPsi.differentiate('x')
nablasqAterm_dx = nablasqAterm.differentiate('x')
Aterm1_dx = Aterm1.differentiate('x')

Aterm2_dx = Aterm2.differentiate('x')

# to test if they sum to zero....
all2DBx = (JAPsi_dx + nablasqAterm_dx + Aterm1_dx).evaluate()

#isn't the above the wrong A term?
all2DBxnew = (JAPsi_dx + nablasqAterm_dx + Aterm2_dx).evaluate()

JAPsi_dx_zavg = np.mean(JAPsi_dx['g'], axis=0)
nablasqAterm_dx_zavg = np.mean(nablasqAterm_dx['g'], axis=0)
Aterm1_dx_zavg = np.mean(Aterm1_dx['g'], axis=0)

slicenum=40

Bzfinalsq = d2D.new_field()
Bzfinalsq['g'] = sat_Bz_field['g']**2
Bzfinalsq_int = Bzfinalsq.integrate('x').integrate('z')
Bzfinalsq_int['g'][0][0]/(Lz*2)
print('Bz final sq = {}'.format(Bzfinalsq_int['g'][0][0]/(nLz*Lz*2)))

Bzinitsq = d2D.new_field()
Bzinitsq['g'] = Bzinitial_2D**2
Bzinitsq_int = Bzinitsq.integrate('x').integrate('z')
Bzinitsq_int['g'][0][0]/(Lz*2)
print('Bz init sq = {}'.format(Bzinitsq_int['g'][0][0]/(nLz*Lz*2)))
     
print("nLz is ", nLz)
save = True
if save is True:
    print('saving output...')

    fn_root = "../data/"
    out_fn = fn_root + "zavg_quantities_"+str(int(nLz))+"Lz_eps"+ str(eps) + fn + "_satamp_"+ivporconst+"_simnorm2.h5"
    
    print(out_fn)
    
    with h5py.File(out_fn,'w') as f:
        dxgrid = f.create_dataset("xgrid", data=xgrid)
        dzgrid = f.create_dataset("zgrid", data=z)
        duphifinal_zavg = f.create_dataset("uphifinal_zavg", data=uphifinal_zavg)
        dBzfinal_zavg = f.create_dataset("Bzfinal_zavg", data=Bzfinal_zavg)
        dnabla_u_zavg = f.create_dataset("nabla_u_zavg", data=nabla_u_zavg)
        dnabla_B_zavg = f.create_dataset("nabla_B_zavg", data=nabla_B_zavg)
        dbase_flow_zavg = f.create_dataset("base_flow_zavg", data=base_flow_zavg)
        dBzinitial_zavg = f.create_dataset("Bzinitial_zavg", data=Bzinitial_zavg)
    
        dJPsiu_zavg = f.create_dataset("JPsiu_zavg", data=JPsiu_zavg)
        dJAB_zavg = f.create_dataset("JAB_zavg", data=JAB_zavg)
        dnablasqu_zavg = f.create_dataset("nablasqu_zavg", data=nablasqu_zavg)
        dshearu_zavg = f.create_dataset("shearu_zavg", data=shearu_zavg)
        ddzBphi_zavg = f.create_dataset("dzBphi_zavg", data=dzBphi_zavg)
    
        dJPsiu_slice = f.create_dataset("JPsiu_slice", data=JPsiu['g'][slicenum, :])
        dJAB_slice = f.create_dataset("JAB_slice", data=JAB['g'][slicenum, :])
        dnablasqu_slice = f.create_dataset("nablasqu_slice", data=nablasqu['g'][slicenum, :])
        dshearu_slice = f.create_dataset("shearu_slice", data=shearu['g'][slicenum, :])
        ddzBphi_slice = f.create_dataset("dzBphi_slice", data=dzBphi['g'][slicenum, :])
    
        dJAPsi_zavg = f.create_dataset("JAPsi_zavg", data=JAPsi_zavg)
        dnablasqAterm_zavg = f.create_dataset("nablasqAterm_zavg", data=nablasqAterm_zavg)
        dAterm1_zavg = f.create_dataset("Aterm1_zavg", data=Aterm1_zavg)
    
        dJAPsi_dx_zavg = f.create_dataset("JAPsi_dx_zavg", data=JAPsi_dx_zavg)
        dnablasqAterm_dx_zavg = f.create_dataset("nablasqAterm_dx_zavg", data=nablasqAterm_dx_zavg)
        dAterm1_dx_zavg = f.create_dataset("Aterm1_dx_zavg", data=Aterm1_dx_zavg)
    
        dJAPsi_dx_slice = f.create_dataset("JAPsi_dx_slice", data=JAPsi_dx['g'][slicenum, :])
        dnablasqAterm_dx_slice = f.create_dataset("nablasqAterm_dx_slice", data=nablasqAterm_dx['g'][slicenum, :])
        dAterm1_dx_slice = f.create_dataset("Aterm1_dx_slice", data=Aterm1_dx['g'][slicenum, :])
    
        BEout = f.create_dataset("BEint", data=BE_int['g'][0][0])
        KEout = f.create_dataset("KEint", data=KE_int['g'][0][0])
        TEout = f.create_dataset("TEint", data=TE_int)
        
        Jdot_Rout = f.create_dataset("Jdot_R", data=Jdot_R)
        Jdot_Mout = f.create_dataset("Jdot_M", data=Jdot_M)
        Jdot_totout = f.create_dataset("Jdot_tot", data=Jdot_tot)
        
        """
        BEbulkout = f.create_dataset("BE_bulk_int", data=avg_KE_bulk)
        KEbulkout = f.create_dataset("KE_bulk_int", data=avg_BE_bulk)
        TEbulkout = f.create_dataset("TE_bulk_int", data=avg_KE_bulk + avg_BE_bulk)
        
        JdotRbulkout = f.create_dataset("Jdot_R_bulk_int", data=avg_TR_bulk)
        JdotMbulkout = f.create_dataset("Jdot_M_bulk_int", data=avg_TM_bulk)
        Jdottotbulkout = f.create_dataset("Jdot_tot_bulk_int", data=avg_TR_bulk + avg_TM_bulk)
        """
        
        psiout = f.create_dataset("psi_sat", data=Vboth_Psi)
        uout = f.create_dataset("u_sat", data=Vboth_u)
        Aout = f.create_dataset("A_sat", data=Vboth_A)
        Bout = f.create_dataset("B_sat", data=Vboth_B)
        
        Reynolds_stress_out = f.create_dataset("Reynolds_stress", data=Reynolds_stress)
        Maxwell_stress_out = f.create_dataset("Maxwell_stress", data=Maxwell_stress)
        
        f.attrs["eps"] = eps
    
    obj.close()    
    
   
