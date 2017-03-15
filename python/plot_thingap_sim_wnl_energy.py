import numpy as np
import matplotlib.pyplot as plt
import h5py 
from matplotlib import pyplot, lines
from scipy import interpolate, optimize
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import rc
rc('text', usetex=True)

simroot1 = "MRI_run_Rm5.70e+00_eps0.00e+00_Pm1.00e-03_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz1024_Lz2_CFL"
simroot2 = "MRI_run_Rm5.30e+00_eps0.00e+00_Pm1.00e-03_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz1024_Lz2_CFL"
simroot3 = "MRI_run_Rm5.10e+00_eps0.00e+00_Pm1.00e-03_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz1024_Lz2_CFL"

epss = [0.1, 0.25, 0.5]
Rms = [5.7, 5.3, 5.1]
simrootfns = [simroot1, simroot2, simroot3]

simroot = simroot3

# data after restart
root = "/Users/susanclark/weakly_nonlinear_mri/data/simulations/"
simname = simroot + "_restart1"
fn = root + simname + "_scalar_s1.h5"
data = h5py.File(fn, "r")

# data before the restart
simname0 = simroot
data0 = h5py.File(root+simname0+"_scalar_s1.h5", "r")

# data from WNL theory
wnl_file_root = "/Users/susanclark/weakly_nonlinear_MRI/data/"
wnl_fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8790_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128"
wnl_fn_root = "../data/"
nLz = 1

newfn =root+"MRI_run_Rm4.90e+00_eps5.00e-01_Pm1.00e-03_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz1024_Lz2_CFL_evalueIC/scalar_s1_p0.h5"
newdata = h5py.File(newfn, "r")

newfn2 = root+"MRI_run_Rm4.90e+00_eps1.00e-01_Pm1.00e-03_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz1024_Lz2_CFL/scalar_s1.h5"
newdata2 = h5py.File(newfn2, "r")

wnl_TE = np.zeros(len(epss))
wnl_BE = np.zeros(len(epss))
wnl_KE = np.zeros(len(epss))
wnl_Jdot_tot = np.zeros(len(epss))
wnl_Jdot_R = np.zeros(len(epss))
wnl_Jdot_M = np.zeros(len(epss))

for i, _eps in enumerate(epss):
    wnl_data = h5py.File(wnl_fn_root + "zavg_quantities_"+str(int(nLz))+"Lz_eps"+str(_eps) + wnl_fn + ".h5", "r")

    wnl_TE[i] = wnl_data['TEint'].value.real
    wnl_BE[i] = wnl_data['BEint'].value.real
    wnl_KE[i] = wnl_data['KEint'].value.real
    
    wnl_Jdot_tot[i] = wnl_data['Jdot_tot'].value.real
    wnl_Jdot_R[i] = wnl_data['Jdot_R'].value.real
    wnl_Jdot_M[i] = wnl_data['Jdot_M'].value.real

# values taken from Jeff's plot_energy.py
t_orb = 2*np.pi
start = 2.5#125
stop = 5#190
e_upper = 10
e_lower = 1e-6

t = np.append(data0['/scales/sim_time'][:], data['/scales/sim_time'][:])
u_rms = np.append(data0['/tasks/vx_rms'][:,0,0], data['/tasks/vx_rms'][:,0,0])
KE = np.append(data0['/tasks/KE'][:,0,0], data['/tasks/KE'][:,0,0])
BE = np.append(data0['/tasks/BE'][:,0,0], data['/tasks/BE'][:,0,0])
TE = KE + BE
BxBy = np.append(data0['/tasks/BxBy'][:,0,0], data['/tasks/BxBy'][:,0,0])
uxuy = np.append(data0['/tasks/uxuy'][:,0,0], data['/tasks/uxuy'][:,0,0])
tot_stress = uxuy + BxBy

fig = plt.figure(facecolor="white")
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

colors=['cornflowerblue', 'darkorange', 'green']
markers = ['.', 'd', '+']

colorfamily1 = ["#66c2a4", "#2ca25f", "#006d2c"] #blue
colorfamily2 = ["#fc8d59", "#e34a33", "#b30000"]
colorfamily3 = ["#41b6c4", "#2c7fb8", "#253494"]
graycolors = ["#999999", "#777777", "#555555"]

ax1.semilogy(t/t_orb, TE, linestyle='-', label=r'$\mathrm{total}$', color=colorfamily3[2])
ax1.semilogy(t/t_orb, KE, linestyle='-.', label=r'$\mathrm{kinetic}$', color=colorfamily3[0])
ax1.semilogy(t/t_orb, BE, linestyle='--', label=r'$\mathrm{magnetic}$', color=colorfamily3[1])

ax1.legend(loc=4, frameon=False)

print('simulation TE saturation: {}'.format(TE[-1]))
print('WNL TE saturation: {}'.format(wnl_TE))

print('simulation KE saturation: {}'.format(KE[-1]))
print('WNL KE saturation: {}'.format(wnl_KE))

tjump = 20
#ax1.plot(t[-1]/t_orb + tjump, wnl_KE, markers[0], markerfacecolor='none', color='black')
#ax1.plot(t[-1]/t_orb + tjump, wnl_BE, markers[1], markerfacecolor='none', color='black')
#ax1.plot(t[-1]/t_orb + tjump, wnl_TE, markers[2], markerfacecolor='none', color='black')

# plot lines rather than points
ntsteps = len(t)
fiducialepsindx = -1
ax1.plot(t/t_orb, [wnl_KE[fiducialepsindx]]*ntsteps, linestyle='-', color=graycolors[0], alpha=0.9, zorder=0, lw=1)
ax1.plot(t/t_orb, [wnl_BE[fiducialepsindx]]*ntsteps, linestyle='-', color=graycolors[1], alpha=0.9, zorder=0, lw=1)
ax1.plot(t/t_orb, [wnl_TE[fiducialepsindx]]*ntsteps, linestyle='-', color=graycolors[2], alpha=0.9, zorder=0, lw=1)

ax1.set_xlabel(r'$\mathrm{Time}$ $\mathrm{(orbits)}$')
ax1.set_ylabel(r'$\mathrm{Energy}$')
tbuffer = 2
ax1.set_xlim(t[0]/t_orb - tbuffer, t[-1]/t_orb + tbuffer)

colorfamilies = [colorfamily1, colorfamily2, colorfamily3]
for i, (simroot, _Rm) in enumerate(zip(simrootfns, Rms)):
    simname = simroot + "_restart1"
    fn = root + simname + "_scalar_s1.h5"
    data = h5py.File(fn, "r")
    
    final_KE = data['/tasks/KE'][-1,0,0]
    final_BE = data['/tasks/BE'][-1,0,0]
    final_TE = TE = final_KE + final_BE
    
    # plot simulation saturation data by Rm
    #ax2.plot(_Rm, final_KE, markers[0], color=colorfamilies[i][0])
    #ax2.plot(_Rm, final_BE, markers[1], color=colorfamilies[i][1])
    #ax2.plot(_Rm, final_TE, markers[2], color=colorfamilies[i][2])

#wnl_Rm = 4.8790
#ax2.plot(wnl_Rm, wnl_KE, markers[0], markerfacecolor='none', color='black')
#ax2.plot(wnl_Rm, wnl_BE, markers[1], markerfacecolor='none', color='black')
#ax2.plot(wnl_Rm, wnl_TE, markers[2], markerfacecolor='none', color='black')
#ax2.set_xlabel('Rm')

for i, _eps in enumerate(epss):
    ax2.plot(_eps, wnl_KE[i], markers[0], markerfacecolor='none', color='black')
    ax2.plot(_eps, wnl_BE[i], markers[0], markerfacecolor='none', color='black')
    ax2.plot(_eps, wnl_TE[i], markers[0], markerfacecolor='none', color='black')
    

ax3.semilogy(t/t_orb, tot_stress, linestyle='-', label=r'$\mathrm{total}$', color=colorfamily2[2])    
ax3.semilogy(t/t_orb, uxuy, linestyle='-.', label=r'$\mathrm{Reynolds}$', color=colorfamily2[0])
ax3.semilogy(t/t_orb, BxBy, linestyle='--', label=r'$\mathrm{Maxwell}$', color=colorfamily2[1])
ax3.legend(loc=4, frameon=False)

ax3.plot(t/t_orb, [wnl_Jdot_R[fiducialepsindx]]*ntsteps, linestyle='-', color=graycolors[0], alpha=0.9, zorder=0, lw=1)
ax3.plot(t/t_orb, [wnl_Jdot_M[fiducialepsindx]]*ntsteps, linestyle='-', color=graycolors[1], alpha=0.9, zorder=0, lw=1)
ax3.plot(t/t_orb, [wnl_Jdot_tot[fiducialepsindx]]*ntsteps, linestyle='-', color=graycolors[2], alpha=0.9, zorder=0, lw=1)

ax3.set_xlabel(r'$\mathrm{Time}$ $\mathrm{(orbits)}$')
ax3.set_ylabel(r'$\mathrm{\dot{J}}$')

for i, _eps in enumerate(epss):
    ax4.plot(_eps, wnl_Jdot_R[i], markers[0], markerfacecolor='none', color='black')
    ax4.plot(_eps, wnl_Jdot_M[i], markers[0], markerfacecolor='none', color='black')
    ax4.plot(_eps, wnl_Jdot_tot[i], markers[0], markerfacecolor='none', color='black')
