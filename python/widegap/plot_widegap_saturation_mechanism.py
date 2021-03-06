import numpy as np
import matplotlib.pyplot as plt
import h5py 
from matplotlib import pyplot, lines
from scipy import interpolate, optimize
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import rc
rc('text', usetex=True)

fn_root = "/Users/susanclark/weakly_nonlinear_mri/data/"
fn = "widegap_amplitude_parameters_Q_0.90_Rm_3.3088_Pm_1.60e-06_Omega1_01.00_Omega2_00.12_beta_41.20_xi_0.00_gridnum_512_norm_True_Amidnorm_final"
thingap_fn = fn_root + "zavg_quantities_" + fn 
obj = h5py.File(thingap_fn + ".h5", "r")

rgrid = obj["rgrid"].value
uphifinal_zavg = obj["uphifinal_zavg"].value
Bzfinal_zavg = obj["Bzfinal_zavg"].value
nabla_u_zavg = obj["nabla_u_zavg"].value
nabla_B_zavg = obj["nabla_B_zavg"].value
base_flow_zavg = obj["base_flow_zavg"].value
Bzinitial_zavg = obj["Bzinitial_zavg"].value

nr = len(rgrid)

    
# final - initial plots, just center of channel
fig = plt.figure(figsize=(8, 8), facecolor="white")
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

lw = 1.3

ax1.plot(rgrid, uphifinal_zavg - base_flow_zavg, color="black", label=r"$u_\phi^{final} - u_\phi^{0}$", lw=lw)
ax1.set_title(r"$u_\phi^{saturated} - u_\phi^{0}$")

ax2.plot(rgrid, Bzinitial_zavg, color="gray", label=r"$B_z^{final} - B_z^{0}$", lw=lw)
ax2.plot(rgrid, Bzfinal_zavg, color="black", label=r"$B_z^{final} - B_z^{0}$", lw=lw)
ax2.set_yticks([np.min(Bzfinal_zavg), 1, np.max(Bzfinal_zavg)])
ax2.set_yticklabels([r"$-6E-9$", r"$1$", r"$+2E-8$"])
ax2.set_title(r"$B_z^{saturated}$")
ax2.text(1.05, 1.000000001, r"$B_z^{0}$", color="darkgray")

ax3.plot(rgrid, nabla_u_zavg, color="black", label=r"$\frac{1}{\mathrm{Re}}|\nabla^2 u|$", lw=lw)
ax3.set_title(r"$\frac{1}{Re}\nabla^2 u$")

ax4.plot(rgrid, nabla_B_zavg, color="black", label=r"$\frac{1}{\mathrm{Rm}}|\nabla^2 B|$", lw=lw)
ax4.set_title(r"$\frac{1}{Rm}\nabla^2 B$")
ax4.set_xlabel(r"$r$", size=15)

# set y limits, y ticks for small axes
for value, ax in zip([uphifinal_zavg - base_flow_zavg, nabla_B_zavg], [ax1, ax4]):
    valdiff = 0.1*np.max(value)
    ax.set_ylim(np.min(value) - valdiff, np.max(value) + valdiff)   
    ax.set_yticks([np.min(value) - valdiff, 0, np.max(value) + valdiff])
    
    # format ticklabels
    out = ax.get_ylim()
    tickout = [out[0], 0.0, out[1]]
    ax.set_yticklabels([r"${:.2}$".format(l) for l in tickout])

for value, ax in zip([nabla_u_zavg], [ax3]):
    valdiff = 0.1*np.max(value)
    ax.set_ylim(np.min(value) - valdiff, np.max(value) + valdiff)   
    ax.set_yticks([0, np.max(value) + valdiff])
    
    # format ticklabels
    out = ax.get_ylim()
    tickout = [0.0, out[1]]
    ax.set_yticklabels([r"${:.2}$".format(l) for l in tickout])

for ax in [ax1, ax3, ax4]:
    #ax.legend(loc=1)
    ax.plot(rgrid, np.zeros(nr), ":", color="gray")
    
for ax in [ax1, ax2, ax3]:
    ax.set_xticklabels([])

plt.subplots_adjust(hspace=0.5)
