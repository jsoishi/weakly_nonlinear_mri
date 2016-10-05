import numpy as np
import matplotlib.pyplot as plt
import h5py 
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import rc
rc('text', usetex=True)
rc('figure', dpi=120)


lambdacrit = 2

root = "/Users/susanclark/weakly_nonlinear_mri/data/"
#fn = "Pm_0.00152830673266_thingap_GLE_IVP_gridnum128_init_0.hdf5"
#fn = "Pm_0.00152830673266_thingap_GLE_IVP_gridnum128_init_1E-15.hdf5"
#fn = "Pm_0.0001_thingap_GLE_IVP_gridnum128_init_0.hdf5"
#fn = "Pm_0.00316227766017_thingap_GLE_IVP_gridnum128_init_0.hdf5"
#fn = "Pm_0.000248162892284_thingap_GLE_IVP_gridnum128_init_0_nomean.hdf5"
#fn = "Pm_0.000248162892284_thingap_GLE_IVP_gridnum128_init_0_noiselvl0.001.hdf5"
if lambdacrit == 10:
    fn = "IVP_thingap_amplitude_parameters_Q_0.75_Rm_4.8790_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128_init_0_noiselvl0.001.hdf5"
elif lambdacrit == 2:
    fn = "IVP_thingap_amplitude_parameters_Q_0.75_Rm_4.8790_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128_init_0_noiselvl0.001_2.hdf5"


#fn = "IVP_widegap_amplitude_parameters_Q_2.33_Rm_0.0015_Pm_1.00e-06_Omega1_313.55_Omega2_84.66_beta_0.02_xi_4.00.h5"
#fn = "IVP_widegap_amplitude_parameters_Q_0.01_Rm_0.6735_Pm_1.00e-03_Omega1_313.55_Omega2_56.43_beta_25.00_xi_0.00.h5"
#fn = "IVP_widegap_amplitude_parameters_Q_0.01_Rm_0.6735_Pm_1.00e-03_Omega1_313.55_Omega2_56.43_beta_25.00_xi_0.00mean_init_0.0e+00_noiselvl_1.0e-03.h5"
#fn = "IVP_widegap_amplitude_parameters_Q_0.01_Rm_0.6735_Pm_1.00e-03_Omega1_313.55_Omega2_56.43_beta_25.00_xi_0.00mean_init_0.0e+00_noiselvl_1.0e-15.h5"

#fn = "IVP_widegap_amplitude_parameters_Q_2.33_Rm_0.0015_Pm_1.00e-06_Omega1_313.55_Omega2_84.66_beta_0.02_xi_4.00mean_init_0.0e+00_noiselvl_1.0e-15.h5"
#fn = "IVP_widegap_amplitude_parameters_Q_2.33_Rm_0.0015_Pm_1.00e-06_Omega1_313.55_Omega2_84.66_beta_0.02_xi_4.00mean_init_1.0e+00_noiselvl_1.0e-03.h5"

ivpdata = h5py.File(root + fn, "r")

alpha_array = ivpdata["alpha_array"].value
t_array = ivpdata["t_array"].value
#Z_array = ivpdata["Z_array"].value

print(list(ivpdata.attrs.keys()))

Q = ivpdata.attrs['Q']
lambda_crit = ivpdata.attrs['lambda_crit']
gridnum = ivpdata.attrs['gridnum']
num_lambda_crit = ivpdata.attrs['num_lambda_crit']
mean_init = ivpdata.attrs['mean_init']
dt = ivpdata.attrs['dt']
Pm = ivpdata.attrs['Pm']
a = ivpdata.attrs['a']
b = ivpdata.attrs['b']
c = ivpdata.attrs['c']
h = ivpdata.attrs['h']

Z_array = np.linspace(-num_lambda_crit*lambda_crit, num_lambda_crit*lambda_crit, alpha_array.shape[1])

coeff_sat_amp = np.sqrt(b/c)
print("coeff sat amp is {}".format(coeff_sat_amp))

#AAstar = alpha_array*alpha_array.conj()
AAstar = (alpha_array.real + alpha_array.imag)*(alpha_array.real - alpha_array.imag)
phases = np.arctan2(alpha_array.imag, alpha_array.real)

fig = plt.figure(figsize = (10, 6), facecolor = "white")
nrows = 4
ncols = 2 

#tstep = 1#00
cmap = "Spectral_r"#"BrBG"

# Colormap
rgbs = matplotlib.image.imread("/Users/susanclark/Dropbox/GALFA-Planck/colormap2.png")
circ_cmap = matplotlib.colors.ListedColormap(rgbs[0, :, :])
circ_cmap=cmap
 
ax1 = plt.subplot2grid((40,110), (0, 0), rowspan=30, colspan=44)
ax2 = plt.subplot2grid((40,110), (0, 60), rowspan=30, colspan=44)
ax3 = plt.subplot2grid((40,110), (35, 0), rowspan=5, colspan=44)
ax4 = plt.subplot2grid((40,110), (35, 60), rowspan=5, colspan=44)

cax1 = plt.subplot2grid((40,110), (23, 45), rowspan=5, colspan=2)
cax2 = plt.subplot2grid((40,110), (24, 105), rowspan=5, colspan=5, projection="polar")

# Plot the colorbar onto the polar axis
#display_axes = fig.add_axes([0.1,0.1,0.8,0.8], projection='polar')
cax2._direction = 2*np.pi 
norm = matplotlib.colors.Normalize(0.0, 2*np.pi)
# note - use orientation horizontal so that the gradient goes around
# the wheel rather than centre out
quant_steps = 2056
cb = matplotlib.colorbar.ColorbarBase(cax2, cmap=circ_cmap,
                                   norm=norm,
                                   orientation='horizontal')
# aesthetics - get rid of border and axis labels                                   
cb.outline.set_visible(False)                                 
cax2.set_axis_off()
#cb.set_ticks([0])
#cb.set_ticklabels([r"$\pi$"])
cax2.text(-.05, 1.15, r"$\pi$")

cax2.tick_params(axis=u'both', which=u'both',length=0)
cax2.tick_params(labelsize=16)
cax2.tick_params(axis='both', which='major', pad=15)

im1 = ax1.imshow(AAstar, cmap = cmap, aspect=192./251)
im2 = ax2.imshow(phases, cmap = circ_cmap, aspect=192./251)

cmap_vals = matplotlib.cm.get_cmap(cmap)

#for ax in [ax1, ax2, ax3, ax4]:
#    ax.set_xlim(0, len(alpha_array[-1, :]))
ax1.set_xlim(0, len(alpha_array[0, :]))#/tstep))
ax2.set_xlim(0, len(alpha_array[0, :]))#/tstep))
ax3.set_xlim(0, len(alpha_array[0, :]))#/tstep))
ax4.set_xlim(0, len(alpha_array[0, :]))#/tstep))

#ax1.set_ylim(0, len(alpha_array[0::tstep, 0]))
#ax2.set_ylim(0, len(alpha_array[0::tstep, 0]))

ax1.set_title(r"$\alpha \alpha^*$")
ax2.set_title(r"$\phi = \mathrm{arctan} \frac{Im(\alpha)}{Re(\alpha)}$")

ax1.set_ylabel(r"$T$", rotation=0, labelpad=15)
ax3.set_xlabel(r"$Z$")
 
ax2.set_ylabel(r"$T$", rotation=0, labelpad=15)
ax4.set_xlabel(r"$Z$")

ax1.set_yticks([])
ax2.set_yticks([])

cbar1 = plt.colorbar(im1, cax=cax1) 
cbar1.outline.set_visible(False)

cbar1_locs = [np.nanmin(AAstar), np.nanmax(AAstar)]
cbar1.set_ticks(cbar1_locs)


for cax in [cax1, cax2]:
    cax.tick_params(labelsize=8)


# plot saturation amplitude
ax3.plot(alpha_array[-1, :].real, color = "black", lw=2, label=r"$Re\{\alpha(t = t_{final})\}$")
ax3.plot(alpha_array[-1, :].imag, color = "gray", lw=2)
ax3.plot([0, len(alpha_array[-1, :])], [coeff_sat_amp.real, coeff_sat_amp.real], '--', color = "grey", lw=1.5, label=r"$+/-\sqrt{b/c}$") # dummy label plot
ax3.plot([0, len(alpha_array[-1, :])], [coeff_sat_amp.real, coeff_sat_amp.real], '--', color = cmap_vals(256), lw=1.6, alpha=0.8)
ax3.plot([0, len(alpha_array[-1, :])], [-coeff_sat_amp.real, -coeff_sat_amp.real], '--', color = cmap_vals(0), lw=1.6, alpha=0.8)
#ax3.legend(loc = "lower center", bbox_to_anchor=(0.5, -0.5), ncol = 2, prop={'size':10})
#ax3.set_xticklabels([])
ax3.set_title(r"$\alpha(t = t_{final})$")
ax4.set_title(r"$\phi(t = t_{final})$")

ax4.plot(phases[-1, :], color = "black", lw=1.6)

ax3.set_yticks([-coeff_sat_amp.real, coeff_sat_amp.real])
ax3.set_yticklabels([r"$-\sqrt{b/c}$", r"$+\sqrt{b/c}$"], size=10)

ax4.set_yticks([-np.pi, np.pi])
ax4.set_yticklabels([r"$-\pi$", r"$+\pi$"], size=10)

for spine in ax3.spines.values():
    spine.set_edgecolor('None')
    
for spine in ax4.spines.values():
    spine.set_edgecolor('None')

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xticks([0, len(Z_array)/2., len(Z_array)])
    ax.set_xticklabels([r"$-5\lambda_{crit}$", r"$0$", r"$5\lambda_{crit}$"])
    

#fig.subplots_adjust(hspace=0.5)

#plt.suptitle(r"$Pm = {}$".format(Pm), size=20)
#"""