import numpy as np
import matplotlib.pyplot as plt
import h5py 
from matplotlib import pyplot, lines
from scipy import interpolate, optimize
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import rc
rc('text', usetex=True)

import sys 
sys.path.insert(0, '../')
import streamplot_uneven as su

file_root = "/Users/susanclark/weakly_nonlinear_mri/data/"
fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8790_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128.h5"
#fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8738_Pm_1.00e-04_q_1.5_beta_25.00_gridnum_128.h5"
obj = h5py.File(file_root + fn, "r")

Q = obj.attrs['Q']
x = obj['x'].value + 1.1

# epsilon (small parameter)
eps = 0.5

# saturation amplitude -- for now just constant, coefficient-determined
satamp = np.sqrt(obj.attrs['b']/obj.attrs['c']) #1
#satamp = 1

# create z grid
nz = obj.attrs['gridnum']
Lz = 2*np.pi/Q
z = np.linspace(0, Lz, nz, endpoint=False)
zz = z.reshape(nz, 1)

dz = z[1] - z[0]

# impart structure in the z direction
eiqz = np.cos(Q*zz) + 1j*np.sin(Q*zz)
eiqz_z = 1j*Q*np.cos(Q*zz) - Q*np.sin(Q*zz) # dz(e^{ikz})

ei2qz = np.cos(2*Q*zz) + 1j*np.sin(2*Q*zz)
ei0qz = np.cos(0*Q*zz) + 1j*np.sin(0*Q*zz)

ei2qz_z = 2*1j*Q*np.cos(2*Q*zz) - 2*Q*np.sin(2*Q*zz)

# two-dimensional u and Bstructure
V1_u = eps*satamp*obj['u11'].value*eiqz
V1_B = eps*satamp*obj['B11'].value*eiqz

V2_u = eps**2*satamp**2*obj['u22'].value*ei2qz + eps**2*(np.abs(satamp))**2*obj['u20'].value*ei0qz
V2_B = eps**2*satamp**2*obj['B22'].value*ei2qz + eps**2*(np.abs(satamp))**2*obj['B20'].value*ei0qz

Vboth_B = V1_B + V2_B
Vboth_u = V1_u + V2_u

xcutoff = False#20
if xcutoff is not False:
    V1_u = V1_u[:, xcutoff:-xcutoff]
    V1_B = V1_B[:, xcutoff:-xcutoff]
    plotx = x[xcutoff:-xcutoff]
else:
    plotx = x

fig = plt.figure(facecolor="white", figsize = (10, 4), dpi=150)
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

#cmap = sns.diverging_palette(240, 10, n = 256, as_cmap=True)
cmap = "RdBu_r"
ticklabelsize = 10
axislabelsize = 12
titlesize = 15
cbarwidth = 0.3
cbarheight = 0.02
cbarleft = 0.56
cbarvert = 0.93

def plot_o1(ax, obj, type="Bfield", labels=True, oplot=True, background=False, cmap="RdBu_r"):

    if type is "velocity":
        cbarmax = np.max(V1_u.real)
        cbarmin = np.min(V1_u.real)
        info = ax.pcolormesh(plotx, z, V1_u.real, cmap=cmap, vmin = cbarmin, vmax = cbarmax,linewidth=0,rasterized=True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)  
        #cax = fig.add_axes([cbarleft, cbarvert, cbarwidth, cbarheight])  
        cbar = plt.colorbar(info, cax=cax)
        cbar.ax.tick_params(labelsize = ticklabelsize)

        if labels == True:
            ax.set_xlabel(r"$x$", size = axislabelsize)
            ax.set_ylabel(r"$z$", size = axislabelsize)
            #cbar.set_label(r"$u_y$ Perturbation", size = axislabelsize)
            ax.set_title(r"$\mathrm{First}$ $\mathrm{Order}$", size = titlesize)

        # take derivatives to find [r] and [z] components of velocity perturbation
        #u_r = dz(psi); u_z = -dr(psi)
        V1_ur1 = eps*satamp*obj['psi11'].value*eiqz_z
        V1_uz1 = -eps*satamp*obj['psi11_x'].value*eiqz
    
        if xcutoff is not False:
            V1_ur1 = V1_ur1[:, xcutoff:-xcutoff]
            V1_uz1 = V1_uz1[:, xcutoff:-xcutoff]

        if oplot == True:
            u1mag = np.sqrt(np.abs(V1_ur1**2) + np.abs(V1_uz1**2))
            norm_mag = u1mag/u1mag.max()
            su.streamplot(ax, plotx, z, V1_ur1.real, V1_uz1.real, linewidth = 3*norm_mag, color = "black")
            print(norm_mag[norm_mag < 0])

    elif type is "Bfield":

        cbarmax = np.max(V1_B.real)
        cbarmin = np.min(V1_B.real)
        info = ax.pcolormesh(plotx, z, V1_B.real, cmap=cmap, vmin = cbarmin, vmax = cbarmax,linewidth=0,rasterized=True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)    
        cbar = plt.colorbar(info, cax=cax)
        cbar.ax.tick_params(labelsize = ticklabelsize)

        if labels == True:
            ax.set_xlabel(r"$x$", size = axislabelsize)
            ax.set_ylabel(r"$z$", size = axislabelsize)
            #cbar.set_label(r"$B_y$ Perturbation", size = 20)
            ax.set_title(r"$\mathrm{First}$ $\mathrm{Order}$", size = titlesize)

        # take derivatives to find [r] and [z] components of velocity perturbation
        if background is True:
            V1_Bz1 = (-eps*satamp*obj['A11_x'].value + eps*1)*eiqz
        else:
            V1_Bz1 = -eps*satamp*obj['A11_x'].value*eiqz
        V1_Br1 = eps*satamp*obj['A11'].value*eiqz_z

        if oplot == True:
            B1mag = np.sqrt(np.abs(V1_Br1**2) + np.abs(V1_Bz1**2))
            su.streamplot(ax, x, z, V1_Br1.real, V1_Bz1.real, linewidth = 2*B1mag/B1mag.max(), color = "black")

def plot_o2(ax, obj, type="Bfield", labels=True, oplot=True, background=False, cmap="RdBu_r"):
    if type is "velocity":
        cbarmax = np.max(V2_u.real)
        cbarmin = np.min(V2_u.real)
        info = ax.pcolormesh(x, z, V2_u.real, cmap=cmap, vmin = cbarmin, vmax = cbarmax,linewidth=0,rasterized=True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)    
        cbar = plt.colorbar(info, cax=cax)
        cbar.ax.tick_params(labelsize = ticklabelsize)
    
        if labels == True:
            ax.set_xlabel(r"$x$", size = axislabelsize)
            ax.set_ylabel(r"$z$", size = axislabelsize)
            #cbar.set_label(r"$u_y$ Perturbation", size = axislabelsize)
            ax.set_title(r"$\mathrm{Second}$ $\mathrm{Order}$")
    
        # take derivatives to find [r] and [z] components of velocity perturbation
        #V2_uz1 = -satamp**2*(1/oe2.r)*ei2qz*oe2.psi22_r['g'] - (satamp*satamp.conj())*(1/oe2.r)*oe2.psi20_r['g']*ei0qz
        V2_uz1 = -eps**2*satamp**2*ei2qz*obj['psi22_x'].value - eps**2*(np.abs(satamp))**2*obj['psi20_x'].value*ei0qz
        V2_ur1 = eps**2*satamp**2*ei2qz_z*obj['psi22'].value
        
        if oplot == True:
            u2mag = np.sqrt(np.abs(V2_ur1**2) + np.abs(V2_uz1**2))
            norm_mag = u2mag/u2mag.max()
            su.streamplot(ax, x, z, V2_ur1.real, V2_uz1.real, linewidth = 2*norm_mag, color = "black")
            print(norm_mag[norm_mag < 0])
    
    elif type is "Bfield":
    
        cbarmax = np.max(V2_B.real)
        cbarmin = np.min(V2_B.real)
        info = ax.pcolormesh(x, z, V2_B.real, cmap=cmap, vmin = cbarmin, vmax = cbarmax,linewidth=0,rasterized=True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)    
        cbar = plt.colorbar(info, cax=cax)
        cbar.ax.tick_params(labelsize = ticklabelsize)
    
        if labels == True:
            ax.set_xlabel(r"$x$", size = axislabelsize)
            ax.set_ylabel(r"$z$", size = axislabelsize)
            #cbar.set_label(r"$B_y$ Perturbation", size = 20)
            ax.set_title(r"$\mathrm{Second}$ $\mathrm{Order}$")
    
        # take derivatives to find [r] and [z] components of velocity perturbation
        #V1_Bz1 = -eps*(1/r)*satamp*obj['A22_r'].value*eiqz
        #V1_Br1 = eps*(1/r)*satamp*obj['A11'].value*eiqz_z
        
        V2_Bz1 = -eps**2*satamp**2*ei2qz*obj['A22_x'].value - (np.abs(satamp))**2*eps**2*obj['A20_x'].value*ei0qz
        V2_Br1 = eps**2*satamp**2*ei2qz_z*obj['A22'].value
    
        if oplot == True:
            B2mag = np.sqrt(np.abs(V2_Br1**2) + np.abs(V2_Bz1**2))
            su.streamplot(ax, x, z, V2_Br1.real, V2_Bz1.real, linewidth = 2*B2mag/B2mag.max(), color = "black")
            
def plot_both(ax, obj, type="Bfield", labels=True, oplot=True, background=False, cmap="RdBu_r"):
    if type is "velocity":
        cbarmax = np.max(eps*Vboth_u.real)
        cbarmin = np.min(eps*Vboth_u.real)
        info = ax.pcolormesh(x, z, eps*Vboth_u.real, cmap=cmap, vmin = cbarmin, vmax = cbarmax,linewidth=0,rasterized=True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)    
        cbar = plt.colorbar(info, cax=cax)
        cbar.ax.tick_params(labelsize = ticklabelsize)   
        
        if labels is True:
            ax.set_xlabel(r"$x$", size = axislabelsize)
            ax.set_ylabel(r"$z$", size = axislabelsize)
            #cbar.set_label(r"$B_y$ Perturbation", size = 20)
            ax.set_title(r"$\mathrm{Total}$ $\mathrm{Perturbation}$")
    
        Vboth_uz1 = -eps*(satamp*obj['psi11_x'].value*eiqz) + eps**2*(-satamp**2*ei2qz*obj['psi22_x'].value - (np.abs(satamp))**2*obj['psi20_x'].value*ei0qz)
        Vboth_ur1 = eps*(satamp*obj['psi11'].value*eiqz_z) + eps**2*(satamp**2*ei2qz_z*obj['psi22'].value)
        
        if oplot == True:
            ubothmag = np.sqrt(np.abs(Vboth_ur1**2) + np.abs(Vboth_uz1**2))
            su.streamplot(ax, x, z, Vboth_ur1.real, Vboth_uz1.real, linewidth = 3*ubothmag/ubothmag.max(), color = "black")

    
    elif type is "Bfield":
    
        cbarmax = np.max(Vboth_B.real)
        cbarmin = np.min(Vboth_B.real)
        info = ax.pcolormesh(x, z, Vboth_B.real, cmap=cmap, vmin = cbarmin, vmax = cbarmax,linewidth=0,rasterized=True)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)    
        cbar = plt.colorbar(info, cax=cax)
        cbar.ax.tick_params(labelsize = ticklabelsize)
        
        if labels is True:
            ax.set_xlabel(r"$x$", size = axislabelsize)
            ax.set_ylabel(r"$z$", size = axislabelsize)
            #cbar.set_label(r"$B_y$ Perturbation", size = 20)
            ax.set_title(r"$\mathrm{Total}$ $\mathrm{Perturbation}$")
    
        Vboth_Bz1 = -eps*(satamp*obj['A11_x'].value*eiqz) + eps**2*(-satamp**2*ei2qz*obj['A22_x'].value - (np.abs(satamp))**2*obj['A20_x'].value*ei0qz)
        Vboth_Br1 = eps*(satamp*obj['A11'].value*eiqz_z) + eps**2*(satamp**2*ei2qz_z*obj['A22'].value)

        if oplot == True:
            Bbothmag = np.sqrt(np.abs(Vboth_Br1**2) + np.abs(Vboth_Bz1**2))
            su.streamplot(ax, x, z, Vboth_Br1.real, Vboth_Bz1.real, linewidth = 2*Bbothmag/Bbothmag.max(), color = "black")

type = "Bfield"
labels = True
oplot = True
background = False

plot_o1(ax1, obj, type = type, labels = labels, oplot = oplot)
plot_o2(ax2, obj, type = type, labels = labels, oplot = oplot)
plot_both(ax3, obj, type = type, labels = labels, oplot = oplot)

axs = [ax1, ax2, ax3]
print(x)

for ax in axs:  
    ax.set_ylim(np.min(z), np.max(z))# - dz)
    ax.set_xlim(np.min(x), np.max(x))
    plt.tick_params(labelsize = ticklabelsize)
    ax.set_yticks([0, np.max(z)])
    ax.set_yticklabels([r"$0$", r"$\lambda_c$"])
    ax.set_xticks([])
    
fig.subplots_adjust(wspace=0.6)
#plt.savefig("../figures/thingap_streamfuncs_"+type+"_Pm_1E-3.png", dpi=fig.dpi)
#plt.savefig("../figures/thingap_streamfuncs_"+type+"_Pm_1E-4.png", dpi=fig.dpi)

#plt.savefig('../paper/thingap_submit/thingap_streamfuncs_velocity_Pm_1E-3.eps', dpi=100)