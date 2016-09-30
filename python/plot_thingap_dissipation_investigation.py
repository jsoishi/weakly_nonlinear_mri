import numpy as np
import matplotlib.pyplot as plt
import h5py 
from matplotlib import pyplot, lines
from scipy import interpolate, optimize
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import rc
import dedalus.public as de
rc('text', usetex=True)

file_root = "/home/jsoishi/hg-projects/weakly_nonlinear_MRI/data/"
fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8790_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128"
obj = h5py.File(file_root + fn + ".h5", "r")

Q = obj.attrs['Q']
x = obj['x'].value
xgrid = x

# epsilon (small parameter)
eps = 0.5

# saturation amplitude -- for now just constant, coefficient-determined
satamp = np.sqrt(obj.attrs['b']/obj.attrs['c']) #1

# create z grid
nz = obj.attrs['gridnum']
Lz = 2*np.pi/Q
z = np.linspace(0, Lz, nz, endpoint=False)
zz = z.reshape(nz, 1)

dz = z[1] - z[0]

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

# two-dimensional u and Bstructure
V1_u = eps*satamp*obj['u11'].value*eiqz + eps*satamp.conj()*obj['u11_star'].value*eiqzstar
V1_B = eps*satamp*obj['B11'].value*eiqz + eps*satamp.conj()*obj['B11_star'].value*eiqzstar

V2_u = eps**2*satamp**2*obj['u22'].value*ei2qz + eps**2*satamp.conj()**2*obj['u22_star'].value*ei2qzstar + eps**2*(np.abs(satamp))**2*obj['u20'].value*ei0qz + eps**2*(np.abs(satamp.conj()))**2*obj['u20_star'].value*ei0qzstar
V2_B = eps**2*satamp**2*obj['B22'].value*ei2qz + eps**2*satamp.conj()**2*obj['B22_star'].value*ei2qzstar + eps**2*(np.abs(satamp))**2*obj['B20'].value*ei0qz + eps**2*(np.abs(satamp.conj()))**2*obj['B20_star'].value*ei0qzstar

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

#Vboth_u = Vboth_u + norm_base_flow
V1_ur1 = eps*satamp*obj['psi11'].value*eiqz_z + eps*satamp.conj()*obj['psi11_star'].value*eiqzstar_z
V1_uz1 = -eps*satamp*obj['psi11_x'].value*eiqz - eps*satamp.conj()*obj['psi11_star_x'].value*eiqzstar

V1_Br1 = eps*satamp*obj['A11'].value*eiqz_z + eps*satamp.conj()*obj['A11_star'].value*eiqzstar_z
V1_Bz1 = -eps*satamp*obj['A11_x'].value*eiqz - eps*satamp.conj()*obj['A11_star_x'].value*eiqzstar

V2_ur1 = eps**2*satamp**2*obj['psi22'].value*ei2qz_z + eps**2*satamp.conj()**2*psi22_star*ei2qzstar_z
V2_uz1 = -eps**2*satamp**2*obj['psi22_x'].value*ei2qz + -eps**2*satamp.conj()**2*psi22_star_x*ei2qzstar - np.abs(satamp)**2*(obj['psi20_x'].value*ei0qz + obj['psi20_star_x'].value*ei0qzstar)

Vboth_ur1 = V1_ur1 + V2_ur1
Vboth_uz1 = V1_uz1 + V2_uz1

V2_Br1 = eps**2*satamp**2*obj['A22'].value*ei2qz_z + eps**2*satamp.conj()**2*A22_star*ei2qzstar_z
V2_Bz1 = -eps**2*satamp**2*(obj['A22_x'].value*ei2qz + A22_star_x*ei2qzstar) - np.abs(satamp)**2*(obj['A20_x'].value*ei0qz + obj['A20_star_x'].value*ei0qzstar)

Vboth_Br1 = V1_Br1 + V2_Br1
Vboth_Bz1 = V1_Bz1 + V2_Bz1

Bzinitial_z0 = np.zeros(obj.attrs['gridnum']) + 1.0 # B0 = 1
Bzfinal_z0 = Bzinitial_z0 + Vboth_Bz1[0, :]

Bzinitial_2D = Bzinitial_z0*ei0qz

uphifinal_z0 = base_flow + Vboth_u[0, :]

# Define 2D fields
d2D = de.Domain([de.Fourier('z',nz,interval=[0,Lz]),de.Chebyshev('x',obj.attrs['gridnum'],interval=[-1,1])],grid_dtype='complex128')

Vboth_uphi_field = d2D.new_field()
Vboth_uphi_field['g'] = base_flow_2d + Vboth_u

Vboth_ur_field = d2D.new_field()
Vboth_ur_field['g'] = Vboth_ur1

Vboth_uz_field = d2D.new_field()
Vboth_uz_field['g'] = Vboth_uz1

Re = obj.attrs['Rm']/obj.attrs['Pm']
diff_tens_u = ((1.0/Re)*(Vboth_ur_field.differentiate(0).differentiate(0) + Vboth_uphi_field.differentiate(0).differentiate(0) + Vboth_uz_field.differentiate(0).differentiate(0)
            + Vboth_ur_field.differentiate(1).differentiate(1) + Vboth_uphi_field.differentiate(1).differentiate(1) + Vboth_uz_field.differentiate(1).differentiate(1))).evaluate()

Vboth_Bphi_field = d2D.new_field()
Vboth_Bphi_field['g'] = Vboth_B

Vboth_Br_field = d2D.new_field()
Vboth_Br_field['g'] = Vboth_Br1

Vboth_Bz_field = d2D.new_field()
Vboth_Bz_field['g'] = Bzinitial_2D + Vboth_Bz1

diff_tens_B = ((1.0/obj.attrs['Rm'])*(Vboth_Br_field.differentiate(0).differentiate(0) + Vboth_Bphi_field.differentiate(0).differentiate(0) + Vboth_Bz_field.differentiate(0).differentiate(0)
            + Vboth_Br_field.differentiate(1).differentiate(1) + Vboth_Bphi_field.differentiate(1).differentiate(1) + Vboth_Bz_field.differentiate(1).differentiate(1))).evaluate()

nabla_u_z0 = diff_tens_u['g'][0, :]
nabla_B_z0 = diff_tens_B['g'][0, :]

# vertical averages
base_flow_zavg = np.mean(base_flow_2d, axis=0)
uphifinal_zavg = np.mean(base_flow_2d + Vboth_u, axis=0)
Bzinitial_zavg = np.mean(Bzinitial_2D, axis=0)
Bzfinal_zavg = np.mean(Bzinitial_2D + Vboth_Bz1, axis=0)
nabla_u_zavg = np.mean(diff_tens_u['g'], axis=0)
nabla_B_zavg = np.mean(diff_tens_B['g'], axis=0)

# plotting
fig = plt.figure(facecolor="white")
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.plot(xgrid, base_flow, color="black", label=r"$u_\phi^{0}$")
ax1.plot(xgrid, uphifinal_z0, color="blue", label=r"$u_\phi^{final}$")

ax2.plot(xgrid, Bzinitial_z0, color="black", label=r"$B_z^{0}$")
ax2.plot(xgrid, Bzfinal_z0, color="blue", label=r"$B_z^{final}$")

ax3.plot(xgrid, nabla_u_z0, color="orange", label=r"$|\nabla^2 u|$")
ax3.plot(xgrid, nabla_B_z0, color="green", label=r"$|\nabla^2 B|$")

for ax in [ax1, ax2, ax3]:
    ax.legend(loc=4)
    
fig = plt.figure(facecolor="white")
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.plot(xgrid, base_flow - uphifinal_z0, color="black", label=r"$u_\phi^{0} - u_\phi^{final}$")
ax2.plot(xgrid, nabla_u_z0, color="orange", label=r"$|\nabla^2 u|$")
ax3.plot(xgrid, nabla_B_z0, color="green", label=r"$|\nabla^2 B|$")

for ax in [ax1, ax2, ax3]:
    ax.legend(loc=4)

# plotting z-averaged quantities
fig = plt.figure(facecolor="white")
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.plot(xgrid, base_flow_zavg, color="black", label=r"$u_\phi^{0}$")
ax1.plot(xgrid, uphifinal_zavg, color="blue", label=r"$u_\phi^{final}$")

ax2.plot(xgrid, Bzinitial_zavg, color="black", label=r"$B_z^{0}$")
ax2.plot(xgrid, Bzfinal_zavg, color="blue", label=r"$B_z^{final}$")

ax3.plot(xgrid, nabla_u_zavg, color="orange", label=r"$\frac{1}{Re}|\nabla^2 u|$")
ax3.plot(xgrid, nabla_B_zavg, color="green", label=r"$\frac{1}{Rm}|\nabla^2 B|$")

for ax in [ax1, ax2, ax3]:
    ax.legend(loc=4)
    
# final - initial plots
fig = plt.figure(facecolor="white")
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax1.plot(xgrid, uphifinal_zavg - base_flow_zavg, color="black", label=r"$u_\phi^{final} - u_\phi^{0}$")
ax2.plot(xgrid, Bzfinal_zavg - Bzinitial_zavg, color="black", label=r"$B_z^{final} - B_z^{0}$")
ax3.plot(xgrid, nabla_u_zavg, color="orange", label=r"$\frac{1}{Re}|\nabla^2 u|$")
ax4.plot(xgrid, nabla_B_zavg, color="green", label=r"$\frac{1}{Rm}|\nabla^2 B|$")

for ax in [ax1, ax2, ax3, ax4]:
    ax.legend(loc=1)
    #ax.set_xlim(1.5, 2.5)
    
# final - initial plots, just center of channel
fig = plt.figure(facecolor="white")
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax1.plot(xgrid[nz/4.:3*nz/4.], uphifinal_zavg[nz/4.:3*nz/4.]  - base_flow_zavg[nz/4.:3*nz/4.], color="black", label=r"$u_\phi^{final} - u_\phi^{0}$")
ax2.plot(xgrid[nz/4.:3*nz/4.], Bzfinal_zavg[nz/4.:3*nz/4.] - Bzinitial_zavg[nz/4.:3*nz/4.], color="black", label=r"$B_z^{final} - B_z^{0}$")
ax3.plot(xgrid[nz/4.:3*nz/4.], nabla_u_zavg[nz/4.:3*nz/4.], color="orange", label=r"$\frac{1}{Re}|\nabla^2 u|$")
ax4.plot(xgrid[nz/4.:3*nz/4.], nabla_B_zavg[nz/4.:3*nz/4.], color="green", label=r"$\frac{1}{Rm}|\nabla^2 B|$")

for ax in [ax1, ax2, ax3, ax4]:
    ax.legend(loc=1)
    #ax.set_xlim(1.5, 2.5)
    
fn_root = "../data/"
out_fn = fn_root + "zavg_quantities_" + fn + ".h5"
with h5py.File(out_fn,'w') as f:
    xgrid = f.create_dataset("xgrid", data=xgrid)
    uphifinal_zavg = f.create_dataset("uphifinal_zavg", data=uphifinal_zavg)
    Bzfinal_zavg = f.create_dataset("Bzfinal_zavg", data=Bzfinal_zavg)
    nabla_u_zavg = f.create_dataset("nabla_u_zavg", data=nabla_u_zavg)
    nabla_B_zavg = f.create_dataset("nabla_B_zavg", data=nabla_B_zavg)
    base_flow_zavg = f.create_dataset("base_flow_zavg", data=base_flow_zavg)
    Bzinitial_zavg = f.create_dataset("Bzinitial_zavg", data=Bzinitial_zavg)
    
    
   
