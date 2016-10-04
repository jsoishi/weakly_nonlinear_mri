import numpy as np
import matplotlib.pyplot as plt
import h5py 
import matplotlib
from matplotlib import pyplot, lines
from scipy import interpolate, optimize
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import rc
rc('text', usetex=True)

#fn_root = "/home/joishi/hg-projects/weakly_nonlinear_mri/data/"
fn_root = "/Users/susanclark/weakly_nonlinear_MRI/data/"
fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8790_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128"
#fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8738_Pm_1.00e-04_q_1.5_beta_25.00_gridnum_256_Anorm"
#fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8790_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_256_Anorm"
thingap_fn = fn_root + "zavg_quantities_" + fn 
obj = h5py.File(thingap_fn + ".h5", "r")

xgrid = obj["xgrid"].value
uphifinal_zavg = obj["uphifinal_zavg"].value
Bzfinal_zavg = obj["Bzfinal_zavg"].value
nabla_u_zavg = obj["nabla_u_zavg"].value
nabla_B_zavg = obj["nabla_B_zavg"].value
base_flow_zavg = obj["base_flow_zavg"].value
Bzinitial_zavg = obj["Bzinitial_zavg"].value

nx = len(xgrid)

    
# final - initial plots, just center of channel
fig = plt.figure(figsize=(8, 8), facecolor="white")
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

lw = 1.3

ax1.plot(xgrid, uphifinal_zavg - base_flow_zavg, color="black", label=r"$u_\phi^{final} - u_\phi^{0}$", lw=lw)
ax1.set_title(r"$u_\phi^{saturated} - u_\phi^{0}$")

ax2.plot(xgrid, Bzinitial_zavg, color="gray", label=r"$B_z^{final} - B_z^{0}$", lw=lw)
ax2.plot(xgrid, Bzfinal_zavg, color="black", label=r"$B_z^{final} - B_z^{0}$", lw=lw)
ax2.set_yticks([np.min(Bzfinal_zavg), 1, np.max(Bzfinal_zavg)])
ax2.set_yticklabels([r"$0.9985$", r"$1$", r"$1.0025$"])
ax2.set_title(r"$B_z^{saturated}$")
ax2.text(-0.95, 1.0005, r"$B_z^{0}$", color="darkgray")

ax3.plot(xgrid, nabla_u_zavg, color="black", label=r"$\frac{1}{\mathrm{Re}}|\nabla^2 u|$", lw=lw)
ax3.set_title(r"$\frac{1}{Re}\nabla^2 u$")

ax4.plot(xgrid, nabla_B_zavg, color="black", label=r"$\frac{1}{\mathrm{Rm}}|\nabla^2 B|$", lw=lw)
ax4.set_title(r"$\frac{1}{Rm}\nabla^2 B$")

# set y limits, y ticks for small axes
for value, ax in zip([uphifinal_zavg - base_flow_zavg, nabla_u_zavg, nabla_B_zavg], [ax1, ax3, ax4]):
    valdiff = 0.1*np.max(value)
    ax.set_ylim(np.min(value) - valdiff, np.max(value) + valdiff)   
    ax.set_yticks([np.min(value) - valdiff, 0, np.max(value) + valdiff])
    
    # format ticklabels
    out = ax.get_ylim()
    tickout = [out[0], 0.0, out[1]]
    ax.set_yticklabels([r"${:.2}$".format(l) for l in tickout])

for ax in [ax1, ax3, ax4]:
    #ax.legend(loc=1)
    ax.plot(xgrid, np.zeros(nx), ":", color="gray")
    
for ax in [ax1, ax2, ax3]:
    ax.set_xticklabels([])

plt.subplots_adjust(hspace=0.5)



# New figure
JPsiu_zavg = obj["JPsiu_zavg"].value
JAB_zavg = obj["JAB_zavg"].value
nablasqu_zavg = obj["nablasqu_zavg"].value
shearu_zavg = obj["shearu_zavg"].value
dzBphi_zavg = obj["dzBphi_zavg"].value
#slice instead of avg
JPsiu_slice = obj["JPsiu_slice"].value
JAB_slice = obj["JAB_slice"].value
nablasqu_slice = obj["nablasqu_slice"].value
shearu_slice = obj["shearu_slice"].value
dzBphi_slice = obj["dzBphi_slice"].value


fig = plt.figure(figsize=(6, 6), facecolor="white")
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(xgrid, uphifinal_zavg - base_flow_zavg, color="black", label=r"$u_\phi^{final} - u_\phi^{0}$", lw=lw)
#ax1.plot(xgrid, uphifinal_zavg, color="black", label=r"$u_\phi^{sat}$", lw=lw)
ax1.set_title(r"$u_\phi^{final} - u_\phi^{0}$")

colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
lw = 2

zavg = False
if zavg is True:
    ax2.plot(xgrid, JPsiu_zavg, "--", color="slateblue", lw=lw, label=r"$J(\Psi, u)$")
    ax2.plot(xgrid, JAB_zavg, ":", color="green", lw=lw, label=r"$-\frac{2}{\beta} J(A, B)$")
    ax2.plot(xgrid, nablasqu_zavg, color="tomato", lw=lw, label=r"$-\frac{1}{\mathrm{Re}} \nabla^2 u$")
    ax2.plot(xgrid, dzBphi_zavg, "-.", color="royalblue", lw=lw, label=r"$-\frac{2}{\beta} B_0 \partial_z B$")
    ax2.plot(xgrid, shearu_zavg, "_", color="orangered", lw=lw, label=r"$(2-q)\Omega_0 \partial_z \Psi$" )
else:
    ax2.plot(xgrid, JPsiu_slice, "--", color=colors[0], lw=lw, label=r"$J(\Psi, u)$")
    ax2.plot(xgrid, JAB_slice, ":", color=colors[1], lw=lw, label=r"$-\frac{2}{\beta} J(A, B)$")
    ax2.plot(xgrid, nablasqu_slice, color=colors[2], lw=lw, label=r"$-\frac{1}{\mathrm{Re}} \nabla^2 u$")
    ax2.plot(xgrid, dzBphi_slice, "-.", color=colors[3], lw=lw, label=r"$-\frac{2}{\beta} B_0 \partial_z B$")
    ax2.plot(xgrid, shearu_slice, dashes = [5,2,10,5], color=colors[4], lw=lw, label=r"$(2-q)\Omega_0 \partial_z \Psi$" )
ax2.set_xlabel(r"$x$", size=15)
plt.legend(prop={'size':10}, bbox_to_anchor=(0.9, 0.35))

#ax2.set_xlim(-0.2, 0.2)
#ax2.set_ylim(-0.00008, 0.00008)
#ax2.set_xticks([-0.2, 0, 0.2])
#ax2.set_yticks([-0.00008, 0, 0.00008])
transFigure = fig.transFigure.inverted()

ymin1, ymax1 = ax1.get_ylim()
ymin2, ymax2 = ax2.get_ylim()

coord1 = transFigure.transform(ax1.transData.transform([-0.2, ymin1]))
coord2 = transFigure.transform(ax2.transData.transform([-0.2, ymax2]))

line1 = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=fig.transFigure, color="darkgray", zorder=-99)
                               
coord1 = transFigure.transform(ax1.transData.transform([0.2, ymin1]))
coord2 = transFigure.transform(ax2.transData.transform([0.2, ymax2]))
line2 = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=fig.transFigure, color="darkgray", zorder=-99)
                               
#fig.lines = line1, line2

for spine in ax2.spines.values():
    spine.set_edgecolor("darkgray")


# New figure
JAPsi_dx_zavg = obj["JAPsi_dx_zavg"].value
nablasqAterm_dx_zavg = obj["nablasqAterm_dx_zavg"].value
Aterm1_dx_zavg = obj["Aterm1_dx_zavg"].value
JAPsi_dx_slice = obj["JAPsi_dx_slice"].value
nablasqAterm_dx_slice = obj["nablasqAterm_dx_slice"].value
Aterm1_dx_slice = obj["Aterm1_dx_slice"].value

fig = plt.figure(figsize=(6, 6), facecolor="white")
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(xgrid, Bzinitial_zavg, color="gray", label=r"$B_z^{0}$", lw=lw)
ax1.plot(xgrid, Bzfinal_zavg, color="black", label=r"$B_z^{sat}$", lw=lw)
ax1.set_yticks([np.min(Bzfinal_zavg), 1, np.max(Bzfinal_zavg)])
ax1.set_yticklabels([r"$0.9985$", r"$1$", r"$1.0025$"])
ax1.set_title(r"$B_z^{saturated}$")
ax1.text(-0.95, 1.00002, r"$B_z^{0}$", color="darkgray")

if zavg is True:
    ax2.plot(xgrid, JAPsi_dx_zavg, label=r"$\partial_x (J(A, \Psi))$")
    ax2.plot(xgrid, nablasqAterm_dx_zavg, label=r"$\partial_x (\nabla^2 A)$")
    ax2.plot(xgrid, Aterm1_dx_zavg, label=r"$-\partial_x (B_0 \partial_z \Psi) $")
else:
    ax2.plot(xgrid, JAPsi_dx_slice, "--", color=colors[0], lw=lw, label=r"$\partial_x (J(A, \Psi))$")
    ax2.plot(xgrid, nablasqAterm_dx_slice, ":", color=colors[1], lw=lw, label=r"$\partial_x (\nabla^2 A)$")
    ax2.plot(xgrid, Aterm1_dx_slice, color=colors[2], lw=lw, label=r"$-\partial_x (B_0 \partial_z \Psi) $")
ax2.set_xlabel(r"$x$", size=15)
plt.legend(prop={'size':10}, bbox_to_anchor=(0.95, 0.3))

#ax2.set_xlim(-0.2, 0.2)
#ax2.set_ylim(-0.00008, 0.00008)
#ax2.set_xticks([-0.2, 0, 0.2])
#ax2.set_yticks([-0.00008, 0, 0.00008])
transFigure = fig.transFigure.inverted()

ymin1, ymax1 = ax1.get_ylim()
ymin2, ymax2 = ax2.get_ylim()

coord1 = transFigure.transform(ax1.transData.transform([-0.2, ymin1]))
coord2 = transFigure.transform(ax2.transData.transform([-0.2, ymax2]))

line1 = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=fig.transFigure, color="darkgray", zorder=-99)
                               
coord1 = transFigure.transform(ax1.transData.transform([0.2, ymin1]))
coord2 = transFigure.transform(ax2.transData.transform([0.2, ymax2]))
line2 = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
                               transform=fig.transFigure, color="darkgray", zorder=-99)
                               
#fig.lines = line1, line2

for spine in ax2.spines.values():
    spine.set_edgecolor("darkgray")

obj.close()
