"""
plot growth rate as a function of B0 to demonstrate that *lowering* B0 at Rm_c, Q does NOT increase growth rate. In fact, it actually makes the MRI STABLE rather that UNstable...

"""
import dedalus.public as de
from allorders_2 import OrderE
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


datafile = "../data/growth_rates_B0_crit.h5"

zoom_start=0.8855
zoom_stop =0.8865
abs_B0 = np.concatenate((np.array([0.1,0.75]),np.linspace(zoom_start,zoom_stop,50),np.array([1.0,1.10,1.15,1.25,1.5])))

# B0 = np.zeros(2*len(abs_B0))
# for i,B in enumerate(abs_B0[::-1]):
#     if i == 0:
#         B0[i] = -B
#         B0[-1] = B
#     else:
#         B0[i] = -B
#         B0[-(i+1)] = B
B0 = abs_B0
def growth_rate(domain,B0, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0):
    lev = OrderE(lev_domain, Q=Q, Rm=Rm, Pm=Pm, q=q, beta=beta, B0=B0)
    growth_rate = np.max(lev.EP.evalues_good.real)
    
    return growth_rate

try:
    data = h5py.File(datafile,"r")
except OSError:
    # if file doesn't exist, assume data needs to be created.
    nx = 32
    x = de.Chebyshev('x', nx, interval=[-1., 1.])
    lev_domain = de.Domain([x,])

    growth_rates = []
    for B in B0:
        growth_rates.append(growth_rate(lev_domain,B))

    data = h5py.File(datafile,'w')
    data['B0'] = B0
    data['growth_rates'] = growth_rates


fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.85,0.85])
ax_inset = fig.add_axes([0.6,0.6,0.2,0.2])
ax.plot(data['B0'],data['growth_rates'],lw=2,color='black')
ax.plot(data['B0'],data['growth_rates'],'kx')

ax_inset.plot(data['B0'],data['growth_rates'],lw=2,color='black')
ax_inset.plot(data['B0'],data['growth_rates'],'kx')
ax_inset.set_xlim(zoom_start,zoom_stop)
ax_inset.set_ylim(-0.0180,-0.0176)
ax_inset.xaxis.set_major_locator(plt.MaxNLocator(3))

ax.axhline(0,alpha=0.25,color='k')
ax.set_xlabel(r"$B_0$")
ax.set_ylabel(r"$\gamma$")
fig.savefig("../figs/growth_vs_B0.png",dpi=300)
data.close()
