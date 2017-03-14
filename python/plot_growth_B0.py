"""
plot growth rate as a function of B0 to demonstrate that *lowering* B0 at Rm_c, Q does NOT increase growth rate. In fact, it actually makes the MRI STABLE rather that UNstable...

"""
import dedalus.public as de
from allorders_2 import OrderE
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as patches
rc('text', usetex=True)


def growth_rate(domain,B0, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0):
    lev = OrderE(domain, Q=Q, Rm=Rm, Pm=Pm, q=q, beta=beta, B0=B0)
    growth_rate = np.max(lev.EP.evalues_good.real)
    
    return growth_rate


if __name__ == "__main__":
    datafile = "../data/growth_rates_B0_crit.h5"
    zoom_start=0.8855
    zoom_stop =0.8865
    zoom = np.linspace(zoom_start,zoom_stop,26)
    zoom_neg = np.linspace(-zoom_start,-zoom_stop,26)
    B0_coarse = np.linspace(-1.5,1.5,100)
    B0 = np.concatenate((B0_coarse,zoom,zoom_neg))
    B0.sort()
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
    ax = fig.add_axes([0.1,0.1,0.85,0.8])
    ax_inset = fig.add_axes([0.6,0.6,0.2,0.2])
    ax.plot(data['B0'],data['growth_rates'],lw=2,color='black')
    ax.scatter([-1,1],[0,0],marker='o')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-0.02,0.06)

    # inset
    zoom_bottom = -0.0180
    zoom_top = -0.0176
    # rect = patches.Rectangle((zoom_start,zoom_bottom), width=(zoom_stop-zoom_start), height=np.abs(zoom_bottom-zoom_top),
    #                      transform=ax.transData, color='yellow', alpha=0.5)
    
    rect = patches.Rectangle((0.8,-0.01875), width=0.15, height=0.003,
                             transform=ax.transData, fill=False, alpha=0.2)
    ax.add_patch(rect)
    end = ax.transData.inverted().transform(fig.transFigure.transform((0.6,0.6)))
    end2 = ax.transData.inverted().transform(fig.transFigure.transform((0.8,0.6)))

    ax.plot([0.8,end[0]],[-0.01875+0.003,end[1]],color='black',alpha=0.2)
    ax.plot([0.8+0.15,end2[0]],[-0.01875+0.003,end2[1]],color='black',alpha=0.2)
    
    ax_inset.plot(data['B0'],data['growth_rates'],lw=2,color='black')
    ax_inset.plot(data['B0'],data['growth_rates'],'kx')
    ax_inset.set_xlim(zoom_start,zoom_stop)
    ax_inset.set_ylim(zoom_bottom,zoom_top)
    ax_inset.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax_inset.yaxis.set_major_locator(plt.MaxNLocator(3))

    ax.axhline(0,alpha=0.5,color='k')
    ax.set_xlabel(r"$B_0$",size=15)
    ax.set_ylabel(r"$\gamma$",size=15)
    fig.savefig("../figs/growth_vs_B0.png",dpi=300)
    data.close()
