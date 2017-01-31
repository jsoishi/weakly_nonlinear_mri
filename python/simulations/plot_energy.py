import sys
import re
import pathlib

import h5py 
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['lines.linewidth']=2
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from parse_params import parse_params

def latex_float(f):
    # from http://stackoverflow.com/a/13490601
    float_str = "{0:5.2e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def compute_growth(f, t, period, start, stop, g_scale=80., verbose=True):
    """compute a growth rate gamma for given timeseries f sampled at
    points t, assuming an exponential growth:
    
    f(t) = f0 exp(gamma t)

    inputs:
    f -- timeseries
    t -- time points
    period -- the unit for t
    start -- beginning of timeseries to fit in units of period
    stop -- end of timeseries to fit in units of period

    outputs:
    f0 -- t=0 value
    gamma -- growth rate

    """
    t_window = (t/period > start) & (t/period < stop)

    gamma_f, log_f0 = np.polyfit(t[t_window], np.log(f[t_window]),1)

    return gamma_f, np.exp(log_f0)



if __name__ == "__main__":
    datadir = sys.argv[-1]
    base = pathlib.Path(datadir)

    f = base.joinpath("scalar/scalar_s1.h5")

    params = parse_params(str(base.stem), "MRI_run")

    t_orb = 2*np.pi
    start = 2.5#125
    stop = 5#190
    e_upper = 10
    e_lower = 1e-6
    with h5py.File(str(f),'r') as ts:
        t=  ts['/scales/sim_time'][:]
        u_rms = ts['/tasks/vx_rms'][:,0,0]
        gamma, f0 = compute_growth(u_rms, t, t_orb, start, stop)
        plt.subplot(211)
        TE = ts['/tasks/BE'][:,0,0] + ts['/tasks/KE'][:,0,0]
        plt.semilogy(t/t_orb,TE,linestyle='-', label='total')
        plt.semilogy(t/t_orb,ts['/tasks/KE'][:,0,0],linestyle='-.', label='kinetic')
        plt.semilogy(t/t_orb,ts['/tasks/BE'][:,0,0],linestyle='--', label='magnetic')
        plt.fill_between([start,stop],e_lower,e_upper,alpha=0.4)
        plt.text(stop+0.01,1e-6,r'$\gamma = '+latex_float(gamma)+'$')
        plt.legend(loc='lower right')
        plt.ylabel("Energy")
        #plt.xlabel("time (orbits)")
        plt.ylim(e_lower, e_upper)
        plt.title(r"$\mathrm{Pm} = "+latex_float(float(params["Pm"]))+"$")

        plt.subplot(212)
        Jdot = ts['/tasks/BxBy'][:,0,0] + ts['/tasks/uxuy'][:,0,0]
        plt.semilogy(t/t_orb, Jdot, linestyle='-',label='total')
        plt.semilogy(t/t_orb, ts['/tasks/uxuy'][:,0,0], linestyle='-.', label=r'Reynolds')
        plt.semilogy(t/t_orb, ts['/tasks/BxBy'][:,0,0], linestyle='--', label=r'Maxwell')
        plt.legend(loc='lower right')
        plt.ylabel(r"$\dot{J}$")
        plt.xlabel("time (orbits)")

    outfile = "../../figs/kinetic_energy_jdot_{}.png".format(base.parts[-1])
    plt.savefig(str(outfile))
