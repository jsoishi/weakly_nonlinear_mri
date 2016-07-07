import sys
import re
import pathlib

import h5py 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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



def parse_params(dirname,basename):
    parstr = dirname.split(basename, 1)[1].lstrip("_")
    parstr = parstr.split("_")

    params = {}
    for p in parstr:
        m = re.match("([a-zA-Z]+)([\d.+-e]+)",p)
        k, v = m.groups()
        params[k] = v

    return params


if __name__ == "__main__":
    datadir = sys.argv[-1]
    base = pathlib.Path(datadir)

    f = base.joinpath("scalar/scalar_s1.h5")

    params = parse_params(str(base.stem), "MRI_run")

    t_orb = 2*np.pi
    start = 125
    stop = 190
    with h5py.File(str(f),'r') as ts:
        t=  ts['/scales/sim_time'][:]
        u_rms = ts['/tasks/vx_rms'][:,0,0]
        gamma, f0 = compute_growth(u_rms, t, t_orb, start, stop)
        plt.subplot(211)
        plt.semilogy(t/t_orb,ts['/tasks/KE'][:,0,0],linestyle='-', label='kinetic')
        plt.semilogy(t/t_orb,ts['/tasks/BE'][:,0,0],linestyle='--', label='magnetic')
        plt.legend(loc='lower right')
        plt.ylabel("Energy")
        #plt.xlabel("time (orbits)")
        plt.title("Pm = {:5.2e}".format(float(params["Pm"])))

        plt.subplot(212)
        plt.semilogy(t/t_orb,u_rms,linestyle='-')
        plt.semilogy(t/t_orb,f0*np.exp(gamma*t),linestyle='--',label=r'$\gamma = {:5.3e}$'.format(gamma))
        plt.legend(loc='upper left')
        plt.ylabel("r$<v_x>_{rms}$")
        plt.xlabel("time (orbits)")

    outfile = "../../figs/kinetic_energy_{}.png".format(base.parts[-1])
    plt.savefig(str(outfile))
