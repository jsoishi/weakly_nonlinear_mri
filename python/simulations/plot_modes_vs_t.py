import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import rc
rc('text', usetex=True)

t_orb = 2*np.pi

#colorbrewer 9 
colors = ["#ffffcc",
          "#c2e699",
          "#78c679",
          "#31a354",
          "#006837"]

infile = sys.argv[-1]
p = pathlib.Path(infile)
runname = p.parent.parent.stem

with h5py.File(infile,"r") as df:
    for i,m in enumerate(range(0,10,2)):
        plt.semilogy(df['time'][:]/t_orb,df['u'][:,m].real,label="mode {}".format(m),color=colors[i])

plt.xlabel(r'$t/t_{orb}$',size=15)
plt.ylabel(r'$\hat{u}\hat{u}^*$',size=15)
plt.legend(loc='lower right')
for ext in ['png','pdf']:
    plt.savefig('../../figs/{}_modes_vs_t.{}'.format(runname, ext),dpi=300)
