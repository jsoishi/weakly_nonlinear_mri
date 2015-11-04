import sys
import pathlib

import h5py 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

datadir = sys.argv[-1]
base = pathlib.Path(datadir)

f = base.joinpath("scalar/scalar_s1.h5")
with h5py.File(str(f),'r') as ts:
    plt.semilogy(ts['/scales/sim_time'],ts['/tasks/KE'][:,0,0],marker='o',linestyle='-')

plt.ylabel("Kinetic Energy")
plt.xlabel("time")

outfile = base.joinpath("kinetic_energy.png")
plt.savefig(str(outfile))
