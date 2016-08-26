import re
import glob
import h5py
import numpy as np
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

outfilename = "../../data/center_amplitude_vs_time.h5"
basename = "scratch/MRI_run_Rm5.00e+00_eps0.00e+00_Pm{:5.2e}_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz128_Lz2_CFL"

pms = np.array([3e-4,1e-3,3e-3])

outdata = h5py.File(outfilename,"w")

for pm in pms:
    base = pathlib.Path(basename.format(pm))
    f = base.joinpath("slices")

    datafiles = glob.glob(str(f.joinpath("slices_s*.h5")))
    datafiles.sort(key=lambda m: int(re.match("[\D\d]*slices_s(\d+).h5",m).group(1)))
    u_center = []
    t = []

    for df in datafiles:
        print(df)

        with h5py.File(df,"r") as data:
            nt,nz,nx = data['/tasks/u'].shape

            u_center.append(data['/tasks/u'][:,nz/2,nx/2])
            t.append(data['/scales/sim_time'][:])
    datapath = str(pm) + "/u_center"
    scalepath = str(pm) + "/t"
    outdata[datapath] = np.concatenate(u_center)
    outdata[scalepath] = np.concatenate(t)

outdata.close()

