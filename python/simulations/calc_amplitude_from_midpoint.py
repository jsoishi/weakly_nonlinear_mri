import re
import glob
import h5py
import numpy as np
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

outfilename = "../../data/rms_amplitude_vs_time.h5"
basename = "scratch/MRI_run_Rm5.00e+00_eps0.00e+00_Pm{:5.2e}_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz128_Lz2_CFL"

pms = np.array([3e-4,1e-3,3e-3])

outdata = h5py.File(outfilename,"w")

for pm in pms:
    base = pathlib.Path(basename.format(pm))
    f = base.joinpath("slices")

    datafiles = glob.glob(str(f.joinpath("slices_s*.h5")))
    datafiles.sort(key=lambda m: int(re.match("[\D\d]*slices_s(\d+).h5",m).group(1)))
    u_rms = []
    t = []

    for df in datafiles:
        print(df)

        with h5py.File(df,"r") as data:
            nt,nz,nx = data['/tasks/u'].shape

            urms_z = np.sqrt((data['/tasks/u'][:,:,nx/2]**2).sum(axis=1)/nz)
            u_rms.append(urms_z)
            t.append(data['/scales/sim_time'][:])
    datapath = str(pm) + "/u_rms"
    scalepath = str(pm) + "/t"
    outdata[datapath] = np.concatenate(u_rms)
    outdata[scalepath] = np.concatenate(t)

outdata.close()

