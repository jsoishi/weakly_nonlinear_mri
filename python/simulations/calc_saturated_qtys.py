import h5py
import numpy as np
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

basename = "scratch/MRI_run_Rm5.00e+00_eps0.00e+00_Pm{:5.2e}_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz128_Lz2_CFL"

pms = np.array([3e-4,1e-3,3e-3])
ampl = []
ke = []
me = []
for pm in pms:
    base = pathlib.Path(basename.format(pm))
    f = base.joinpath("scalar/scalar_s1.h5")

    with h5py.File(str(f),"r") as ts:
        u_rms = ts['/tasks/vx_rms'][:,0,0]

        amp = u_rms[-1]
        ampl.append(amp)
        ke.append(ts['/tasks/KE'][-1,0,0])
        me.append(ts['/tasks/BE'][-1,0,0])

ampl = np.array(ampl)
q = np.sqrt(4/3.)
qq,logampl0 = np.polyfit(np.log(pms),np.log(ampl),1)
print("power law fit: {:5.2e}".format(qq))
plt.loglog(pms,ampl,'kx',label='simulations')
plt.loglog(pms,ampl[0]*(pms/pms[0])**q, label=r'$Pm^{4/3}$')
plt.loglog(pms,ampl[0]*(pms/pms[0])**qq, label=r'$Pm^{0.221}$')
plt.xlabel('Pm')
plt.ylabel('Amplitude')
plt.legend(loc='upper left').draw_frame(False)
plt.savefig('../../figs/ampl_vs_pm_narrow_gap_sims.png')

plt.clf()
plt.loglog(pms,ke,'kx')
plt.loglog(pms,me,'bo')
#plt.loglog(pms,ampl[0]*(pms/pms[0])**q)
#plt.loglog(pms,ampl[0]*(pms/pms[0])**(1/3))
plt.xlabel('Pm')
plt.ylabel('Energy')
plt.savefig('../../figs/energy_vs_pm_narrow_gap_sims.png')
