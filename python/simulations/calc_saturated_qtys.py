import h5py
import numpy as np
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

basename = "scratch/MRI_run_Rm5.00e+00_eps0.00e+00_Pm{:5.2e}_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz128_Lz2_CFL_evalueIC"

pms = np.array([1e-4,2e-4,3e-4,5e-4,1e-3])
ampl = []
ke = []
me = []
reyn = []
maxw = []
for pm in pms:
    base = pathlib.Path(basename.format(pm))
    f = base.joinpath("scalar/scalar_s1.h5")

    with h5py.File(str(f),"r") as ts:
        u_rms = ts['/tasks/vx_rms'][:,0,0]

        amp = u_rms[-1]**2
        ampl.append(amp)
        ke.append(ts['/tasks/KE'][-1,0,0])
        me.append(ts['/tasks/BE'][-1,0,0])
        reyn.append(ts['/tasks/uxuy'][-1,0,0])
        maxw.append(ts['/tasks/BxBy'][-1,0,0])

ampl = np.array(ampl)
q = np.sqrt(4/3.)
qq,logampl0 = np.polyfit(np.log(pms),np.log(ampl),1)
print("<vx>_rms power law fit: {:5.2e}".format(qq))
plt.loglog(pms,ampl,'r^',label='simulations')
plt.loglog(pms,ampl[0]*(pms/pms[0])**q, label=r'$Pm^{2/3}$')
plt.loglog(pms,ampl[0]*(pms/pms[0])**qq, label=r'$Pm^{{{:4.2}}}$'.format(qq))
plt.xlabel('Pm')
plt.ylabel('Amplitude')
plt.legend(loc='upper left').draw_frame(False)
plt.savefig('../../figs/ampl_vs_pm_narrow_gap_sims.png')

plt.clf()

ke = np.array(ke)
me = np.array(me)
te = ke + me
eq, logen0 = np.polyfit(np.log(pms),np.log(te),1)
eq_k, logen0_k =  np.polyfit(np.log(pms),np.log(ke),1)
eq_m, logen0_m =  np.polyfit(np.log(pms),np.log(me),1)
print("energy power law fit: {:5.2e}".format(eq))
print("kin energy power law fit: {:5.2e}".format(eq_k))
print("mag energy power law fit: {:5.2e}".format(eq_m))
plt.loglog(pms,ke,'kx',label='kinetic')
plt.loglog(pms,me,'bo',label='magnetic')
plt.loglog(pms,te,'r^',label='total')
plt.loglog(pms,te[0]*(pms/pms[0])**eq,label=r'$Pm^{{{:4.2}}}$'.format(eq))
plt.loglog(pms,me[0]*(pms/pms[0])**(4/3.),label=r'$Pm^{4/3}$')
plt.legend(loc='lower right').draw_frame(False)
plt.xlabel('Pm')
plt.ylabel('Energy')
plt.savefig('../../figs/energy_vs_pm_narrow_gap_sims.png')

plt.clf()

reyn = np.array(reyn)
maxw = np.array(maxw)
jdot = reyn + maxw
q = 4/3.
jq,logjdot0 = np.polyfit(np.log(pms),np.log(jdot),1)
print("jdot power law fit: {:5.2e}".format(jq))
plt.loglog(pms,jdot,'r^',label='simulations')
plt.loglog(pms,jdot[0]*(pms/pms[0])**q, label=r'$Pm^{4/3}$')
plt.loglog(pms,jdot[0]*(pms/pms[0])**jq, label=r'$Pm^{{{:4.2}}}$'.format(jq))
plt.xlabel('Pm')
plt.ylabel(r'$\dot{J}$')
plt.legend(loc='upper left').draw_frame(False)
plt.savefig('../../figs/jdot_vs_pm_narrow_gap_sims.png')
