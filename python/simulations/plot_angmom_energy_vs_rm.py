import h5py 
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

runs = ['Rm5.00e+00_eps0.00e+00_Pm1.00e-03_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz2048_Lz4_CFL_restart1',
        'Rm5.10e+00_eps0.00e+00_Pm1.00e-03_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz1024_Lz2_CFL_restart1',
        'Rm5.30e+00_eps0.00e+00_Pm1.00e-03_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz1024_Lz2_CFL_restart1',
        'Rm5.70e+00_eps0.00e+00_Pm1.00e-03_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz1024_Lz2_CFL_restart1']

Rms = np.array([5.00,
                5.10,
                5.30,
                5.70])

jdot = []
e = []
for i,r in enumerate(runs):
    ts = h5py.File('scratch/MRI_run_'+r+'/scalar/scalar_s1.h5',"r")
    jdot.append(ts['/tasks/BxBy'][-1,0,0] + ts['/tasks/uxuy'][-1,0,0])
    e.append(ts['/tasks/BE'][-1,0,0] + ts['/tasks/KE'][-1,0,0])
    ts.close()

jdot = np.array(jdot)
e = np.array(e)
q,logjdot0 = np.polyfit(np.log(Rms),np.log(jdot),1)
plt.loglog(Rms,jdot,'kx')
plt.loglog(Rms,jdot[0]*(Rms/Rms[0])**q, label=r'$Rm^{{{:4.2}}}$'.format(q))
plt.xlabel("Rm")
plt.ylabel(r"$\dot{J}$")
plt.legend(loc="lower right").draw_frame(False)

plt.savefig('../../figs/sims_angmom_vs_rm.png')

plt.clf()
q,logjdot0 = np.polyfit(np.log(Rms),np.log(e),1)
plt.loglog(Rms,e,'kx')
plt.loglog(Rms,e[0]*(Rms/Rms[0])**q, label=r'$Rm^{{{:4.2}}}$'.format(q))
plt.xlabel("Rm")
plt.ylabel(r"$E_{tot}$")
plt.legend(loc="lower right").draw_frame(False)

plt.savefig('../../figs/sims_energy_vs_rm.png')

outfilename = "../../data/energy_angmom_vs_rm.h5"

outfile = h5py.File(outfilename,"w")
outfile['Rms'] = Rms
outfile['jdot'] = jdot
outfile['e'] = e
outfile.close()
