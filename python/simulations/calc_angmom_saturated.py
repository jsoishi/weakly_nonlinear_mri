import pathlib
import glob
import re
import numpy as np
import dedalus.public as de
import h5py
import numpy 

from parse_params import parse_params

outfilename = "../../data/ang_mom_vs_time.h5"
basename = "scratch/MRI_run_Rm5.00e+00_eps0.00e+00_Pm{:5.2e}_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz128_Lz2_CFL"


pms = np.array([3e-4,1e-3,3e-3])
tmp_str = basename.format(pms[0])

params = parse_params(tmp_str, "MRI_run")

nx = int(params['nx'])
nz = int(params['nz'])
Lz = float(params['Lz'])
Lx = 2.
beta = float(params['beta'])

d = de.Domain([de.Fourier('z',nz,interval=[0,Lz],dealias=3/2),de.Chebyshev('x',nx,interval=[-1.,1.],dealias=3/2)],grid_dtype='float64')

ux = d.new_field(name='ux')
uz = d.new_field(name='uz')
u = d.new_field(name='u')
psi = d.new_field(name='psi')
A = d.new_field(name='A')
Bx = d.new_field(name='bx')
Bz = d.new_field(name='bz')

b = d.new_field(name='b')
ME = d.new_field(name='ME')
KE = d.new_field(name='KE')
reyn_stress = d.new_field(name='Tu')
maxw_stress = d.new_field(name='Tb')

outdata = h5py.File(outfilename,"w")

for pm in pms:
    base = pathlib.Path(basename.format(pm))
    f = base.joinpath("slices")

    datafiles = glob.glob(str(f.joinpath("slices_s*.h5")))
    datafiles.sort(key=lambda m: int(re.match("[\D\d]*slices_s(\d+).h5",m).group(1)))
    me = []
    ke = []
    reyn = []
    maxw = []
    t = []

    for df in datafiles:
        print(df)

        with h5py.File(df,"r") as data:
            t.append(data['/scales/sim_time'][:])
            nt,nz,nx = data['/tasks/u'].shape

            for i in range(nt):
                u['g'] = data['/tasks/u'][i,:,:]
                psi['g'] = data['/tasks/psi'][i,:,:]
                b['g'] = data['/tasks/b'][i,:,:]
                A['g'] = data['/tasks/A'][i,:,:]

                psi.differentiate('z',out=ux)
                psi.differentiate('x',out=uz)
                uz['g'] *= -1

                A.differentiate('z',out=Bx)
                A.differentiate('x',out=Bz)
                Bz['g'] *= -1

                for f in [ux,uz,Bx,Bz,psi,A,reyn_stress,maxw_stress,KE,ME]:
                    f.set_scales((1,1),keep_data=True)

                reyn_stress['g'] = ux['g']*u['g']
                maxw_stress['g'] = -2/beta * Bx['g']*b['g']
                reyn.append(reyn_stress.integrate()['g'][0][0]/(Lx*Lz))
                maxw.append(maxw_stress.integrate()['g'][0][0]/(Lx*Lz))
                # in these units, ME = 1/2 * 2/beta * B**2 = B**2/beta 
                KE['g'] = 0.5* (ux['g']**2 + u['g']**2 + uz['g']**2)
                ME['g'] = 1/beta * (Bx['g']**2 + b['g']**2 + Bz['g']**2)
                ke.append(KE.integrate()['g'][0][0])
                me.append(ME.integrate()['g'][0][0])
            

    scalepath = str(pm) + "/t"
    print("reyn shape = {}".format(len(reyn)))
    outdata[str(pm) + '/reyn'] = np.array(reyn)
    outdata[str(pm) + '/maxw'] = np.array(maxw)
    outdata[str(pm) + '/KE'] = np.array(ke)
    outdata[str(pm) + '/ME'] = np.array(me)
    outdata[scalepath] = np.concatenate(t)

outdata.close()

