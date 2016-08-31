import sys
import dedalus.public as de
import h5py
import numpy 

from parse_params import parse_params


outfilename = "../../data/ang_mom_vs_time.h5"
basename = "scratch/MRI_run_Rm5.00e+00_eps0.00e+00_Pm{:5.2e}_beta2.50e+01_Q7.50e-01_qsh1.50e+00_Omega1.00e+00_nx128_nz128_Lz2_CFL"


pms = np.array([3e-4,1e-3,3e-3])
params = parse_params(basename.format(pms[0]))
nx = int(params['nx'])
nz = int(params['nz'])
Lz = float(params['Lz'])
beta = float(params['beta'])

d = de.Domain([de.Fourier('z',nz,interval=[0,Lz],dealias=3/2),de.Chebyshev('x',nx,interval=[-1.,1.],dealias=3/2)],grid_dtype='float64')

ux = d.new_field(name='ux')
uz = d.new_field(name='uz')
u = d.new_field(name='u')
psi = d.new_field(name='psi')
A = d.new_field(name='A')
bx = d.new_field(name='bx')
bz = d.new_field(name='bz')

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

                A.differentiate('z',out=Ax)
                A.differentiate('x',out=Az)
                Az['g'] *= -1

                reyn_stress['g'] = ux['g']*u['g']
                maxw_stress['g'] = -2/beta * bx['g']*b['g']
            
                reyn.append(reyn_stress.integrate()['g'][0][0])
                maxw.append(maxw_stress.integrate()['g'][0][0])

    scalepath = str(pm) + "/t"
    outdata[str(pm) + '/reyn'] = np.concatenate(reyn)
    outdata[str(pm) + '/maxw'] = np.concatenate(maxw)
    outdata[scalepath] = np.concatenate(t)

outdata.close()

