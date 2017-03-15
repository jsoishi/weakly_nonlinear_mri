import sys 
import glob
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import dedalus.public as de
from parse_params import parse_params
from filter_field import filter_field_onedim

def calc_ang_mom(nt, data, fields, frac = None):
    """
    calculate angular momentum transport. if frac is not None, filter
    the input fields in the z direction first to construct the
    transport from only low wavenumber modes.

    """
    for f in ['u_kspace','A_kspace','psi_kspace','b_kspace']:
        fields[f]['c'] = data['tasks'][f][nt,:,:]
        if frac:
            filter_field_onedim(fields[f],0,frac=frac)
        fields[f].set_scales(domain.dealias)

    fields['Re_stress']['g'] = fields['u_kspace']['g'] * fields['psi_kspace'].differentiate('z')['g']
    fields['Ma_stress']['g'] = fields['b_kspace']['g'] * fields['A_kspace'].differentiate('z')['g']
    z_int = fields['Ma_stress'].domain.bases[0].interval
    x_int = fields['Ma_stress'].domain.bases[1].interval
    Lz = z_int[1] - z_int[0]
    Lx = x_int[1] - x_int[0]
    avg_Ma_stress = fields['Ma_stress'].integrate('x').integrate('z')['g'][0,0]/(Lz*Lx)
    avg_Re_stress = fields['Re_stress'].integrate('x').integrate('z')['g'][0,0]/(Lz*Lx)
    
    return avg_Ma_stress, avg_Re_stress, fields['u_kspace']['g']

basedir = sys.argv[-1]
filterfrac = 0.003

basedir = basedir.rstrip('/')
print("basedir")
print(basedir.split('/'))

run_name = basedir.split('/')[-1]

basename = 'MRI_run'
print(run_name)
params = parse_params(run_name,basename)

nx = int(params['nx'])
nz = int(params['nz'])
slices = glob.glob(basedir+'/slices/slices_s*.h5')
slices.sort()

print("(nx, nz) = ({}, {})".format(nx,nz))

with h5py.File(slices[0],'r') as data:
    zlim = [data['scales/z/1.0'][0],data['scales/z/1.0'][-1]]

xb = de.Chebyshev('x', nx, dealias=3/2)
zb = de.Fourier('z', nz, interval=zlim, dealias=3/2)
domain = de.Domain([zb,xb],grid_dtype='float')
field_names = ['psi_kspace','u_kspace', 'A_kspace', 'b_kspace', 'Re_stress','Ma_stress']
fields = {}
for f in field_names:
    fields[f] = domain.new_field(name=f)

for f in ['Re_stress','Ma_stress']:
    fields[f].set_scales(domain.dealias)

t = []
re = []
ma = []
uu = []
for sl in slices:
    with h5py.File(sl,'r') as data:
        nwrites = data['scales/sim_time'].shape[0]

        for nt in range(nwrites):
            m, r, u = calc_ang_mom(nt,data, fields, frac=filterfrac)
            t.append(data['scales/sim_time'][nt])
            re.append(r)
            ma.append(m)
            uu.append(u)

t = np.array(t)
re = np.array(re)
ma = np.array(ma)
uu = np.array(uu)
outfilename = "scratch/data/ang_mom_" + run_name + "_filter{}.h5".format(filterfrac)
with h5py.File(outfilename,"w") as outfile:
    outfile['time'] = t
    outfile['Reyn_stress'] = re
    outfile['Maxw_stress'] = ma
    outfile['u'] = uu


