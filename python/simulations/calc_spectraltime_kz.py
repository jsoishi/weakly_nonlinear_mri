"""
Calculate spectral time diagram for kz and t, summing over chebyshev coefficients

Usage:
    calc_spectraltime_kz.py <files>... 

"""
from docopt import docopt
import pathlib
import sys

import h5py
import dedalus.public as de
from dedalus.tools.general import natural_sort

args = docopt(__doc__)
files = args['<files>']

files = natural_sort(str(sp) for sp in files)

root = pathlib.Path(files[0])
root = root.parent
outfilename = root.joinpath("spectraltime.h5")

print(files)
for i,p in enumerate(files):
    with h5py.File(p,"r") as data:
        u = (data['tasks/u_kspace'][:,:,:]*data['tasks/u_kspace'][:,:,:].conj()).sum(axis=2)
        A = (data['tasks/A_kspace'][:,:,:]*data['tasks/A_kspace'][:,:,:].conj()).sum(axis=2)
        psi = (data['tasks/psi_kspace'][:,:,:]*data['tasks/psi_kspace'][:,:,:].conj()).sum(axis=2)
        b = (data['tasks/b_kspace'][:,:,:]*data['tasks/b_kspace'][:,:,:].conj()).sum(axis=2)
        time = data['scales/sim_time'][:]
    if i == 0:
        u_spectral_time = u  
        A_spectral_time = A
        psi_spectral_time = psi
        b_spectral_time  = b
        time_scale = time
    else:
        u_spectral_time =  np.concatenate((u_spectral_time,u),axis=0)
        A_spectral_time = np.concatenate((A_spectral_time, A),axis=0)
        psi_spectral_time = np.concatenate((psi_spectral_time, psi),axis=0)
        b_spectral_time  = np.concatenate((b_spectral_time, b),axis=0)
        time_scale = np.concatenate((time_scale,time))

outfile = h5py.File(str(outfilename),"w")
outfile['psi'] = psi_spectral_time
outfile['B'] = b_spectral_time
outfile['u'] = u_spectral_time
outfile['A'] = A_spectral_time
outfile['time'] = time
outfile.close()
