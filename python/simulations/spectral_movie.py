import sys 

import glob
import h5py
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import dedalus.public as de

def plot_frame(nt, data, log=True):
    for i,f in enumerate(field_names):
        field = domain.new_field(name=f)
        field_path = 'tasks/'+f
        field['g'] = data[field_path][nt,:,:]
        t = data['scales/sim_time'][nt]
        write_no = data['scales/write_number'][nt]
        spec = (field['c']*field['c'].conj()).real
        print("spec (min,max) = ({},{})".format(spec.min(),spec.max()))
        plt.subplot(2,2,i+1)
        #plt.imshow(np.log10(spec[:256,:256].T),interpolation='nearest',origin='lower',cmap='viridis',vmin=-20,vmax=-5)
        if log:
            plt.imshow(np.log10(spec.T),interpolation='nearest',origin='lower',cmap='viridis')#,vmin=-20,vmax=-5)
        else:
            plt.imshow(spec.T,interpolation='nearest',origin='lower',cmap='viridis')#,vmin=-20,vmax=-5)
        if i == 0:
            title = r"${:}\ (t = {:5.2f}\ t_{{orb}})$".format(f,t/period)
        else:
            title = f
        plt.title(title)
        plt.colorbar()

    filename = 'frames/spectrum_{:06d}.png'.format(write_no)
    if log:
        filename += '_log'
    #plt.savefig('frames/spectrum_trunc_{:06d}.png'.format(write_no),dpi=500)
    plt.savefig(filename,dpi=500)
    plt.clf()


basedir = sys.argv[-1]

period = 2*np.pi

slices = glob.glob(basedir+'/slices/slices_s*.h5')
slices.sort()

print(slices)

with h5py.File(slices[0],'r') as data:
    zlim = [data['scales/z/1.0'][0],data['scales/z/1.0'][-1]]

nx = 256#512#256
nz = 512#1024#4096
xb = de.Chebyshev('x', nx)
zb = de.Fourier('z', nz, interval=zlim)
domain = de.Domain([zb,xb],grid_dtype='float')
field_names = ['psi','u', 'A', 'b']

for sl in slices:
    with h5py.File(sl,'r') as data:
        nwrites = data['scales/sim_time'].shape[0]

        for nt in range(nwrites):
            plot_frame(nt,data,log=False)
