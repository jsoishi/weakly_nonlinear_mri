import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import h5py
import pylab
import streamplot_uneven as su

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

inpath = "/Volumes/DataDavy/MRI/"
#mrirun = "MRI_run_Rm5.00e+00_eps0.00e+00_Pm1.00e-04_beta2.50e+01_Q7.49e-01_qsh1.50e+00_Omega1.00e+00_nz256/"
mrirun = "MRI_run_Rm5.00e+00_eps0.00e+00_Pm1.00e-02_beta2.50e+01_Q7.40e-01_qsh1.50e+00_Omega1.00e+00_nz128"

slicepath = inpath + mrirun + "/slices/"

# Set up figure for movie
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.set_title(r"$\Psi$", size = 20)
ax2.set_title(r"$u_y$", size = 20)
ax3.set_title(r"$A$", size = 20)
ax4.set_title(r"$B$", size = 20)

runsize = 128

def init():
    """ Instantiate axes instances to be animated """
    im1 = ax1.imshow(np.zeros((runsize, runsize), np.float_))
    im2 = ax2.imshow(np.zeros((runsize, runsize), np.float_))
    im3 = ax3.imshow(np.zeros((runsize, runsize), np.float_))
    im4 = ax4.imshow(np.zeros((runsize, runsize), np.float_))
    
    return im1, im2, im3, im4
    
def animate(i):

    slice_i = i + 1000
    
    try:
        fn = slicepath + "slices_s" + str(slice_i) + ".h5"
        slice = h5py.File(fn, 'r')
    except IOError:
        print fn
    
    psi = slice['tasks']['psi'][0, :, :]
    u = slice['tasks']['u'][0, :, :]
    A = slice['tasks']['A'][0, :, :]
    B = slice['tasks']['b'][0, :, :]
    
    im1 = ax1.imshow(psi, cmap = "RdBu")
    im2 = ax2.imshow(u, cmap = "RdBu")
    im3 = ax3.imshow(A, cmap = "RdBu")
    im4 = ax4.imshow(B, cmap = "RdBu")

    return im1, im2, im3, im4
    
def animate_singleslice(i):

    slice_i = 1
    
    try:
        fn = slicepath + "slices_s" + str(slice_i) + ".h5"
        slice = h5py.File(fn, 'r')
    except IOError:
        print fn
    
    psi = slice['tasks']['psi'][i, :, :]
    u = slice['tasks']['u'][i, :, :]
    A = slice['tasks']['A'][i, :, :]
    B = slice['tasks']['b'][i, :, :]
    
    im1 = ax1.imshow(psi, cmap = "RdBu")
    im2 = ax2.imshow(u, cmap = "RdBu")
    im3 = ax3.imshow(A, cmap = "RdBu")
    im4 = ax4.imshow(B, cmap = "RdBu")

    return im1, im2, im3, im4
    
def animate_all_slices(i, cmap = "RdBu"):
    
    # Set slice number from 
    slice_i = int(np.floor(i/20)) + 1
    
    # Set sub slice from mod, assuming slice has structure [20, :, :]
    sub_slice_i = i % 20
    
    print("slice {}, sub_slice {}".format(slice_i, sub_slice_i))
    
    try:
        fn = slicepath + "slices_s" + str(slice_i) + ".h5"
        slice = h5py.File(fn, 'r')
    except IOError:
        print("File {} does not exist".format(fn))
    
    psi = slice['tasks']['psi'][sub_slice_i, :, :]
    u = slice['tasks']['u'][sub_slice_i, :, :]
    A = slice['tasks']['A'][sub_slice_i, :, :]
    B = slice['tasks']['b'][sub_slice_i, :, :]
    
    im1 = ax1.imshow(psi, cmap = cmap)
    im2 = ax2.imshow(u, cmap = cmap)
    im3 = ax3.imshow(A, cmap = cmap)
    im4 = ax4.imshow(B, cmap = cmap)

    return im1, im2, im3, im4
    
def get_fluid_variables(i):
    
    # Set slice number from 
    slice_i = int(np.floor(i/20)) + 1
    
    # Set sub slice from mod, assuming slice has structure [20, :, :]
    sub_slice_i = i % 20
    
    print("slice {}, sub_slice {}".format(slice_i, sub_slice_i))
    
    try:
        fn = slicepath + "slices_s" + str(slice_i) + ".h5"
        slice = h5py.File(fn, 'r')
    except IOError:
        print("File {} does not exist".format(fn))
    
    psi = slice['tasks']['psi'][sub_slice_i, :, :]
    u = slice['tasks']['u'][sub_slice_i, :, :]
    A = slice['tasks']['A'][sub_slice_i, :, :]
    B = slice['tasks']['b'][sub_slice_i, :, :]
    
    return psi, u, A, B
    
def single_frame(i, cmap = "RdBu", savename = "test.png"):

    # Set up figure for movie
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.set_title(r"$\Psi$", size = 20)
    ax2.set_title(r"$u_y$", size = 20)
    ax3.set_title(r"$A$", size = 20)
    ax4.set_title(r"$B$", size = 20)
    
    psi, u, A, B = get_fluid_variables(i)
    
    im1 = ax1.imshow(psi, cmap = cmap)
    im2 = ax2.imshow(u, cmap = cmap)
    im3 = ax3.imshow(A, cmap = cmap)
    im4 = ax4.imshow(B, cmap = cmap)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    pylab.savefig(savename)
    
    plt.close()
    

# Number of frames should be equal to frames_per_slice * num_slices
numframes = 20*350

type = "pngs"
if type is "mp4":
    anim = animation.FuncAnimation(fig, animate_all_slices, init_func=init,
                                   frames=numframes, interval=20, blit=True)

    anim.save('test_anim_psi_u_A_B_10E-2_all.mp4', fps=20, extra_args=['-vcodec', 'libx264'])

elif type is "pngs":
    matplotlib.use('TkAgg')
    
    # Loop over drawing function and save as individual pngs
    for i in xrange(numframes):
        
        imroot = inpath + mrirun + "/figures/"
        basename = "psi_u_A_B_streams_"
        savename = imroot + basename + str(i) + ".png"
        
        single_frame(i, cmap = "RdBu", savename = savename)
        
    
    
    

    