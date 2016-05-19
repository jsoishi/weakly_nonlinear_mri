import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import h5py

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

inpath = "/Users/susanclark/weakly_nonlinear_mri/python/thinsims/"
mrirun = "MRI_run_Rm5.00e+00_eps0.00e+00_Pm1.00e-04_beta2.50e+01_Q7.49e-01_qsh1.50e+00_Omega1.00e+00_nz256/"

slicepath = inpath + mrirun + "/slices/"

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_title(r"$\Psi$", size = 20)
ax2.set_title(r"$u_y$", size = 20)

"""
for slice_i in xrange(1000, 1123, 1):

    fn = slicepath + "slices_s" + str(slice_i) + ".h5"
    slice = h5py.File(fn, 'r')
    
    psi = slice['tasks']['psi'][0, :, :]
    u = slice['tasks']['u'][0, :, :]
    A = slice['tasks']['A'][0, :, :]
    B = slice['tasks']['b'][0, :, :]
    
    ax1.imshow(u)
    plt.show()
"""

def init():
    im1 = ax1.imshow(np.zeros((256, 256), np.float_))
    im2 = ax2.imshow(np.zeros((256, 256), np.float_))
    
    return im1, im2
    
def animate(i):

    slice_i = i + 1000
    
    try:
        fn = slicepath + "slices_s" + str(slice_i) + ".h5"
        slice = h5py.File(fn, 'r')
    except IOError:
        print fn
    
    psi = slice['tasks']['psi'][0, :, :]
    u = slice['tasks']['u'][0, :, :]
    
    im1 = ax1.imshow(psi, cmap = "RdBu")
    im2 = ax2.imshow(u, cmap = "RdBu")

    return im1, im2
    
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=122, interval=20, blit=True)

anim.save('test_anim_psi_u.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    