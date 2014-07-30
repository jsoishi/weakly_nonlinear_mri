import numpy as np
import time
import os
import sys

from dedalus2.public import *
from dedalus2.tools  import post
from dedalus2.extras import flow_tools

nx = np.int(256*3/2)
nz  = np.int(256*3/2)

data_dir = sys.argv[0].split('.py')[0]+'/'

# Set domain
Lz = 2*np.pi/Q

z_basis = Fourier(nz,   interval=[0., Lz], dealias=2/3)
x_basis = Chebyshev(nx, dealias=2/3)
domain = Domain([x_basis, z_basis], grid_dtype=np.float64)

if domain.distributor.rank == 0:
  if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))


mri_problem = ParsedProblem(axes_names=['x','z'],
                            field_names=['phi','phi_x','phi_xx','phi_xxx', 
                                         'u', 'u_x',
                                         'A', 'A_x', 'A_xx',
                                         'b','b_x'],
                            param_names=['B0', 'Re', 'Rm', 'q', 'Omega0', 'beta'])


