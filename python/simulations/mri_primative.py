import numpy as np
import time
import os
import sys
import checkpointing

import logging
logger = logging.getLogger(__name__)

import dedalus2.public as de
from dedalus2.tools import post
from dedalus2.extras import flow_tools

gridnum = 256
x_basis = de.Chebyshev(gridnum)
z_basis = de.Fourier(gridnum)
domain = de.Domain([x_basis, z_basis])

mri = de.ParsedProblem(['z','x'],
                       field_names=['u','v','w','bx','by','bz','u_x','u_x','w_x','bx_x', 'by_x', 'bz_x','p'],
                       param_names=['Re','Rm','B0','Omega0','q','beta'])

mri.add_equation("dt(bx) - (dx(bx_x) + dz(dz(bx)))/Rm = -bx*dz(dz(u)) - w*dz(bx_x) + bz*dz(u_x) + u*dz(bz)")

