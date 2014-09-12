import dedalus2.public as de
import numpy as np

gridnum = 256
x_basis = de.Chebyshev(gridnum)
z_basis = de.Fourier(gridnum)
domain = de.Domain([x_basis, z_basis])

mri = de.ParsedProblem(fields=['psi','u','A','b','psi_x','psi_xx','psi_xxx','u_x','A_x','b_x'],
                       parameters=['Re','Rm','B0','Omega0'])

#streamfunction
mri.add_equation("dt(dx(psi_x) + dz(dz(psi))) - 2*dz(u) + (dx(psi_xxx) + dz(dz(dz(dz(psi)))))/Re - B0/4*pi ")

#u


