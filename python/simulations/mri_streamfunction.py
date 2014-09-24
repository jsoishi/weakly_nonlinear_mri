import dedalus2.public as de
import numpy as np

gridnum = 256
x_basis = de.Chebyshev(gridnum)
z_basis = de.Fourier(gridnum)
domain = de.Domain([x_basis, z_basis])

mri = de.ParsedProblem(['z','x'],
                       field_names=['psi','u','A','b','psi_x','psi_xx','psi_xxx','u_x','A_x','A_xx', 'b_x'],
                       param_names=['Re','Rm','B0','Omega0','q','pi'])

#streamfunction
mri.add_equation("dt(dx(psi_x)) + dz(dz(dt(psi))) - 2*dz(u) + (dx(psi_xxx) + dz(dz(dz(dz(psi)))))/Re - B0*(dz(A_xx) + dz(dz(dz(A))))/(4*pi) = pi/4*((dx(A_xx) + dz(dz(Ax))) * dz(A) - (dz(A_xx) + dz(dz(dz(A))))*A_x) - ((psi_xxx + dz(dz(A_x))) * dz(psi) - (dz(psi_xx) + dz(dz(dz(psi))))*psi_x)")

#u (y-velocity)
mri.add_equation("dt(u) + (2-q)*Omega0*dz(psi) - B0/4*pi * dz(b) - (dx(u_x) + dz(dz(u)))/Re = (dz(A) * b_x - A_x * dz(b))/(4.*pi)")

#vector potential
mri.add_equation("dt(A) - B0 * dz(psi) - (A_xx + dz(dz(A)))/Rm = dz(a) * psi_x - A_x * dz(psi)")

#b (y-field)
mri.add_equation("dt(b) - B0*dz(u) + q*Omega0 * dz(A) - (dx(b_x) + dz(dz(b)))/Rm = dz(A) * u_x - A_x * dz(u) - (b_x*dz(psi) - dz(b)*psi_x)")

# first-order scheme definitions
mri.add_equation("psi_x + dx(psi) = 0")
mri.add_equation("psi_xx + dx(psi_x) = 0")
mri.add_equation("psi_xxx + dx(psi_xx) = 0")
mri.add_equation("u_x + dx(u) = 0")
mri.add_equation("A_x + dx(A) = 0")
mri.add_equation("A_xx + dx(A_x) = 0")
mri.add_equation("b_x + dx(b) = 0")

