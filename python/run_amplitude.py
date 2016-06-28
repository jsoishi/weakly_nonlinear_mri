import dedalus.public as de
import numpy as np
from mpi4py import MPI
from allorders_2 import AmplitudeAlpha

# Q = 0.74955
# Rm = 4.898
# Pm = 5e-3
# Q = 0.7467
# Rm = 4.876
# Pm = 5e-4

Q = 0.7470
Rm = 4.879
Pm = 1e-3

gridnum = 50
x_basis = de.Chebyshev('x',gridnum)
domain = de.Domain([x_basis], np.complex128, comm=MPI.COMM_SELF)
print("running at gridnum", gridnum)

aa = AmplitudeAlpha(domain,Q = Q,Rm = Rm, Pm = Pm)
aa.print_coeffs()
