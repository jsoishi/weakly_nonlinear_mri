"""
Driver to find critical Rm, k_z for widegap MRI 

Usage:
    find_widegap_crit.py [--R1=<R1> --R2=<R2> --Omega1=<Omega1> --Omega2=<Omega2> --Pm=<Pm> --beta=<beta> --Rm_min=<Rm_min> --Rm_max=<Rm_max> --k_min=<k_min> --k_max=<k_max> --n_Rm=<n_Rm> --n_k=<n_k> --n_r=<n_r>]

Options:

    --R1=<R1>                  inner cylinder Radius (in cm) [default: 5.]
    --R2=<R2>                  outer cylinder Radius (in cm) [default: 15.]
    --Omega1=<Omega1>          inner cylinder rotation rate (units of 1/s) [default: 313.55]
    --Omega2=<Omega2>          outer cylinder rotation rate (units of 1/s) [default: 67.0631]
    --Pm=<Pm>                  magnetic Prandtl number [default: 1e-3]
    --beta=<beta>              plasma beta [default: 25.]
    --xi=<xi>                  helical base field strength for HMRI [default: 0.]
    --Rm_min=<Rm_min>          minimum magnetic Reynolds number [default: 0.5]
    --Rm_max=<Rm_max>          maximum magnetic Reynolds number [default: 2.0]
    --k_min=<k_min>            minimum z wavenumber [default: 0.001]
    --k_max=<k_max>            maximum z wavenumber [default: 0.2]
    --n_Rm=<n_Rm>              number of points on Rm grid [default: 20]
    --n_k=<n_k>                number of points on k grid [default: 20]
    --n_r=<n_r>                number of points on Chebyshev r grid for eigenvalue solution [default: 100]
"""
from docopt import docopt
from find_crit import find_crit

# parse arguments
args = docopt(__doc__)


R1 = float(args['--R1'])
R2 = float(args['--R2'])
Omega1 = float(args['--Omega1'])
Omega2 = float(args['--Omega2'])
Pm = float(args['--Pm'])
beta = float(args['--beta'])
xi = float(args['--xi'])
Rm_min = float(args['--Rm_min'])
Rm_max = float(args['--Rm_max'])
k_min = float(args['--k_min'])
k_max = float(args['--k_max'])
n_Rm = int(args['--n_Rm'])
n_k = int(args['--n_k'])
n_r = int(args['--n_r'])

find_crit(R1, R2, Omega1, Omega2, beta, xi, Pm, Rm_min, Rm_max, k_min, k_max, n_Rm, n_k, n_r)
