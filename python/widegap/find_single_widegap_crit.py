"""
Driver to find critical Rm, k_z for widegap MRI 

Usage:
    find_widegap_crit.py [--R1=<R1> --R2=<R2> --Omega1=<Omega1> --Omega2=<Omega2> --Pm=<Pm> --beta=<beta> --xi=<xi> --Rm_min=<Rm_min> --Rm_max=<Rm_max> --k_min=<k_min> --k_max=<k_max> --n_Rm=<n_Rm> --n_k=<n_k> --n_r=<n_r> --insulate]

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
    --insulate                 if true, insulating boundary conditions
"""
from mpi4py import MPI
from docopt import docopt
from find_crit import find_crit

comm = MPI.COMM_WORLD

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
insulate = args['--insulate']

Q, Rmc, omega = find_crit(comm, R1, R2, Omega1, Omega2, beta, xi, Pm, Rm_min, Rm_max, k_min, k_max, n_Rm, n_k, n_r, insulate)

if comm.rank == 0:
    print("PARAMS R1 = {}".format(R1))
    print("PARAMS R2 = {}".format(R2))
    print("PARAMS Omega1 = {}".format(Omega1))
    print("PARAMS Omega2 = {}".format(Omega2))
    print("PARAMS Pm = {}".format(Pm))
    print("PARAMS beta = {}".format(beta))
    print("PARAMS xi = {}".format(xi))
    print("PARAMS Rm_min = {}".format(Rm_min))
    print("PARAMS Rm_max = {}".format(Rm_max))
    print("PARAMS k_min = {}".format(k_min))
    print("PARAMS k_max = {}".format(k_max))
    print("PARAMS n_Rm =  {}".format(n_Rm))
    print("PARAMS n_k = {}".format(n_k))
    print("PARAMS n_r = {}".format(n_r))
    print("PARAMS insulate = {}".format(insulate))

    print("OUTPUT Q = {:10.5e}".format(Q))
    print("OUTPUT Rmc = {:10.5e}".format(Rmc))
    print("OUTPUT omega = {:10.5e}".format(omega))
