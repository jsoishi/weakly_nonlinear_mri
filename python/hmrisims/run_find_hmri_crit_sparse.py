"""
wrapper for find_hmri_crit_sparse.py

takes Rainer's critical parameters as starting point for sparse eigenvalue search
finds critical Rm, k for HMRI
"""
from mpi4py import MPI
from eigentools import Eigenproblem, CriticalFinder
import time
import dedalus.public as de
import numpy as np
import matplotlib.pylab as plt
from scipy import special

from mpi4py import MPI
comm = MPI.COMM_WORLD

from find_hmri_crit_sparse import find_crit

# First we need critical values as computed by Rainer
#Pms = [1, 0.1, 0.01, 0.001, 0.0001]
Pms = [0.0001, 0.001, 0.01, 0.1, 1]
xis = [0, 1, 2, 3, 4, 5]

# Set up file names
root = "/Users/susanclark/Projects/HMRISims/data/New1D/"

# Fixed parameters
nr = 64
R1 = 1
R2 = 2
Omega1 = 1
Omega2 = 0.27

n_Rm = 11
n_k = 11

insulate = True
    

for Pm in Pms:
    all_params_dict = {}
    
    if Pm > 0.001:
        xis = [0, 1, 2, 3, 4, 5]
    
    # Smallest Pms only have xi >= 3
    elif Pm <= 0.001: 
        #xis = [3, 4, 5]
        xis = [4]
    
    for xi in xis:
        print("Finding critical parameters for Pm={}, xi={}".format(Pm, xi))
        
        #fortfn = root+"Pm"+str(Pm)+"/beta"+str(xi)+"/fort.99"
        fortfn = root+"Pm"+str(Pm)+"/beta"+str(xi)+"/ConstantLambda1Dcut"
        fortdata = np.loadtxt(fortfn)
        (Ha, Re, k_optimal, sigma_Re, sigma_Im) = fortdata[0]
        
        print("Data from {}: Ha={}, Re={}, k_opt={}, sig_Re={}, sig_Im={}".format(fortfn, Ha, Re, k_optimal, sigma_Re, sigma_Im))
        
        Rm = Pm*Re
        beta = (2*Re*Rm)/Ha**2
        
        print("Rainer's Rm = {}, beta = {}, Co = {}".format(Rm, beta, 2.0/beta))
        
        # make bounds based on Ha instead, since Rm is a nonlinear func of Ha.
        Ha_min = Ha - 0.007*Ha
        Ha_max = Ha + 0.001*Ha
        
        Rm_1 = (beta*Ha_min**2)/(2*Re)
        Rm_2 = (beta*Ha_max**2)/(2*Re)
        
        Rm_min = min(Rm_1, Rm_2)
        Rm_max = max(Rm_1, Rm_2)

        #Rm_min = max(Rm - 10*Rm, 0.01)
        #Rm_min = Rm - 0.005*Rm
        #Rm_max = Rm + 0.01*Rm
        print("Rm min is {}, Rm max is {}".format(Rm_min, Rm_max))
        print("that is, Ha min = {}, Ha max = {}".format(np.sqrt((2*Re*Rm_min)/beta), np.sqrt((2*Re*Rm_max)/beta)))
        
        #k_min = max(k_optimal - 10*k_optimal, 0.01)
        k_min = k_optimal - 0.015*k_optimal
        k_max = k_optimal + 0.005*k_optimal
        print("k min is {}, k max is {}".format(k_min, k_max))
    
        Q, Rmc, omega = find_crit(comm, R1, R2, Omega1, Omega2, beta, xi, Pm, Rm_min, Rm_max, k_min, k_max, n_Rm, n_k, nr, insulate)

        Ha_crit = np.sqrt((2*Re*Rmc)/beta)
        print("critical Ha = sqrt(Re*Rm_c/beta) = {}".format(Ha_crit))
            
        all_params_dict[(Pm, xi)] = (Q, Rmc, omega, Ha_crit, Re, beta, insulate)
        
    np.save(root+"Pm"+str(Pm)+"/Pm_{}_all_xi_Omega2_{}_critical_parameters.npy".format(str(Pm)), all_params_dict, Omega2) 



