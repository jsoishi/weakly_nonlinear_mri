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
Pms = [0.0001]
xis = [0, 1, 2, 3, 4, 5]

# Set up file names
root = "/Users/susanclark/Projects/HMRISims/data/New1D/"
root2 = "/Users/susanclark/Projects/HMRISims/data/GRIDS/"

# Fixed parameters
nr = 32
R1 = 1
R2 = 2
Omega1 = 1
Omega2 = 0.27

#n_Rm = 15
#n_k = 11

insulate = True
    

for Pm in Pms:
    all_params_dict = {}
    
    if Pm > 0.001:
        #xis = [0, 1, 2, 3, 4, 5]
        xis = [0]
    
    # Smallest Pms only have xi >= 3
    elif Pm <= 0.001: 
        #xis = [3, 4, 5]
        xis = [4]
    
    for xi in xis:
        print("Finding critical parameters for Pm={}, xi={}".format(Pm, xi))
        
        #fortfn = root+"Pm"+str(Pm)+"/beta"+str(xi)+"/fort.99"
        #fortfn = root+"Pm"+str(Pm)+"/beta"+str(xi)+"/ConstantLambda1Dcut"
        #fortdata = np.loadtxt(fortfn)
        #(Ha, Re, k_optimal, sigma_Re, sigma_Im) = fortdata[0]
        
        # rainer provided new data 9/26
        fortfn2 = root2+"Pm"+str(Pm)+"/beta"+str(xi)+"/fort.4"
        fortdata2 = np.loadtxt(fortfn2)
        
        Rmkgrid = np.loadtxt(root2+"Pm"+str(Pm)+"/beta"+str(xi)+"/RmkGrid")
        print("min k: {} max k: {}".format(np.nanmin(Rmkgrid[:, 2]), np.nanmax(Rmkgrid[:, 2])))
        print("min Rm: {} max Rm: {}".format(np.nanmin(Rmkgrid[:, 1]), np.nanmax(Rmkgrid[:, 1])))
        print("Rmk grid Co = {}".format(Rmkgrid[0, 0]))
        
        #print("Data from {}: Ha={}, Re={}, k_opt={}, sig_Re={}, sig_Im={}".format(fortfn, Ha, Re, k_optimal, sigma_Re, sigma_Im))
        
        (Ha, Re, k_optimal) = fortdata2     
        print("New data: Ha={}, Re={}, k_opt={}".format(Ha, Re, k_optimal))
        
        
        # define grid based on Rainer's
        n_Rm = len(np.unique(Rmkgrid[:, 1]))
        n_k = len(np.unique(Rmkgrid[:, 2]))
        print("n_Rm = {}, n_k = {}".format(n_Rm, n_k))
        
        # plot Rainer's data on grid
        gridRm = Rmkgrid[:, 1]
        gridk = Rmkgrid[:, 2]
        growthratevals = Rmkgrid[:, 3]
        Imsigmas = Rmkgrid[:, 4]
        dRm = sorted(gridRm)[-1] - sorted(gridRm)[-n_k-1]
        dk = sorted(gridk)[-1] - sorted(gridk)[-n_Rm-1]
        iRm = ((gridRm - np.nanmin(gridRm)) / dRm).astype(int)
        jk = ((gridk - np.nanmin(gridk)) / dk).astype(int)
        gridvals = np.zeros((n_Rm, n_k), np.float_)
        gridvals[iRm, jk] = growthratevals
        gridImsigmas = np.zeros((n_Rm, n_k), np.float_)
        gridImsigmas[iRm, jk] = Imsigmas
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.pcolormesh(np.unique(gridk), np.unique(gridRm), gridvals, cmap="autumn")
        fig.colorbar(cax)
        plt.xlim(np.nanmin(gridk), np.nanmax(gridk))
        plt.ylim(np.nanmin(gridRm), np.nanmax(gridRm))
        plt.xlabel("k")
        plt.ylabel("Rm")
        plt.title("Rainer's growth rates for Pm {}, xi {}".format(Pm, xi))
        plt.savefig("../figs/Rainer_growth_rates_Pm_{}_xi_{}.png".format(Pm, xi))
        
        # plot Im(sigma)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.pcolormesh(np.unique(gridk), np.unique(gridRm), gridImsigmas, cmap="autumn")
        fig.colorbar(cax)
        plt.xlim(np.nanmin(gridk), np.nanmax(gridk))
        plt.ylim(np.nanmin(gridRm), np.nanmax(gridRm))
        plt.xlabel("k")
        plt.ylabel("Rm")
        plt.title("Rainer's Im(sigma) for Pm {}, xi {}".format(Pm, xi))
        plt.savefig("../figs/Rainer_Im_sigma_Pm_{}_xi_{}.png".format(Pm, xi))
        
        
        #n_Rm = 3
        #n_k = 3
        
        Rm = Pm*Re
        beta = (2*Re*Rm)/Ha**2
        #beta = (Re*Rm)/(Omega2*Ha**2)
        
        print("Rainer's Rm = {}, beta = {}, Co = {}".format(Rm, beta, 2.0/beta))
        
        # make bounds based on Ha instead, since Rm is a nonlinear func of Ha.
        #Ha_min = Ha - 0.05*Ha
        #Ha_max = Ha + 0.05*Ha
        
        #Rm_1 = (beta*Ha_min**2)/(2*Re)
        #Rm_2 = (beta*Ha_max**2)/(2*Re)
        
        #Rm_min = min(Rm_1, Rm_2)
        #Rm_max = max(Rm_1, Rm_2)

        #Rm_min = max(Rm - 10*Rm, 0.01)
        Rm_min = Rm - 0.01*Rm
        Rm_max = Rm + 0.1*Rm
        print("Rm min is {}, Rm max is {}".format(Rm_min, Rm_max))
        print("that is, Ha min = {}, Ha max = {}".format(np.sqrt((2*Re*Rm_min)/beta), np.sqrt((2*Re*Rm_max)/beta)))
        
        #k_min = max(k_optimal - 10*k_optimal, 0.01)
        k_min = k_optimal - 0.2*k_optimal
        k_max = k_optimal + 0.2*k_optimal
        print("k min is {}, k max is {}".format(k_min, k_max))
    
        Q, Rmc, omega = find_crit(comm, R1, R2, Omega1, Omega2, beta, xi, Pm, Rm_min, Rm_max, k_min, k_max, n_Rm, n_k, nr, insulate)

        Ha_crit = np.sqrt((2*Re*Rmc)/beta)
        print("critical Ha = sqrt(Re*Rm_c/beta) = {}".format(Ha_crit))
            
        all_params_dict[(Pm, xi)] = (Q, Rmc, omega, Ha_crit, Re, beta, insulate)
        
        np.save(root+"Pm"+str(Pm)+"/Pm_{0}_xi_{1}_Omega2_{2}_critical_parameters.npy".format(Pm, xi, Omega2), all_params_dict, Omega2) 



