import numpy as np
import matplotlib.pyplot as plt
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue, LinearBVP
from scipy.linalg import eig, norm
import pylab
import copy
import pickle
import plot_tools
import streamplot_uneven as su
import random
from allorders_2 import *
import itertools

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def get_parameter_space_data(data, allgoodeigs = True):

    Qsearch = data.pop("Qsearch")
    Rmsearch = data.pop("Rmsearch")
    Pm = data.pop("Pm")
    q = data.pop("q")
    beta = data.pop("beta")
    dQ = data.pop("dQ")
    dRm = data.pop("dRm")
    
    QRm = np.array(list(itertools.product(Qsearch, Rmsearch)))
    Qs = QRm[:, 0]
    Rms = QRm[:, 1]
    
    ids = np.zeros(len(data))
    evals = np.zeros(len(data), np.complex128)
    
    allRms = np.zeros(len(data))
    allQs = np.zeros(len(data))
    
    if allgoodeigs == False:
        for i in range(len(data)):
            datum = data.popitem()
            print(datum)
            if np.isnan(datum[1]) == True:
                evals[i] = None
            else:
                evals[i] = datum[1]
        
            allRms[i] = datum[0][0]
            allQs[i] = datum[0][1]
    else:
        goodeigs = {}
        for i in range(len(data)):
            datum = data.popitem()
            if np.isnan(datum[1][0]) == True:
                evals[i] = None
            else:
                evals[i] = datum[1][0]
        
            allRms[i] = datum[0][0]
            allQs[i] = datum[0][1]
            
            # Store all good eigenvalues
            goodeigs[(allRms[i], allQs[i])] = datum[1][1]
        
    return allRms, allQs, evals

def plot_paramspace_run(Rm, Q, evals):

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    cb = ax1.scatter(Q, Rm, c=np.real(evals), marker="s", s=40, cmap = "RdBu", vmin=-1E-7, vmax=1E-7)
    fig.colorbar(cb)
    ax1.set_title("Real")
    ax1.set_xlabel("Q")
    ax1.set_ylabel("Rm")
    
    ax2 = fig.add_subplot(122)
    cb2 = ax2.scatter(Q, Rm, c=np.imag(evals), marker="s", s=40, cmap = "RdBu")
    fig.colorbar(cb2)
    ax2.set_title("Imaginary")
    ax2.set_xlabel("Q")
    ax2.set_ylabel("Rm")
    
    plt.tight_layout()
    plt.show()
    pylab.savefig("scrap.png", dpi = 100)

if __name__ == "__main__":

    #Pm = 5.0E-3
    Pm = 1.0E-2
    
    path = "/Users/susanclark/weakly_nonlinear_mri/python/multirun/"
    #datafile = "gridnum_128_Pm_"+str(Pm)+"_Q_0.74_dQ_0.001_Rm_4.87_dRm_0.001.p"
    datafile = "gridnum_128_Pm_"+str(Pm)+"_Q_0.74_dQ_0.001_Rm_4.87_dRm_0.001_allgoodeigs.p"
    datafile = "gridnum_128_Pm_0.01_Q_0.74_dQ_0.001_Rm_4.91_dRm_0.001_allgoodeigs.p"
    datafile = "gridnum_128_Pm_0.0001_Q_0.74_dQ_0.001_Rm_4.87_dRm_0.001_allgoodeigs.p"
    datafile = "gridnum_128_Pm_0.0005_Q_0.74_dQ_0.001_Rm_4.87_dRm_0.001_allgoodeigs.p"

    pspace_grid = pickle.load(open(path+datafile, "rb"))
    
    search_Rms, search_Qs, max_evals = get_parameter_space_data(pspace_grid, allgoodeigs = True)
    
    plot_paramspace_run(search_Rms, search_Qs, max_evals)
    
    # Modes with Re{eval} > 0 are growing
    growing_modes = np.zeros(len(max_evals), np.int)
    growing_modes[max_evals.real > 0] = 1
    
    if np.sum(growing_modes) < 1:
        raise ValueError(" There are no growing modes in this run! ") 
    
    # Critical eigenvalues are the smallest Rm with a growing mode, and its associated vertical wavenumber Q
    marginal_indx = np.nanargmin(search_Rms[np.where(growing_modes == 1)])
    marginal_Rm = search_Rms[marginal_indx]
    marginal_Q = search_Qs[marginal_indx]
    
    print("Critical Rm is", marginal_Rm, " critical Q is", marginal_Q)
    
    
    