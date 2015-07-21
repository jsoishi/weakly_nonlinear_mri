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
import itertools

import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def ploteigs(goodevals):

    """
    Plot real vs imaginary parts of eigenvalues.
    """

    fig = plt.figure()
    
    # Color is sign of imaginary part
    colors = ["blue" for i in range(len(goodevals))]
    imagpos = np.where(goodevals.imag >= 0)
    for p in imagpos[0]:
        colors[p] = "red"
  
    # Symbol is sign of real part
    symbols = ["." for i in range(len(goodevals))]
    thickness = np.zeros(len(goodevals))
    realpos = np.where(goodevals.real >= 0)
    for p in realpos[0]:
        symbols[p] = "+"
        thickness[p] = 2
    
    print("Number of positive real parts", len(realpos[0]))
    
    for x, y, c, s, t in zip(np.abs(goodevals.real), np.abs(goodevals.imag), colors, symbols, thickness):
        plt.plot(x, y, s, c=c, alpha = 0.5, ms = 8, mew = t)
        
    # Dummy plot for legend
    plt.plot(0, 0, '+', c = "red", alpha = 0.5, mew = 2, label = r"$\gamma \geq 0$, $\omega > 0$")
    plt.plot(0, 0, '+', c = "blue", alpha = 0.5, mew = 2, label = r"$\gamma \geq 0$, $\omega < 0$")
    plt.plot(0, 0, '.', c = "red", alpha = 0.5, label = r"$\gamma < 0$, $\omega > 0$")
    plt.plot(0, 0, '.', c = "blue", alpha = 0.5, label = r"$\gamma < 0$, $\omega < 0$")
        
    plt.legend()
        
    plt.loglog()
    plt.xlabel(r"$\left|\gamma\right|$", size = 15)
    plt.ylabel(r"$\left|\omega\right|$", size = 15, rotation = 0)
    plt.title(r"$\mathrm{MRI}$ $\mathrm{eigenvalues}$ $\lambda = \gamma + i \omega$")
    
    #pylab.savefig("mri_eigenvalues.png", dpi = 500)
    

def plot_paramspace(path="/Users/susanclark/weakly_nonlinear_mri/python/multirun/",datafile="gridnum_128_Pm_0.001_Q_0.74_dQ_0.001_Rm_4.87_dRm_0.001.p",quiet=False):
    data = pickle.load(open(path+datafile, "rb"))
    
    if not quiet:
        print(len(data))
    
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
    

    for i in range(len(data)):
        jj = data.popitem()
        if not quiet:
            print(jj)
        if np.isnan(jj[1]) == True:
            evals[i] = None
        else:
            evals[i] = jj[1]
        #ids[i] = jj[0]
        
        allRms[i] = jj[0][0]
        allQs[i] = jj[0][1]
        
        #if evals[i] > 1:
        #print(allRms[i], allQs[i], evals[i])
    
    """
    all_possible = np.zeros(len(Qs), np.int)
    all_possible[ids.astype(int)] = 1
    
    allQs = np.zeros(len(Qs), np.float)
    allRms = np.zeros(len(Qs), np.float)
    allevals = np.zeros(len(Qs), np.complex128)
    
    allQs[all_possible == True] = [Qs[i] for i in ids.astype(int)]
    allRms[all_possible == True] = [Rms[i] for i in ids.astype(int)]
    allevals[all_possible == True] = evals
    allevals[all_possible == False] = np.nan + np.nan*1j
    """
    #print(allevals)
    #print(allRms)
    #print(allevals)
    
    #Q = Qs[ids.astype(int)]
    #Rm = Rms[ids.astype(int)]
    Q = allQs
    Rm = allRms
    #evals = allevals
    
    # Some values are crazy!
    #evals.real[evals.real > 1] = None
    #evals.real[evals.real < -1E-3] = None

    #e = evals[ids.astype(int)]
    # Get bounds for plotting
    if np.abs(np.nanmax(evals.real)) > np.abs(np.nanmin(evals.real)):
        vmax = np.nanmax(evals.real)
        vmin = -np.nanmax(evals.real)
    else:
        vmin = np.nanmin(evals.real)
        if np.nanmin(evals.real) < 0:
            vmax = -np.nanmin(evals.real)
        else:
            vmax = np.nanmax(evals.real)
            
    
    vmin = np.nanmin(evals.real)#-1e-7
    vmax = np.nanmax(evals.real)#1e-7

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    cb = ax1.scatter(Q, Rm, c=np.real(evals), marker="s", s=40, vmin = vmin, vmax = vmax, cmap = "RdBu")#, vmin=-1E-7, vmax=1E-7, cmap="bone")
    fig.colorbar(cb)
    ax1.set_title("Real")
    ax1.set_xlabel("Q")
    ax1.set_ylabel("Rm")
    ax1.set_xlim(np.min(Qs) - dQ, np.max(Qs) + dQ)
    ax1.set_ylim(np.min(Rms) - dRm, np.max(Rms) + dRm)
    
    ax2 = fig.add_subplot(122)
    cb2 = ax2.scatter(Q, Rm, c=np.imag(evals), marker="s", s=40, cmap = "RdBu")
    fig.colorbar(cb2)
    ax2.set_title("Imaginary")
    ax2.set_xlabel("Q")
    ax2.set_ylabel("Rm")
    ax2.set_xlim(np.min(Q) - dQ, np.max(Q) + dQ)
    ax2.set_ylim(np.min(Rms) - dRm, np.max(Rms) + dRm)
    
    plt.tight_layout()
    plt.show()
    
    
