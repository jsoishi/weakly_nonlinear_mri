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

def plot_eigenfunctions(allorders_object, savename = "scrap.png"):

    """
    Plot all eigenfunctions of a given allorders object, e.g.
    
    o1 = OrderE(**kwargs)
    plot_eigenfunctions(o1)
    
    will plot the eigenfunctions of the order E object.
    """
    
    fig = plt.figure(facecolor = "white")
    
    ax1 = fig.add_subplot(221)
    l1, = ax1.plot(allorders_object.x, allorders_object.psi['g'].real, color="black", label = "real")
    l2, = ax1.plot(allorders_object.x, allorders_object.psi['g'].imag, color="red", label = "imag")
    ax1.set_title(r"Im($\psi$)")

    ax2 = fig.add_subplot(222)
    ax2.plot(allorders_object.x, allorders_object.u['g'].real, color="black")
    ax2.plot(allorders_object.x, allorders_object.u['g'].imag, color="red")
    ax2.set_title("Re($u$)")

    ax3 = fig.add_subplot(223)
    ax3.plot(allorders_object.x, allorders_object.A['g'].real, color="black")
    ax3.plot(allorders_object.x, allorders_object.A['g'].imag, color="red")
    ax3.set_title("Re($A$)")

    ax4 = fig.add_subplot(224)
    ax4.plot(allorders_object.x, allorders_object.B['g'].real, color="black")
    ax4.plot(allorders_object.x, allorders_object.B['g'].imag, color="red")
    ax4.set_title("Im($B$)")

    fig.legend((l1, l2), ("real", "imag"), loc = "upper right")
    
    pylab.savefig(savename)
    
def plotvector(obj, savetitle = "vectorplot"):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    ax1.plot(obj.x, obj.psi['g'].real, color = "black")
    ax1.plot(obj.x, obj.psi['g'].imag, color = "red")
    ax1.set_title(r"$\Psi$")
    
    ax2.plot(obj.x, obj.u['g'].real, color = "black")
    ax2.plot(obj.x, obj.u['g'].imag, color = "red")
    ax2.set_title(r"$u$")
    
    ax3.plot(obj.x, obj.A['g'].real, color = "black")
    ax3.plot(obj.x, obj.A['g'].imag, color = "red")
    ax3.set_title(r"$A$")
    
    ax4.plot(obj.x, obj.B['g'].real, color = "black")
    ax4.plot(obj.x, obj.B['g'].imag, color = "red")
    ax4.set_title(r"$B$")
    
    plt.savefig(savetitle+".png")
    
def plotN2(obj, savetitle = "vectorplot"):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(244)
    
    ax1.plot(obj.x, obj.N20_psi['g'].real, color = "black")
    ax1.plot(obj.x, obj.N20_psi['g'].imag, color = "red")
    ax1.set_title(r"$\Psi$")
    
    ax2.plot(obj.x, obj.N20_u['g'].real, color = "black")
    ax2.plot(obj.x, obj.N20_u['g'].imag, color = "red")
    ax2.set_title(r"$u$")
    
    ax3.plot(obj.x, obj.N20_A['g'].real, color = "black")
    ax3.plot(obj.x, obj.N20_A['g'].imag, color = "red")
    ax3.set_title(r"$A$")
    
    ax4.plot(obj.x, obj.N20_B['g'].real, color = "black")
    ax4.plot(obj.x, obj.N20_B['g'].imag, color = "red")
    ax4.set_title(r"$B$")
    
    ax1 = fig.add_subplot(245)
    ax2 = fig.add_subplot(246)
    ax3 = fig.add_subplot(247)
    ax4 = fig.add_subplot(248)
    
    ax1.plot(obj.x, obj.N22_psi['g'].real, color = "black")
    ax1.plot(obj.x, obj.N22_psi['g'].imag, color = "red")
    ax1.set_title(r"$\Psi$")
    
    ax2.plot(obj.x, obj.N22_u['g'].real, color = "black")
    ax2.plot(obj.x, obj.N22_u['g'].imag, color = "red")
    ax2.set_title(r"$u$")
    
    ax3.plot(obj.x, obj.N22_A['g'].real, color = "black")
    ax3.plot(obj.x, obj.N22_A['g'].imag, color = "red")
    ax3.set_title(r"$A$")
    
    ax4.plot(obj.x, obj.N22_B['g'].real, color = "black")
    ax4.plot(obj.x, obj.N22_B['g'].imag, color = "red")
    ax4.set_title(r"$B$")
    
    plt.savefig(savetitle+".png")
    
def plotN3(obj, savetitle = "vectorplot"):
    
    fig = plt.figure(figsize = (12, 4))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    
    ax1.plot(obj.x, obj.N31_psi['g'].real, color = "black")
    ax1.plot(obj.x, obj.N31_psi['g'].imag, color = "red")
    ax1.set_title(r"$\Psi$")
    
    ax2.plot(obj.x, obj.N31_u['g'].real, color = "black")
    ax2.plot(obj.x, obj.N31_u['g'].imag, color = "red")
    ax2.set_title(r"$u$")
    
    ax3.plot(obj.x, obj.N31_A['g'].real, color = "black")
    ax3.plot(obj.x, obj.N31_A['g'].imag, color = "red")
    ax3.set_title(r"$A$")
    
    ax4.plot(obj.x, obj.N31_B['g'].real, color = "black")
    ax4.plot(obj.x, obj.N31_B['g'].imag, color = "red")
    ax4.set_title(r"$B$")
    
    plt.savefig(savetitle+".png")
    
    
def plotvector2(obj, savetitle = "vectorplot2"):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(341)
    ax2 = fig.add_subplot(342)
    ax3 = fig.add_subplot(343)
    ax4 = fig.add_subplot(344)
    
    ax1.plot(obj.x, obj.psi20['g'].real, color = "black")
    ax1.plot(obj.x, obj.psi20['g'].imag, color = "red")
    ax1.set_title(r"$\Psi$")
    
    ax2.plot(obj.x, obj.u20['g'].real, color = "black")
    ax2.plot(obj.x, obj.u20['g'].imag, color = "red")
    ax2.set_title(r"$u$")
    
    ax3.plot(obj.x, obj.A20['g'].real, color = "black")
    ax3.plot(obj.x, obj.A20['g'].imag, color = "red")
    ax3.set_title(r"$A$")
    
    ax4.plot(obj.x, obj.B20['g'].real, color = "black")
    ax4.plot(obj.x, obj.B20['g'].imag, color = "red")
    ax4.set_title(r"$B$")
    
    ax1 = fig.add_subplot(345)
    ax2 = fig.add_subplot(346)
    ax3 = fig.add_subplot(347)
    ax4 = fig.add_subplot(348)
    
    ax1.plot(obj.x, obj.psi21['g'].real, color = "black")
    ax1.plot(obj.x, obj.psi21['g'].imag, color = "red")
    ax1.set_title(r"$\Psi$")
    
    ax2.plot(obj.x, obj.u21['g'].real, color = "black")
    ax2.plot(obj.x, obj.u21['g'].imag, color = "red")
    ax2.set_title(r"$u$")
    
    ax3.plot(obj.x, obj.A21['g'].real, color = "black")
    ax3.plot(obj.x, obj.A21['g'].imag, color = "red")
    ax3.set_title(r"$A$")
    
    ax4.plot(obj.x, obj.B21['g'].real, color = "black")
    ax4.plot(obj.x, obj.B21['g'].imag, color = "red")
    ax4.set_title(r"$B$")
    
    ax1 = fig.add_subplot(349)
    ax2 = fig.add_subplot(3,4,10)
    ax3 = fig.add_subplot(3,4,11)
    ax4 = fig.add_subplot(3,4,12)
    
    ax1.plot(obj.x, obj.psi22['g'].real, color = "black")
    ax1.plot(obj.x, obj.psi22['g'].imag, color = "red")
    ax1.set_title(r"$\Psi$")
    
    ax2.plot(obj.x, obj.u22['g'].real, color = "black")
    ax2.plot(obj.x, obj.u22['g'].imag, color = "red")
    ax2.set_title(r"$u$")
    
    ax3.plot(obj.x, obj.A22['g'].real, color = "black")
    ax3.plot(obj.x, obj.A22['g'].imag, color = "red")
    ax3.set_title(r"$A$")
    
    ax4.plot(obj.x, obj.B22['g'].real, color = "black")
    ax4.plot(obj.x, obj.B22['g'].imag, color = "red")
    ax4.set_title(r"$B$")
    
    plt.savefig(savetitle+".png")

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
    
    
