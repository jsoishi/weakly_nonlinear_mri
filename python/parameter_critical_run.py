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
    print(Pm)
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
    
def get_critical_parameters_by_Pm(Pm, allgoodeigs = True):
    
    """
    Return critical Rm and vertical wavenumber Q for given input Pm
    """
    
    path = "/Users/susanclark/weakly_nonlinear_mri/python/multirun/"
    
    if Pm == 0.01:
        datafile = "gridnum_128_Pm_0.01_Q_0.74_dQ_0.001_Rm_4.91_dRm_0.001_allgoodeigs.p"
    elif (Pm == 0.001) or (Pm == 0.005):
        datafile = "gridnum_128_Pm_"+str(Pm)+"_Q_0.74_dQ_0.001_Rm_4.87_dRm_0.001.p"
        allgoodeigs = False
    else:     
        datafile = "gridnum_128_Pm_"+str(Pm)+"_Q_0.74_dQ_0.001_Rm_4.87_dRm_0.001_allgoodeigs.p"
   
    print(datafile)
    pspace_grid = pickle.load(open(path+datafile, "rb"))
    
    search_Rms, search_Qs, max_evals = get_parameter_space_data(pspace_grid, allgoodeigs = allgoodeigs)
    
    plot_paramspace_run(search_Rms, search_Qs, max_evals)
    
    # Modes with Re{eval} > 0 are growing
    growing_modes = np.zeros(len(max_evals), np.int)
    growing_modes[max_evals.real > 0] = 1
    
    if np.sum(growing_modes) < 1:
        raise ValueError(" There are no growing modes in this run! ") 
    else: print(" There are %d growing modes! " %np.sum(growing_modes))
    
    # Critical eigenvalues are the smallest Rm with a growing mode, and its associated vertical wavenumber Q
    #marginal_indx = np.nanargmin(search_Rms[np.where(growing_modes == 1)]) # this is wrong

    jj = [(i, a) for i, a in enumerate(search_Rms) if growing_modes[i] == 1]
    jj = np.asarray(jj)
    print(jj.shape, len(jj))
    jj_i = jj[:, 0]
    jj_Rm = jj[:, 1]
    print(np.nanargmin(jj_Rm))
    print(np.nanmin(jj_Rm))
    print(jj_i[np.nanargmin(jj_Rm)])
    print(search_Rms[jj_i[np.nanargmin(jj_Rm)]])
    
    print(jj_Rm[jj_Rm == np.nanmin(jj_Rm)])
    print(search_Qs[search_Rms == np.nanmin(jj_Rm)])
    
    marginal_indx = jj_i[np.nanargmin(jj_Rm)]
    
    print(marginal_indx.dtype)
    marginal_Rm = search_Rms[marginal_indx]
    marginal_Q = search_Qs[marginal_indx]
    
    if marginal_Rm == np.nanmax(search_Rms):
        print("Caution!! marginal_Rm == nanmax(search_Rms)")
    if marginal_Rm == np.nanmin(search_Rms):
        print("Caution!! marginal_Rm == nanmin(search_Rms)")
    if marginal_Q == np.nanmax(search_Qs):
        print("Caution!! marginal_Q == nanmax(search_Qs)")
    if marginal_Q == np.nanmin(search_Qs):
        print("Caution!! marginal_Q == nanmin(search_Qs)")
    
    print("For Pm of", Pm, "critical Rm is", marginal_Rm, " critical Q is", marginal_Q)
    
    return marginal_Rm, marginal_Q

def plot_paramspace_run(Rm, Q, evals):

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    cb = ax1.scatter(Q, Rm, c=np.real(evals), marker="s", s=40, cmap = "RdBu")#, vmin=-1E-7, vmax=1E-7)
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

    Pms = [1.0E-4, 5.0E-4, 1.0E-3, 5.0E-3, 1.0E-2]
    #Pms = [1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3]
    coeffs = np.zeros(len(Pms), np.complex)
    ivpsa = np.zeros(len(Pms), np.complex)
    covera = np.zeros(len(Pms), np.complex)
    
    objs = {}
    
    for i, Pm in enumerate(Pms):
        marginal_Rm, marginal_Q = get_critical_parameters_by_Pm(Pm)
        """
        amplitude_obj = AmplitudeAlpha(Pm = Pm, Rm = marginal_Rm, Q = marginal_Q, norm = True)
        coeffs[i] = amplitude_obj.sat_amp_coeffs
        ivpsa[i] = amplitude_obj.saturation_amplitude
        
        # c/a value plotted in Umurhan+
        covera[i] = amplitude_obj.c/amplitude_obj.a
        
        objs[Pm] = amplitude_obj
        """
    
    # Saturation amplitude vs Pm plots 
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(122)
    
    ax1.plot(Pms, coeffs, 'o', color = "purple")
    ax1.semilogx()
    ax1.semilogy()
    ax1.set_xlabel("Pm")
    ax1.set_title("from coefficients")
    pylab.savefig("scrap3.png", dpi = 100)
    
    """
    ax2.plot(Pms, ivpsa, 'o', color = "orange")
    ax2.semilogx()
    ax2.semilogy()
    ax2.set_xlabel("Pm")
    ax2.set_title("from IVP solve")
    
    plt.suptitle("saturation amplitude", size=20)
    plt.show()
    pylab.savefig("scrap3.png", dpi = 100)
    
    # Amplitude as a function of time plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    colors = _get_colors(len(Pms))
    for i, Pm_ in enumerate(Pms):
        ax1.plot(objs[Pm_].t_array, objs[Pm_].alpha_array[:, 0], color = colors[i])
        
    pylab.savefig("amplitude_plot.png")
    """
    
    #c over a plot
    plt.figure()
    plt.plot(Pms, covera, 'o')
    plt.semilogx()
    plt.semilogy()
    plt.xlabel("Pm")
    plt.ylabel("c/a")
    plt.show()
    pylab.savefig("covera.png", dpi = 100)

def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors
    