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

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

if __name__ == "__main__":

    norm = False
    
    #Pm = 1E-5, Q = 0.747, Rm = 4.88
    #Q = 0.75#0.747
    #Rm = 4.9#88
    
    #Q = [0.747, 0.75, 0.75, 0.75, 0.75] # THESE WORK for Rm = 4.88
    #Pms = [1E-5, 2E-5, 3E-5, 5E-5, 7E-5] # THESE WORK
    
    #Rm = 4.88, Q = 0.75, Pm = 2E-4 works... same Rm and Q blow up for Pm = 1E-4 and 3E-4
    #Rm = 4.88, Q = 0.747, Pm = 1E-4 works... same Rm and Q blow up for Pm = 2E-4 and 3E-4
    #Rm = 4.9, Q = 0.747, Pm = 3E-4 works ... others blow up
    
    #Q = [0.747, 0.747, 0.747]
    #Pms = [5E-4, 7E-4, 1E-3] # these work with Rm = 4.9
    
    #Q = [0.747]
    #Pms = [3E-4]
    #Rm = [4.88]
    
    # These all work
    #Q = [0.747, 0.75, 0.75, 0.75, 0.75, 0.747, 0.75, 0.747, 0.747, 0.747, 0.747]
    #Pms = [1E-5, 2E-5, 3E-5, 5E-5, 7E-5, 1E-4, 2E-4, 3E-4, 5E-4, 7E-4, 1E-3]
    #Rm = [4.88, 4.88, 4.88, 4.88, 4.88, 4.88, 4.88, 4.9, 4.9, 4.9, 4.9]
    
    # Sparser
    Q = [0.747, 0.75, 0.747, 0.747]
    Pms = [1E-5, 5E-5, 3E-4, 1E-3]
    Rm = [4.88, 4.88, 4.9, 4.9]
    
    satamp = np.zeros(len(Pms))
    linear_term = np.zeros(len(Pms))
    coeff_a = np.zeros(len(Pms))
    
    for i, Pm in enumerate(Pms):
        aa = AmplitudeAlpha(Pm = Pm, norm = norm, Rm = Rm[i], Q = Q[i])
        satamp[i] = aa.sat_amp_coeffs
        linear_term[i] = aa.linear_term
        coeff_a[i] = aa.a
    
    Pms = np.array(Pms)
    
    plt.figure()
    plt.loglog(Pms, satamp, '.')
    
    # Fit data
    fitdata = np.polyfit(np.log10(Pms), np.log10(satamp.real), 1)
    xs = np.linspace(np.min(Pms) - 1E-5, np.max(Pms) + 1E-5, 1000) 
    eqnstr = r"$10^{"+str(round(fitdata[1], 3))+"}\cdot Pm^{"+str(round(fitdata[0], 3))+"}$"
    plt.loglog(xs, 10**fitdata[1]*xs**fitdata[0], label = eqnstr)
    plt.legend()
    plt.xlabel(r"$\mathrm{Pm}$")
    
    # Fit data
    plt.figure()
    plt.loglog(Pms, linear_term)
    fitdata = np.polyfit(np.log10(Pms), np.log10(linear_term.real), 1)
    xs = np.linspace(np.min(Pms) - 1E-5, np.max(Pms) + 1E-5, 1000) 
    eqnstr = r"$10^{"+str(round(fitdata[1], 3))+"}\cdot Pm^{"+str(round(fitdata[0], 3))+"}$"
    plt.loglog(xs, 10**fitdata[1]*xs**fitdata[0], label = "linear term: "+eqnstr)

    plt.xlabel(r"$\mathrm{Pm}$")
    
    plt.loglog(Pms, coeff_a)
    fitdata = np.polyfit(np.log10(Pms), np.log10(coeff_a.real), 1)
    xs = np.linspace(np.min(Pms) - 1E-5, np.max(Pms) + 1E-5, 1000) 
    eqnstr = r"$10^{"+str(round(fitdata[1], 3))+"}\cdot Pm^{"+str(round(fitdata[0], 3))+"}$"
    plt.loglog(xs, 10**fitdata[1]*xs**fitdata[0], label = "a coeff: "+eqnstr)
 
    plt.legend()

    
    pylab.savefig("scrap2_amps_Rm"+str(Rm)+"_Pm_"+str(Pms[0])+"_to_"+str(Pms[-1])+".png")
    
    