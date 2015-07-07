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

if __name__ == "__main__":

    norm = False
    
    Pms = [1E-5, 2E-5, 3E-5, 5E-5, 7E-5]
    satamp = np.zeros(len(Pms))
    
    for i, Pm in enumerate(Pms):
        aa = AmplitudeAlpha(Pm = Pm, norm = norm)
        satamp[i] = aa.sat_amp_coeffs
    
    Pms = np.array(Pms)
    
    plt.loglog(Pms, satamp, '.')
    
    # Fit data
    fitdata = np.polyfit(np.log10(Pms), np.log10(satamp.real), 1)
    xs = np.linspace(np.min(Pms) - 1E-5, np.max(Pms) + 1E-5, 1000) 
    plt.loglog(xs, 10**fitdata[1]*xs**fitdata[0])
    
    pylab.savefig("scrap_amps.png")