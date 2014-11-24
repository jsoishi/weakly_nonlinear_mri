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
from allorders import *

import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
matplotlib.rcParams.update({'figure.autolayout': True})

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

gridnum = 512
x_basis = Chebyshev(gridnum)
domain = Domain([x_basis], grid_dtype=np.complex128)

#Pm = 1E-2: Q = 0.757, Rm = 4.93
"""
Pm = 1.0E-2
Q = 0.757
Rm = 4.93
q = 1.5
beta = 25.0
"""

#Pm = 1E-3: Q = 0.75, Rm = 4.8775
"""
Pm = 1.0E-3
Q = 0.75
Rm = 4.8775
q = 1.5
beta = 25.0
"""

#Pm = 1E-4: Q = 0.747, Rm = 4.88
"""
Pm = 1.0E-4
Q = 0.747
Rm = 4.88
q = 1.5
beta = 25.0
"""

#Pm = 1E-5, Q = 0.747, Rm = 4.88
"""
Pm = 1.0E-5
Q = 0.747
Rm = 4.88
q = 1.5
beta = 25.0
"""

# Pm = 1E-6: Q = 0.75, Rm = 4.88, beta = 25
"""
Pm = 1.0E-6
Q = 0.75
Rm = 4.88
q = 1.5
beta = 25.0
"""

#Pm = 1E-7: Q = 0.745, Rm = 4.90


Pm = 1.0E-7
Q = 0.745
Rm = 4.90
q = 1.5
beta = 25.0


#Pm = 1E-8: Q = 0.815, Rm = 4.75
"""
Pm = 1.0E-8
Q = 0.815
Rm = 4.75
q = 1.5
beta = 25.0
"""

pc = PlotContours(Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, run = True, norm = True)
pc.plot_streams()
pc.plot_Bfield()

coeffs = {}
coeffs["Q"] = Q
coeffs["Rm"] = Rm
coeffs["Pm"] = Pm
coeffs["beta"] = beta
coeffs["q"] = q
coeffs["a"] = pc.saa.alpha_amp.a
coeffs["c"] = pc.saa.alpha_amp.c
coeffs["ctwiddle"] = pc.saa.alpha_amp.c_twiddle
coeffs["b"] = pc.saa.alpha_amp.b
coeffs["h"] = pc.saa.alpha_amp.h
coeffs["g"] = pc.saa.alpha_amp.g
coeffs["t_array"] = pc.saa.t_array
coeffs["alpha_array"] = pc.saa.alpha_array

outname = "gridnum_"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)
plot_uy_firstorder(pc, oplot = True, labels = False, outname=outname)
plot_By_firstorder(pc, oplot = True, labels = False, outname=outname)
plot_uy_secondorder(pc, oplot = True, labels = False, outname=outname)
plot_By_secondorder(pc, oplot = True, labels = False, outname=outname)
plot_uy(pc, oplot = True, labels = False, outname=outname)
plot_By(pc, oplot = True, labels = False, outname=outname)
plotN2(pc.saa.alpha_amp.o2, outname=outname)
plotN3(pc.saa.alpha_amp.n3, outname=outname)
plotOE(pc.saa.alpha_amp, outname=outname)

print(coeffs)

pickle.dump(coeffs, open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+".p", "wb"))