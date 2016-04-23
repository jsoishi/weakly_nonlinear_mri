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

import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
matplotlib.rcParams.update({'figure.autolayout': True})

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import allorders

pc = allorders.PlotContours(Q = 0.75, Rm = 4.9, Pm = 0.0001, q = 1.5, beta = 25.0)

plot_uy_firstorder(pc, outname="Pm1E-4")
plot_uy_secondorder(pc, outname="Pm1E-4")
plot_uy(pc, outname="Pm1E-4")

plot_By_firstorder(pc, outname="Pm1E-4")
plot_By_secondorder(pc, outname="Pm1E-4")
plot_By(pc, outname="Pm1E-4")