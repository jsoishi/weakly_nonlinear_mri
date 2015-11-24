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
import decimal
from scipy import polyfit
from scipy.optimize import curve_fit

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
    
def plotvector(obj, Pm, savetitle = "vectorplot", setlims = True, psimax = 1, psimin = -1, umax = 1, umin = -1, Amax = 1, Amin = -1, Bmax = 1, Bmin = -1):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    ax1.plot(obj.x, obj.psi['g'].real, color = "black")
    ax1.plot(obj.x, obj.psi['g'].imag, color = "red")
    if setlims == True:
        ax1.set_ylim(psimin, psimax)
    ax1.set_title(r"$\Psi$")
    
    ax2.plot(obj.x, obj.u['g'].real, color = "black")
    ax2.plot(obj.x, obj.u['g'].imag, color = "red")
    if setlims == True:
        ax2.set_ylim(umin, umax)
    ax2.set_title(r"$u$")
    
    ax3.plot(obj.x, obj.A['g'].real, color = "black")
    ax3.plot(obj.x, obj.A['g'].imag, color = "red")
    if setlims == True:
        ax3.set_ylim(Amin, Amax)
    ax3.set_title(r"$A$")
    
    ax4.plot(obj.x, obj.B['g'].real, color = "black")
    ax4.plot(obj.x, obj.B['g'].imag, color = "red")
    if setlims == True:
        ax4.set_ylim(Bmin, Bmax)
    ax4.set_title(r"$B$")
    
    plt.suptitle(r"$\mathrm{Pm} = $"+str(Pm), size = 20)
    plt.tight_layout()
    
    plt.savefig(savetitle+".png")
    
def plotN2(obj, Pm, savetitle = "N2vectorplot", psimax20 = 1, psimin20 = -1, umax20 = 1, umin20 = -1, Amax20 = 1, Amin20 = -1, Bmax20 = 1, Bmin20 = -1, psimax22 = 1, psimin22 = -1, umax22 = 1, umin22 = -1, Amax22 = 1, Amin22 = -1, Bmax22 = 1, Bmin22 = -1):
    
    fig = plt.figure(figsize = (10, 6))
    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(244)
    
    ax1.plot(obj.x, obj.N20_psi['g'].real, color = "black")
    ax1.plot(obj.x, obj.N20_psi['g'].imag, color = "red")
    ax1.set_ylim(psimin20, psimax20)
    ax1.set_title(r"$\Psi_{20}$")
    
    ax2.plot(obj.x, obj.N20_u['g'].real, color = "black")
    ax2.plot(obj.x, obj.N20_u['g'].imag, color = "red")
    ax2.set_ylim(umin20, umax20)
    ax2.set_title(r"$u_{20}$")
    
    ax3.plot(obj.x, obj.N20_A['g'].real, color = "black")
    ax3.plot(obj.x, obj.N20_A['g'].imag, color = "red")
    ax3.set_ylim(Amin20, Amax20)
    ax3.set_title(r"$A_{20}$")
    
    ax4.plot(obj.x, obj.N20_B['g'].real, color = "black")
    ax4.plot(obj.x, obj.N20_B['g'].imag, color = "red")
    ax4.set_ylim(Bmin20, Bmax20)
    ax4.set_title(r"$B_{20}$")
    
    ax1 = fig.add_subplot(245)
    ax2 = fig.add_subplot(246)
    ax3 = fig.add_subplot(247)
    ax4 = fig.add_subplot(248)
    
    ax1.plot(obj.x, obj.N22_psi['g'].real, color = "black")
    ax1.plot(obj.x, obj.N22_psi['g'].imag, color = "red")
    ax1.set_ylim(psimin22, psimax22)
    ax1.set_title(r"$\Psi_{22}$")
    
    ax2.plot(obj.x, obj.N22_u['g'].real, color = "black")
    ax2.plot(obj.x, obj.N22_u['g'].imag, color = "red")
    ax2.set_ylim(umin22, umax22)
    ax2.set_title(r"$u_{22}$")
    
    ax3.plot(obj.x, obj.N22_A['g'].real, color = "black")
    ax3.plot(obj.x, obj.N22_A['g'].imag, color = "red")
    ax3.set_ylim(Amin22, Amax22)
    ax3.set_title(r"$A_{22}$")
    
    ax4.plot(obj.x, obj.N22_B['g'].real, color = "black")
    ax4.plot(obj.x, obj.N22_B['g'].imag, color = "red")
    ax4.set_ylim(Bmin22, Bmax22)
    ax4.set_title(r"$B_{22}$")
    
    plt.suptitle(r"$\mathrm{Pm} = $"+str(Pm), size = 20)
    plt.tight_layout()
    plt.savefig(savetitle+".png")
    
def plotN3(obj, Pm, savetitle = "N3vectorplot", psimax31 = 1, psimin31 = -1, umax31 = 1, umin31 = -1, Amax31 = 1, Amin31 = -1, Bmax31 = 1, Bmin31 = -1):
    
    fig = plt.figure(figsize = (12, 4))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    
    ax1.plot(obj.x, obj.N31_psi['g'].real, color = "black")
    ax1.plot(obj.x, obj.N31_psi['g'].imag, color = "red")
    ax1.set_ylim(psimin31, psimax31)
    ax1.set_title(r"$\Psi$")
    
    ax2.plot(obj.x, obj.N31_u['g'].real, color = "black")
    ax2.plot(obj.x, obj.N31_u['g'].imag, color = "red")
    ax2.set_ylim(umin31, umax31)
    ax2.set_title(r"$u$")
    
    ax3.plot(obj.x, obj.N31_A['g'].real, color = "black")
    ax3.plot(obj.x, obj.N31_A['g'].imag, color = "red")
    ax3.set_ylim(Amin31, Amax31)
    ax3.set_title(r"$A$")
    
    ax4.plot(obj.x, obj.N31_B['g'].real, color = "black")
    ax4.plot(obj.x, obj.N31_B['g'].imag, color = "red")
    ax4.set_ylim(Bmin31, Bmax31)
    ax4.set_title(r"$B$")
    
    plt.suptitle(r"$\mathrm{Pm} = $"+str(Pm), size = 20)
    plt.tight_layout()
    plt.savefig(savetitle+".png")

def plotV21_RHS(obj, Pm, savetitle = "V21_RHSvectorplot", psimax = 1, psimin = -1, umax = 1, umin = -1, Amax = 1, Amin = -1, Bmax = 1, Bmin = -1):
    
    fig = plt.figure(figsize = (12, 4))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    
    ax1.plot(obj.x, obj.term2_psi['g'].real, color = "black")
    ax1.plot(obj.x, obj.term2_psi['g'].imag, color = "red")
    ax1.set_ylim(psimin, psimax)
    ax1.set_title(r"$\Psi$")
    
    ax2.plot(obj.x, obj.term2_u['g'].real, color = "black")
    ax2.plot(obj.x, obj.term2_u['g'].imag, color = "red")
    ax2.set_ylim(umin, umax)
    ax2.set_title(r"$u$")
    
    ax3.plot(obj.x, obj.term2_A['g'].real, color = "black")
    ax3.plot(obj.x, obj.term2_A['g'].imag, color = "red")
    ax3.set_ylim(Amin, Amax)
    ax3.set_title(r"$A$")
    
    ax4.plot(obj.x, obj.term2_B['g'].real, color = "black")
    ax4.plot(obj.x, obj.term2_B['g'].imag, color = "red")
    ax4.set_ylim(Bmin, Bmax)
    ax4.set_title(r"$B$")
    
    decPm = '%.2E' % decimal.Decimal(Pm)
    plt.suptitle(r"$\mathrm{Pm} = $"+decPm, size = 20)
    plt.tight_layout()
    plt.savefig(savetitle+".png")    
    
def plotvector2(obj, Pm, savetitle = "vectorplot2", psimax20 = 1, psimin20 = -1, umax20 = 1, umin20 = -1, Amax20 = 1, Amin20 = -1, Bmax20 = 1, Bmin20 = -1, psimax21 = 1, psimin21 = -1, umax21 = 1, umin21 = -1, Amax21 = 1, Amin21 = -1, Bmax21 = 1, Bmin21 = -1, psimax22 = 1, psimin22 = -1, umax22 = 1, umin22 = -1, Amax22 = 1, Amin22 = -1, Bmax22 = 1, Bmin22 = -1):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(341)
    ax2 = fig.add_subplot(342)
    ax3 = fig.add_subplot(343)
    ax4 = fig.add_subplot(344)
    
    ax1.plot(obj.x, obj.psi20['g'].real, color = "black")
    ax1.plot(obj.x, obj.psi20['g'].imag, color = "red")
    ax1.set_ylim(psimin20, psimax20)
    ax1.set_title(r"$\Psi_{20}$")
    
    ax2.plot(obj.x, obj.u20['g'].real, color = "black")
    ax2.plot(obj.x, obj.u20['g'].imag, color = "red")
    ax2.set_ylim(umin20, umax20)
    ax2.set_title(r"$u_{20}$")
    
    ax3.plot(obj.x, obj.A20['g'].real, color = "black")
    ax3.plot(obj.x, obj.A20['g'].imag, color = "red")
    ax3.set_ylim(Amin20, Amax20)
    ax3.set_title(r"$A_{20}$")
    
    ax4.plot(obj.x, obj.B20['g'].real, color = "black")
    ax4.plot(obj.x, obj.B20['g'].imag, color = "red")
    ax4.set_ylim(Bmin20, Bmax20)
    ax4.set_title(r"$B_{20}$")
    
    ax1 = fig.add_subplot(345)
    ax2 = fig.add_subplot(346)
    ax3 = fig.add_subplot(347)
    ax4 = fig.add_subplot(348)
    
    ax1.plot(obj.x, obj.psi21['g'].real, color = "black")
    ax1.plot(obj.x, obj.psi21['g'].imag, color = "red")
    ax1.set_ylim(psimin21, psimax21)
    ax1.set_title(r"$\Psi_{21}$")
    
    ax2.plot(obj.x, obj.u21['g'].real, color = "black")
    ax2.plot(obj.x, obj.u21['g'].imag, color = "red")
    ax2.set_ylim(umin21, umax21)
    ax2.set_title(r"$u_{21}$")
    
    ax3.plot(obj.x, obj.A21['g'].real, color = "black")
    ax3.plot(obj.x, obj.A21['g'].imag, color = "red")
    ax3.set_ylim(Amin21, Amax21)
    ax3.set_title(r"$A_{21}$")
    
    ax4.plot(obj.x, obj.B21['g'].real, color = "black")
    ax4.plot(obj.x, obj.B21['g'].imag, color = "red")
    ax4.set_ylim(Bmin21, Bmax21)
    ax4.set_title(r"$B_{21}$")
    
    ax1 = fig.add_subplot(349)
    ax2 = fig.add_subplot(3,4,10)
    ax3 = fig.add_subplot(3,4,11)
    ax4 = fig.add_subplot(3,4,12)
    
    ax1.plot(obj.x, obj.psi22['g'].real, color = "black")
    ax1.plot(obj.x, obj.psi22['g'].imag, color = "red")
    ax1.set_ylim(psimin22, psimax22)
    ax1.set_title(r"$\Psi_{22}$")
    
    ax2.plot(obj.x, obj.u22['g'].real, color = "black")
    ax2.plot(obj.x, obj.u22['g'].imag, color = "red")
    ax2.set_ylim(umin22, umax22)
    ax2.set_title(r"$u_{22}$")
    
    ax3.plot(obj.x, obj.A22['g'].real, color = "black")
    ax3.plot(obj.x, obj.A22['g'].imag, color = "red")
    ax3.set_ylim(Amin22, Amax22)
    ax3.set_title(r"$A_{22}$")
    
    ax4.plot(obj.x, obj.B22['g'].real, color = "black")
    ax4.plot(obj.x, obj.B22['g'].imag, color = "red")
    ax4.set_ylim(Bmin22, Bmax22)
    ax4.set_title(r"$B_{22}$")
    
    decPm = '%.2E' % decimal.Decimal(Pm)
    plt.suptitle(r"$\mathrm{Pm} = $"+decPm, size = 20)
    plt.tight_layout()
    plt.savefig(savetitle+".png")
    
def plot_all_O2s(objs, Pms):
    
    psimax20, psimin20 = get_minmax_o20_psi_allPms(objs, Pms)
    umax20, umin20 = get_minmax_o20_u_allPms(objs, Pms)
    Amax20, Amin20 = get_minmax_o20_A_allPms(objs, Pms)
    Bmax20, Bmin20 = get_minmax_o20_B_allPms(objs, Pms)
    
    psimax21, psimin21 = get_minmax_o21_psi_allPms(objs, Pms)
    umax21, umin21 = get_minmax_o21_u_allPms(objs, Pms)
    Amax21, Amin21 = get_minmax_o21_A_allPms(objs, Pms)
    Bmax21, Bmin21 = get_minmax_o21_B_allPms(objs, Pms)
    
    psimax22, psimin22 = get_minmax_o22_psi_allPms(objs, Pms)
    umax22, umin22 = get_minmax_o22_u_allPms(objs, Pms)
    Amax22, Amin22 = get_minmax_o22_A_allPms(objs, Pms)
    Bmax22, Bmin22 = get_minmax_o22_B_allPms(objs, Pms)

    for Pm in Pms:
        plotvector2(objs[Pm].o2, Pm, savetitle = "vectorplot2_norm_indiv2_"+str(Pm), psimax20 = psimax20, psimin20 = psimin20, umax20 = umax20, umin20 = umin20, Amax20 = Amax20, Amin20 = Amin20, Bmax20 = Bmax20, Bmin20 = Bmin20, psimax21 = psimax21, psimin21 = psimin21, umax21 = umax21, umin21 = umin21, Amax21 = Amax21, Amin21 = Amin21, Bmax21 = Bmax21, Bmin21 = Bmin21, psimax22 = psimax22, psimin22 = psimin22, umax22 = umax22, umin22 = umin22, Amax22 = Amax22, Amin22 = Amin22, Bmax22 = Bmax22, Bmin22 = Bmin22)
        
def plot_all_N3s(objs, Pms):

    psimax31, psimin31 = get_minmax_N31_psi_allPms(objs, Pms)
    umax31, umin31 = get_minmax_N31_u_allPms(objs, Pms)
    Amax31, Amin31 = get_minmax_N31_A_allPms(objs, Pms)
    Bmax31, Bmin31 = get_minmax_N31_B_allPms(objs, Pms)

    for Pm in Pms:
        plotN3(objs[Pm].n3, Pm, savetitle = "N3_"+str(Pm), psimax31 = psimax31, psimin31 = psimin31, umax31 = umax31, umin31 = umin31, Amax31 = Amax31, Amin31 = Amin31, Bmax31 = Bmax31, Bmin31 = Bmin31)
        
def plot_all_ahs(objs, Pms):
    psimax, psimin = get_minmax_ah_psi_allPms(objs, Pms)
    umax, umin = get_minmax_ah_u_allPms(objs, Pms)
    Amax, Amin = get_minmax_ah_A_allPms(objs, Pms)
    Bmax, Bmin = get_minmax_ah_B_allPms(objs, Pms)

    for Pm in Pms:
        plotvector(objs[Pm].ah, Pm, savetitle = "adjoint_homogenous_ipnorms_"+str(Pm), psimax = psimax, psimin = psimin, umax = umax, umin = umin, Amax = Amax, Amin = Amin, Bmax = Bmax, Bmin = Bmin)
    
def plot_all_o1s(objs, Pms):
    psimax, psimin = get_minmax_psi_allPms(objs, Pms)
    umax, umin = get_minmax_u_allPms(objs, Pms)
    Amax, Amin = get_minmax_A_allPms(objs, Pms)
    Bmax, Bmin = get_minmax_B_allPms(objs, Pms)
    
    print(psimax, psimin, umax, umin, Amax, Amin, Bmax, Bmin)
    
    for Pm in Pms:
        plotvector(objs[Pm].o1, Pm, savetitle = "Order_E_ipnorms"+str(Pm), psimax = psimax, psimin = psimin, umax = umax, umin = umin, Amax = Amax, Amin = Amin, Bmax = Bmax, Bmin = Bmin)

def plot_all_v21_rhss(objs, Pms):
    psimax, psimin = get_minmax_psi21_rhs_allPms(objs, Pms)
    umax, umin = get_minmax_u21_rhs_allPms(objs, Pms)
    Amax, Amin = get_minmax_A21_rhs_allPms(objs, Pms)
    Bmax, Bmin = get_minmax_B21_rhs_allPms(objs, Pms)
    
    print(psimax, psimin, umax, umin, Amax, Amin, Bmax, Bmin)
    
    for Pm in Pms:
        plotV21_RHS(objs[Pm].o2, Pm, savetitle = "V21_RHS_"+str(Pm), psimax = psimax, psimin = psimin, umax = umax, umin = umin, Amax = Amax, Amin = Amin, Bmax = Bmax, Bmin = Bmin)


def plot_all_N2s(objs, Pms):
    
    psimax20, psimin20 = get_minmax_N20_psi_allPms(objs, Pms)
    umax20, umin20 = get_minmax_N20_u_allPms(objs, Pms)
    Amax20, Amin20 = get_minmax_N20_A_allPms(objs, Pms)
    Bmax20, Bmin20 = get_minmax_N20_B_allPms(objs, Pms)
    
    psimax22, psimin22 = get_minmax_N22_psi_allPms(objs, Pms)
    umax22, umin22 = get_minmax_N22_u_allPms(objs, Pms)
    Amax22, Amin22 = get_minmax_N22_A_allPms(objs, Pms)
    Bmax22, Bmin22 = get_minmax_N22_B_allPms(objs, Pms)

    for Pm in Pms:
        plotN2(objs[Pm].n2, Pm, savetitle = "N2_"+str(Pm), psimax20 = psimax20, psimin20 = psimin20, umax20 = umax20, umin20 = umin20, Amax20 = Amax20, Amin20 = Amin20, Bmax20 = Bmax20, Bmin20 = Bmin20, psimax22 = psimax22, psimin22 = psimin22, umax22 = umax22, umin22 = umin22, Amax22 = Amax22, Amin22 = Amin22, Bmax22 = Bmax22, Bmin22 = Bmin22)
    
def plotcoeffs(objs, Pms):
    
    all_as = np.zeros(len(Pms), np.complex_)
    all_bs = np.zeros(len(Pms), np.complex_)
    all_cs = np.zeros(len(Pms), np.complex_)
    all_hs = np.zeros(len(Pms), np.complex_)
    
    for i, Pm in enumerate(Pms):
        all_as[i] = objs[Pm].a
        all_bs[i] = objs[Pm].b
        all_cs[i] = objs[Pm].c
        all_hs[i] = objs[Pm].h
    
    all_lambda = all_bs/all_as
    all_D = all_hs/all_as
    all_alpha = all_cs/all_as
    
    plt.figure()
    plt.plot(Pms, all_alpha, '.')
    plt.plot(Pms, all_lambda, '.')
    plt.plot(Pms, all_D, '.')
    plt.legend(["alpha", "lambda", "D"])
    pylab.savefig("scrap.png")
    
    return all_lambda, all_D, all_alpha, all_as, all_bs, all_cs, all_hs
    
def get_minmax_psi_allPms(objs, Pms):

    psi_maxreal = [np.max(objs[Pm].o1.psi['g'].real) for Pm in Pms]
    psi_maximag = [np.max(objs[Pm].o1.psi['g'].imag) for Pm in Pms]
    psi_minreal = [np.min(objs[Pm].o1.psi['g'].real) for Pm in Pms]
    psi_minimag = [np.min(objs[Pm].o1.psi['g'].imag) for Pm in Pms]
    
    psimax = max(np.nanmax(psi_maxreal), np.nanmax(psi_maximag))
    psimin = min(np.nanmin(psi_minreal), np.nanmin(psi_minimag))
    
    return psimax, psimin
    
def get_minmax_u_allPms(objs, Pms):

    u_maxreal = [np.max(objs[Pm].o1.u['g'].real) for Pm in Pms]
    u_maximag = [np.max(objs[Pm].o1.u['g'].imag) for Pm in Pms]
    u_minreal = [np.min(objs[Pm].o1.u['g'].real) for Pm in Pms]
    u_minimag = [np.min(objs[Pm].o1.u['g'].imag) for Pm in Pms]
    
    umax = max(np.nanmax(u_maxreal), np.nanmax(u_maximag))
    umin = min(np.nanmin(u_minreal), np.nanmin(u_minimag))
    
    return umax, umin
    
def get_minmax_A_allPms(objs, Pms):

    A_maxreal = [np.max(objs[Pm].o1.A['g'].real) for Pm in Pms]
    A_maximag = [np.max(objs[Pm].o1.A['g'].imag) for Pm in Pms]
    A_minreal = [np.min(objs[Pm].o1.A['g'].real) for Pm in Pms]
    A_minimag = [np.min(objs[Pm].o1.A['g'].imag) for Pm in Pms]
    
    Amax = max(np.nanmax(A_maxreal), np.nanmax(A_maximag))
    Amin = min(np.nanmin(A_minreal), np.nanmin(A_minimag))
    
    return Amax, Amin
    
def get_minmax_B_allPms(objs, Pms):

    B_maxreal = [np.max(objs[Pm].o1.B['g'].real) for Pm in Pms]
    B_maximag = [np.max(objs[Pm].o1.B['g'].imag) for Pm in Pms]
    B_minreal = [np.min(objs[Pm].o1.B['g'].real) for Pm in Pms]
    B_minimag = [np.min(objs[Pm].o1.B['g'].imag) for Pm in Pms]
    
    Bmax = max(np.nanmax(B_maxreal), np.nanmax(B_maximag))
    Bmin = min(np.nanmin(B_minreal), np.nanmin(B_minimag))
    
    return Bmax, Bmin
    
def get_minmax_psi21_rhs_allPms(objs, Pms):

    psi_maxreal = [np.max(objs[Pm].o2.term2_psi['g'].real) for Pm in Pms]
    psi_maximag = [np.max(objs[Pm].o2.term2_psi['g'].imag) for Pm in Pms]
    psi_minreal = [np.min(objs[Pm].o2.term2_psi['g'].real) for Pm in Pms]
    psi_minimag = [np.min(objs[Pm].o2.term2_psi['g'].imag) for Pm in Pms]
    
    psimax = max(np.nanmax(psi_maxreal), np.nanmax(psi_maximag))
    psimin = min(np.nanmin(psi_minreal), np.nanmin(psi_minimag))
    
    return psimax, psimin
    
def get_minmax_u21_rhs_allPms(objs, Pms):

    u_maxreal = [np.max(objs[Pm].o2.term2_u['g'].real) for Pm in Pms]
    u_maximag = [np.max(objs[Pm].o2.term2_u['g'].imag) for Pm in Pms]
    u_minreal = [np.min(objs[Pm].o2.term2_u['g'].real) for Pm in Pms]
    u_minimag = [np.min(objs[Pm].o2.term2_u['g'].imag) for Pm in Pms]
    
    umax = max(np.nanmax(u_maxreal), np.nanmax(u_maximag))
    umin = min(np.nanmin(u_minreal), np.nanmin(u_minimag))
    
    return umax, umin
    
def get_minmax_A21_rhs_allPms(objs, Pms):

    A_maxreal = [np.max(objs[Pm].o2.term2_A['g'].real) for Pm in Pms]
    A_maximag = [np.max(objs[Pm].o2.term2_A['g'].imag) for Pm in Pms]
    A_minreal = [np.min(objs[Pm].o2.term2_A['g'].real) for Pm in Pms]
    A_minimag = [np.min(objs[Pm].o2.term2_A['g'].imag) for Pm in Pms]
    
    Amax = max(np.nanmax(A_maxreal), np.nanmax(A_maximag))
    Amin = min(np.nanmin(A_minreal), np.nanmin(A_minimag))
    
    return Amax, Amin
    
def get_minmax_B21_rhs_allPms(objs, Pms):

    B_maxreal = [np.max(objs[Pm].o2.term2_B['g'].real) for Pm in Pms]
    B_maximag = [np.max(objs[Pm].o2.term2_B['g'].imag) for Pm in Pms]
    B_minreal = [np.min(objs[Pm].o2.term2_B['g'].real) for Pm in Pms]
    B_minimag = [np.min(objs[Pm].o2.term2_B['g'].imag) for Pm in Pms]
    
    Bmax = max(np.nanmax(B_maxreal), np.nanmax(B_maximag))
    Bmin = min(np.nanmin(B_minreal), np.nanmin(B_minimag))
    
    return Bmax, Bmin
    
def get_minmax_ah_psi_allPms(objs, Pms):

    psi_maxreal = [np.max(objs[Pm].ah.psi['g'].real) for Pm in Pms]
    psi_maximag = [np.max(objs[Pm].ah.psi['g'].imag) for Pm in Pms]
    psi_minreal = [np.min(objs[Pm].ah.psi['g'].real) for Pm in Pms]
    psi_minimag = [np.min(objs[Pm].ah.psi['g'].imag) for Pm in Pms]
    
    psimax = max(np.nanmax(psi_maxreal), np.nanmax(psi_maximag))
    psimin = min(np.nanmin(psi_minreal), np.nanmin(psi_minimag))
    
    return psimax, psimin
    
def get_minmax_ah_u_allPms(objs, Pms):

    u_maxreal = [np.max(objs[Pm].ah.u['g'].real) for Pm in Pms]
    u_maximag = [np.max(objs[Pm].ah.u['g'].imag) for Pm in Pms]
    u_minreal = [np.min(objs[Pm].ah.u['g'].real) for Pm in Pms]
    u_minimag = [np.min(objs[Pm].ah.u['g'].imag) for Pm in Pms]
    
    umax = max(np.nanmax(u_maxreal), np.nanmax(u_maximag))
    umin = min(np.nanmin(u_minreal), np.nanmin(u_minimag))
    
    return umax, umin
    
def get_minmax_ah_A_allPms(objs, Pms):

    A_maxreal = [np.max(objs[Pm].ah.A['g'].real) for Pm in Pms]
    A_maximag = [np.max(objs[Pm].ah.A['g'].imag) for Pm in Pms]
    A_minreal = [np.min(objs[Pm].ah.A['g'].real) for Pm in Pms]
    A_minimag = [np.min(objs[Pm].ah.A['g'].imag) for Pm in Pms]
    
    Amax = max(np.nanmax(A_maxreal), np.nanmax(A_maximag))
    Amin = min(np.nanmin(A_minreal), np.nanmin(A_minimag))
    
    return Amax, Amin
    
def get_minmax_ah_B_allPms(objs, Pms):

    B_maxreal = [np.max(objs[Pm].ah.B['g'].real) for Pm in Pms]
    B_maximag = [np.max(objs[Pm].ah.B['g'].imag) for Pm in Pms]
    B_minreal = [np.min(objs[Pm].ah.B['g'].real) for Pm in Pms]
    B_minimag = [np.min(objs[Pm].ah.B['g'].imag) for Pm in Pms]
    
    Bmax = max(np.nanmax(B_maxreal), np.nanmax(B_maximag))
    Bmin = min(np.nanmin(B_minreal), np.nanmin(B_minimag))
    
    return Bmax, Bmin
    
def get_minmax_N20_psi_allPms(objs, Pms):

    psi_maxreal = [np.max(objs[Pm].n2.N20_psi['g'].real) for Pm in Pms]
    psi_maximag = [np.max(objs[Pm].n2.N20_psi['g'].imag) for Pm in Pms]
    psi_minreal = [np.min(objs[Pm].n2.N20_psi['g'].real) for Pm in Pms]
    psi_minimag = [np.min(objs[Pm].n2.N20_psi['g'].imag) for Pm in Pms]
    
    psimax = max(np.nanmax(psi_maxreal), np.nanmax(psi_maximag))
    psimin = min(np.nanmin(psi_minreal), np.nanmin(psi_minimag))
    
    return psimax, psimin
    
def get_minmax_N20_u_allPms(objs, Pms):

    u_maxreal = [np.max(objs[Pm].n2.N20_u['g'].real) for Pm in Pms]
    u_maximag = [np.max(objs[Pm].n2.N20_u['g'].imag) for Pm in Pms]
    u_minreal = [np.min(objs[Pm].n2.N20_u['g'].real) for Pm in Pms]
    u_minimag = [np.min(objs[Pm].n2.N20_u['g'].imag) for Pm in Pms]
    
    umax = max(np.nanmax(u_maxreal), np.nanmax(u_maximag))
    umin = min(np.nanmin(u_minreal), np.nanmin(u_minimag))
    
    return umax, umin
    
def get_minmax_N20_A_allPms(objs, Pms):

    A_maxreal = [np.max(objs[Pm].n2.N20_A['g'].real) for Pm in Pms]
    A_maximag = [np.max(objs[Pm].n2.N20_A['g'].imag) for Pm in Pms]
    A_minreal = [np.min(objs[Pm].n2.N20_A['g'].real) for Pm in Pms]
    A_minimag = [np.min(objs[Pm].n2.N20_A['g'].imag) for Pm in Pms]
    
    Amax = max(np.nanmax(A_maxreal), np.nanmax(A_maximag))
    Amin = min(np.nanmin(A_minreal), np.nanmin(A_minimag))
    
    return Amax, Amin
    
def get_minmax_N20_B_allPms(objs, Pms):

    B_maxreal = [np.max(objs[Pm].n2.N20_B['g'].real) for Pm in Pms]
    B_maximag = [np.max(objs[Pm].n2.N20_B['g'].imag) for Pm in Pms]
    B_minreal = [np.min(objs[Pm].n2.N20_B['g'].real) for Pm in Pms]
    B_minimag = [np.min(objs[Pm].n2.N20_B['g'].imag) for Pm in Pms]
    
    Bmax = max(np.nanmax(B_maxreal), np.nanmax(B_maximag))
    Bmin = min(np.nanmin(B_minreal), np.nanmin(B_minimag))
    
    return Bmax, Bmin
    
def get_minmax_N22_psi_allPms(objs, Pms):

    psi_maxreal = [np.max(objs[Pm].n2.N22_psi['g'].real) for Pm in Pms]
    psi_maximag = [np.max(objs[Pm].n2.N22_psi['g'].imag) for Pm in Pms]
    psi_minreal = [np.min(objs[Pm].n2.N22_psi['g'].real) for Pm in Pms]
    psi_minimag = [np.min(objs[Pm].n2.N22_psi['g'].imag) for Pm in Pms]
    
    psimax = max(np.nanmax(psi_maxreal), np.nanmax(psi_maximag))
    psimin = min(np.nanmin(psi_minreal), np.nanmin(psi_minimag))
    
    return psimax, psimin
    
def get_minmax_N22_u_allPms(objs, Pms):

    u_maxreal = [np.max(objs[Pm].n2.N22_u['g'].real) for Pm in Pms]
    u_maximag = [np.max(objs[Pm].n2.N22_u['g'].imag) for Pm in Pms]
    u_minreal = [np.min(objs[Pm].n2.N22_u['g'].real) for Pm in Pms]
    u_minimag = [np.min(objs[Pm].n2.N22_u['g'].imag) for Pm in Pms]
    
    umax = max(np.nanmax(u_maxreal), np.nanmax(u_maximag))
    umin = min(np.nanmin(u_minreal), np.nanmin(u_minimag))
    
    return umax, umin
    
def get_minmax_N22_A_allPms(objs, Pms):

    A_maxreal = [np.max(objs[Pm].n2.N22_A['g'].real) for Pm in Pms]
    A_maximag = [np.max(objs[Pm].n2.N22_A['g'].imag) for Pm in Pms]
    A_minreal = [np.min(objs[Pm].n2.N22_A['g'].real) for Pm in Pms]
    A_minimag = [np.min(objs[Pm].n2.N22_A['g'].imag) for Pm in Pms]
    
    Amax = max(np.nanmax(A_maxreal), np.nanmax(A_maximag))
    Amin = min(np.nanmin(A_minreal), np.nanmin(A_minimag))
    
    return Amax, Amin
    
def get_minmax_N22_B_allPms(objs, Pms):

    B_maxreal = [np.max(objs[Pm].n2.N22_B['g'].real) for Pm in Pms]
    B_maximag = [np.max(objs[Pm].n2.N22_B['g'].imag) for Pm in Pms]
    B_minreal = [np.min(objs[Pm].n2.N22_B['g'].real) for Pm in Pms]
    B_minimag = [np.min(objs[Pm].n2.N22_B['g'].imag) for Pm in Pms]
    
    Bmax = max(np.nanmax(B_maxreal), np.nanmax(B_maximag))
    Bmin = min(np.nanmin(B_minreal), np.nanmin(B_minimag))
    
    return Bmax, Bmin
    
def get_minmax_N31_psi_allPms(objs, Pms):

    psi_maxreal = [np.max(objs[Pm].n3.N31_psi['g'].real) for Pm in Pms]
    psi_maximag = [np.max(objs[Pm].n3.N31_psi['g'].imag) for Pm in Pms]
    psi_minreal = [np.min(objs[Pm].n3.N31_psi['g'].real) for Pm in Pms]
    psi_minimag = [np.min(objs[Pm].n3.N31_psi['g'].imag) for Pm in Pms]
    
    psimax = max(np.nanmax(psi_maxreal), np.nanmax(psi_maximag))
    psimin = min(np.nanmin(psi_minreal), np.nanmin(psi_minimag))
    
    return psimax, psimin
    
def get_minmax_N31_u_allPms(objs, Pms):

    u_maxreal = [np.max(objs[Pm].n3.N31_u['g'].real) for Pm in Pms]
    u_maximag = [np.max(objs[Pm].n3.N31_u['g'].imag) for Pm in Pms]
    u_minreal = [np.min(objs[Pm].n3.N31_u['g'].real) for Pm in Pms]
    u_minimag = [np.min(objs[Pm].n3.N31_u['g'].imag) for Pm in Pms]
    
    umax = max(np.nanmax(u_maxreal), np.nanmax(u_maximag))
    umin = min(np.nanmin(u_minreal), np.nanmin(u_minimag))
    
    return umax, umin
    
def get_minmax_N31_A_allPms(objs, Pms):

    A_maxreal = [np.max(objs[Pm].n3.N31_A['g'].real) for Pm in Pms]
    A_maximag = [np.max(objs[Pm].n3.N31_A['g'].imag) for Pm in Pms]
    A_minreal = [np.min(objs[Pm].n3.N31_A['g'].real) for Pm in Pms]
    A_minimag = [np.min(objs[Pm].n3.N31_A['g'].imag) for Pm in Pms]
    
    Amax = max(np.nanmax(A_maxreal), np.nanmax(A_maximag))
    Amin = min(np.nanmin(A_minreal), np.nanmin(A_minimag))
    
    return Amax, Amin
    
def get_minmax_N31_B_allPms(objs, Pms):

    B_maxreal = [np.max(objs[Pm].n3.N31_B['g'].real) for Pm in Pms]
    B_maximag = [np.max(objs[Pm].n3.N31_B['g'].imag) for Pm in Pms]
    B_minreal = [np.min(objs[Pm].n3.N31_B['g'].real) for Pm in Pms]
    B_minimag = [np.min(objs[Pm].n3.N31_B['g'].imag) for Pm in Pms]
    
    Bmax = max(np.nanmax(B_maxreal), np.nanmax(B_maximag))
    Bmin = min(np.nanmin(B_minreal), np.nanmin(B_minimag))
    
    return Bmax, Bmin
    
def get_minmax_o20_psi_allPms(objs, Pms):

    psi_maxreal = [np.max(objs[Pm].o2.psi20['g'].real) for Pm in Pms]
    psi_maximag = [np.max(objs[Pm].o2.psi20['g'].imag) for Pm in Pms]
    psi_minreal = [np.min(objs[Pm].o2.psi20['g'].real) for Pm in Pms]
    psi_minimag = [np.min(objs[Pm].o2.psi20['g'].imag) for Pm in Pms]
    
    psimax = max(np.nanmax(psi_maxreal), np.nanmax(psi_maximag))
    psimin = min(np.nanmin(psi_minreal), np.nanmin(psi_minimag))
    
    return psimax, psimin
    
def get_minmax_o20_u_allPms(objs, Pms):

    u_maxreal = [np.max(objs[Pm].o2.u20['g'].real) for Pm in Pms]
    u_maximag = [np.max(objs[Pm].o2.u20['g'].imag) for Pm in Pms]
    u_minreal = [np.min(objs[Pm].o2.u20['g'].real) for Pm in Pms]
    u_minimag = [np.min(objs[Pm].o2.u20['g'].imag) for Pm in Pms]
    
    umax = max(np.nanmax(u_maxreal), np.nanmax(u_maximag))
    umin = min(np.nanmin(u_minreal), np.nanmin(u_minimag))
    
    return umax, umin
    
def get_minmax_o20_A_allPms(objs, Pms):

    A_maxreal = [np.max(objs[Pm].o2.A20['g'].real) for Pm in Pms]
    A_maximag = [np.max(objs[Pm].o2.A20['g'].imag) for Pm in Pms]
    A_minreal = [np.min(objs[Pm].o2.A20['g'].real) for Pm in Pms]
    A_minimag = [np.min(objs[Pm].o2.A20['g'].imag) for Pm in Pms]
    
    Amax = max(np.nanmax(A_maxreal), np.nanmax(A_maximag))
    Amin = min(np.nanmin(A_minreal), np.nanmin(A_minimag))
    
    return Amax, Amin
    
def get_minmax_o20_B_allPms(objs, Pms):

    B_maxreal = [np.max(objs[Pm].o2.B20['g'].real) for Pm in Pms]
    B_maximag = [np.max(objs[Pm].o2.B20['g'].imag) for Pm in Pms]
    B_minreal = [np.min(objs[Pm].o2.B20['g'].real) for Pm in Pms]
    B_minimag = [np.min(objs[Pm].o2.B20['g'].imag) for Pm in Pms]
    
    Bmax = max(np.nanmax(B_maxreal), np.nanmax(B_maximag))
    Bmin = min(np.nanmin(B_minreal), np.nanmin(B_minimag))
    
    return Bmax, Bmin
    
def get_minmax_o21_psi_allPms(objs, Pms):

    psi_maxreal = [np.max(objs[Pm].o2.psi21['g'].real) for Pm in Pms]
    psi_maximag = [np.max(objs[Pm].o2.psi21['g'].imag) for Pm in Pms]
    psi_minreal = [np.min(objs[Pm].o2.psi21['g'].real) for Pm in Pms]
    psi_minimag = [np.min(objs[Pm].o2.psi21['g'].imag) for Pm in Pms]
    
    psimax = max(np.nanmax(psi_maxreal), np.nanmax(psi_maximag))
    psimin = min(np.nanmin(psi_minreal), np.nanmin(psi_minimag))
    
    return psimax, psimin
    
def get_minmax_o21_u_allPms(objs, Pms):

    u_maxreal = [np.max(objs[Pm].o2.u21['g'].real) for Pm in Pms]
    u_maximag = [np.max(objs[Pm].o2.u21['g'].imag) for Pm in Pms]
    u_minreal = [np.min(objs[Pm].o2.u21['g'].real) for Pm in Pms]
    u_minimag = [np.min(objs[Pm].o2.u21['g'].imag) for Pm in Pms]
    
    umax = max(np.nanmax(u_maxreal), np.nanmax(u_maximag))
    umin = min(np.nanmin(u_minreal), np.nanmin(u_minimag))
    
    return umax, umin
    
def get_minmax_o21_A_allPms(objs, Pms):

    A_maxreal = [np.max(objs[Pm].o2.A21['g'].real) for Pm in Pms]
    A_maximag = [np.max(objs[Pm].o2.A21['g'].imag) for Pm in Pms]
    A_minreal = [np.min(objs[Pm].o2.A21['g'].real) for Pm in Pms]
    A_minimag = [np.min(objs[Pm].o2.A21['g'].imag) for Pm in Pms]
    
    Amax = max(np.nanmax(A_maxreal), np.nanmax(A_maximag))
    Amin = min(np.nanmin(A_minreal), np.nanmin(A_minimag))
    
    return Amax, Amin
    
def get_minmax_o21_B_allPms(objs, Pms):

    B_maxreal = [np.max(objs[Pm].o2.B21['g'].real) for Pm in Pms]
    B_maximag = [np.max(objs[Pm].o2.B21['g'].imag) for Pm in Pms]
    B_minreal = [np.min(objs[Pm].o2.B21['g'].real) for Pm in Pms]
    B_minimag = [np.min(objs[Pm].o2.B21['g'].imag) for Pm in Pms]
    
    Bmax = max(np.nanmax(B_maxreal), np.nanmax(B_maximag))
    Bmin = min(np.nanmin(B_minreal), np.nanmin(B_minimag))
    
    return Bmax, Bmin
    
def get_minmax_o22_psi_allPms(objs, Pms):

    psi_maxreal = [np.max(objs[Pm].o2.psi22['g'].real) for Pm in Pms]
    psi_maximag = [np.max(objs[Pm].o2.psi22['g'].imag) for Pm in Pms]
    psi_minreal = [np.min(objs[Pm].o2.psi22['g'].real) for Pm in Pms]
    psi_minimag = [np.min(objs[Pm].o2.psi22['g'].imag) for Pm in Pms]
    
    psimax = max(np.nanmax(psi_maxreal), np.nanmax(psi_maximag))
    psimin = min(np.nanmin(psi_minreal), np.nanmin(psi_minimag))
    
    return psimax, psimin
    
def get_minmax_o22_u_allPms(objs, Pms):

    u_maxreal = [np.max(objs[Pm].o2.u22['g'].real) for Pm in Pms]
    u_maximag = [np.max(objs[Pm].o2.u22['g'].imag) for Pm in Pms]
    u_minreal = [np.min(objs[Pm].o2.u22['g'].real) for Pm in Pms]
    u_minimag = [np.min(objs[Pm].o2.u22['g'].imag) for Pm in Pms]
    
    umax = max(np.nanmax(u_maxreal), np.nanmax(u_maximag))
    umin = min(np.nanmin(u_minreal), np.nanmin(u_minimag))
    
    return umax, umin
    
def get_minmax_o22_A_allPms(objs, Pms):

    A_maxreal = [np.max(objs[Pm].o2.A22['g'].real) for Pm in Pms]
    A_maximag = [np.max(objs[Pm].o2.A22['g'].imag) for Pm in Pms]
    A_minreal = [np.min(objs[Pm].o2.A22['g'].real) for Pm in Pms]
    A_minimag = [np.min(objs[Pm].o2.A22['g'].imag) for Pm in Pms]
    
    Amax = max(np.nanmax(A_maxreal), np.nanmax(A_maximag))
    Amin = min(np.nanmin(A_minreal), np.nanmin(A_minimag))
    
    return Amax, Amin
    
def get_minmax_o22_B_allPms(objs, Pms):

    B_maxreal = [np.max(objs[Pm].o2.B22['g'].real) for Pm in Pms]
    B_maximag = [np.max(objs[Pm].o2.B22['g'].imag) for Pm in Pms]
    B_minreal = [np.min(objs[Pm].o2.B22['g'].real) for Pm in Pms]
    B_minimag = [np.min(objs[Pm].o2.B22['g'].imag) for Pm in Pms]
    
    Bmax = max(np.nanmax(B_maxreal), np.nanmax(B_maximag))
    Bmin = min(np.nanmin(B_minreal), np.nanmin(B_minimag))
    
    return Bmax, Bmin

def linfunc(x, a, b):
    return a*x + b
    
def fit_loglog(xdata, ydata):
    xd, yd = np.log10(xdata), np.log10(ydata)
    polycoef = polyfit(xd, yd, 1)
    yfit = 10**( polycoef[0]*xd+polycoef[1] )
    print("slope", 10**polycoef[0], "intercept", 10**polycoef[1],)
    plt.plot(xdata, yfit, label = r"$"+str(polycoef[1])+"*x^{"+str(polycoef[0])+"}$")
    plt.legend()

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
    
    
