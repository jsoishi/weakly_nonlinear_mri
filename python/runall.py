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
#matplotlib.rcParams['backend'] = "Qt4Agg"
#matplotlib.rcParams.update({'figure.autolayout': True})

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def loader(Pm = 1E-3, Q = 0.75, Rm = 4.8775, gridnum = 64, norm = False, q = 1.5, beta = 25.0):
    c = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+"_norm_"+str(norm)+".p", "rb"))
    return c

def onerun(Pm = 1E-3, Q = 0.75, Rm = 4.8775, gridnum = 64, norm = False, q = 1.5, beta = 25.0, save = True):

    pc = PlotContours(Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, run = True, norm = norm)
    pc.plot_streams()
    pc.plot_Bfield()

    coeffs = {}
    coeffs["Q"] = Q
    coeffs["Rm"] = Rm
    coeffs["Pm"] = Pm
    coeffs["beta"] = beta
    coeffs["gridnum"] = gridnum
    coeffs["q"] = q
    coeffs["a"] = pc.saa.alpha_amp.a
    coeffs["c"] = pc.saa.alpha_amp.c
    coeffs["ctwiddle"] = pc.saa.alpha_amp.c_twiddle
    coeffs["b"] = pc.saa.alpha_amp.b
    coeffs["h"] = pc.saa.alpha_amp.h
    coeffs["g"] = pc.saa.alpha_amp.g
    coeffs["alpha_s"] = pc.alpha_s
    coeffs["t_array"] = pc.saa.t_array
    coeffs["alpha_array"] = pc.saa.alpha_array[:, 0]
    coeffs["x"] = pc.saa.alpha_amp.x
    coeffs["eps"] = pc.eps
    coeffs["u_x first order"] = pc.V1_ux1
    coeffs["u_y first order"] = pc.V1_u
    coeffs["u_z first order"] = pc.V1_uz1
    coeffs["u_x second order"] = pc.V2_ux1
    coeffs["u_y second order"] = pc.V2_u
    coeffs["u_z second order"] = pc.V2_uz1
    coeffs["B_x first order"] = pc.V1_Bx1
    coeffs["B_y first order"] = pc.V1_B
    coeffs["B_z first order"] = pc.V1_Bz1
    coeffs["B_x second order"] = pc.V2_Bx1
    coeffs["B_y second order"] = pc.V2_B
    coeffs["B_z second order"] = pc.V2_Bz1

    coeffs["A first order"] = pc.V1_A
    coeffs["Psi first order"] = pc.V1_psi

    if save == True:
        outname = "gridnum_"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+"_norm_"+str(norm)
        plot_uy_firstorder(pc, oplot = True, labels = False, outname=outname)
        plot_By_firstorder(pc, oplot = True, labels = False, outname=outname)
        plot_uy_secondorder(pc, oplot = True, labels = False, outname=outname)
        plot_By_secondorder(pc, oplot = True, labels = False, outname=outname)
        plot_uy(pc, oplot = True, labels = False, outname=outname)
        plot_By(pc, oplot = True, labels = False, outname=outname)
        plotN2(pc.saa.alpha_amp.o2, outname=outname)
        plotN3(pc.saa.alpha_amp.n3, outname=outname)
        plotOE(pc.saa.alpha_amp, outname=outname)

    #print(coeffs)
    plt.close("all")
    pickle.dump(coeffs, open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+"_norm_"+str(norm)+".p", "wb"))
    
def sat_amp_from_IVP(coeffs):
    fig = plt.figure()
    Pms = np.zeros(len(coeffs))
    amps = np.zeros(len(coeffs), dtype=np.complex128)
    asat = np.zeros(len(coeffs), dtype=np.complex128)
    for i in range(len(coeffs)):
        Pms[i] = coeffs[i]["Pm"]
        amps[i] = coeffs[i]["alpha_s"]
        asat[i] = np.sqrt((coeffs[i]["b"]*1j*coeffs[i]["Q"] - coeffs[i]["g"]*1j*coeffs[i]["Q"]**3)/coeffs[i]["c"])
    plt.loglog(Pms, amps, '+')
    plt.legend(["from ivp", "from coefficients"])
    plt.xlim(min(Pms) - min(Pms)/2, max(Pms) + max(Pms)/2)
    
def plot_alpha_vs_T(coeffs):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    for i in range(len(coeffs)):
        ax1.plot(coeffs[i]["t_array"], coeffs[i]["alpha_array"], '.', label = coeffs[i]["Pm"])
        
    plt.legend()
    
def saturation_amp_from_coeffs(coeffs, mag=False):
    fig = plt.figure()
    Pms = np.zeros(len(coeffs))
    asat = np.zeros(len(coeffs), dtype = np.complex128)
    for i in range(len(coeffs)):
        Pms[i] = coeffs[i]["Pm"]
        asat[i] = np.sqrt((coeffs[i]["b"]*1j*coeffs[i]["Q"] - coeffs[i]["g"]*1j*coeffs[i]["Q"]**3)/coeffs[i]["c"])
        
    print(Pms, asat)
    
    if mag == True:
        plt.loglog(Pms, np.sqrt(asat.real**2 + asat.imag**2), "+", markersize = 10)
    else:
        plt.loglog(Pms, asat, "+", markersize = 10)
    
    plt.xlim(min(Pms) - min(Pms)/2, max(Pms) + max(Pms)/2)
    plt.xlabel("Pm")
    plt.ylabel(r"$\alpha_{saturation}$")
    
def plot_U_coeffs():

    num = len(coeffs)

    a_arr = np.zeros(num, dtype=np.complex128)
    c_arr = np.zeros(num, dtype=np.complex128)
    b_arr = np.zeros(num, dtype=np.complex128)
    h_arr = np.zeros(num, dtype=np.complex128)
    g_arr = np.zeros(num, dtype=np.complex128)
    Pm_arr = np.zeros(num, dtype=np.complex128)
    Rm_arr = np.zeros(num, dtype=np.complex128)
    Q_arr = np.zeros(num, dtype=np.complex128)
    for i in range(num):
        a_arr[i] = coeffs[i]["a"]
        c_arr[i] = coeffs[i]["c"]
        b_arr[i] = coeffs[i]["b"]
        h_arr[i] = coeffs[i]["h"]
        g_arr[i] = coeffs[i]["g"]
        Pm_arr[i] = coeffs[i]["Pm"]
        Rm_arr[i] = coeffs[i]["Rm"]
        Q_arr[i] = coeffs[i]["Q"]
    
    # Reproducing Umurhan+ Fig 3
    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    #ax1.semilogx(Pm_arr, np.abs(c_arr/a_arr), '.')
    ax1.semilogx(Pm_arr, Rm_arr, '.')
    ax1.set_title("Rm")
    ax1.set_xlim(10**(-7.5), 10**(-1.5))

    ax2 = fig.add_subplot(412)
    #ax2.semilogx(Pm_arr, np.abs(b_arr/a_arr), '.')
    ax2.semilogx(Pm_arr, Q_arr, '.')
    ax2.set_title("Q")
    ax2.set_xlim(10**(-7.5), 10**(-1.5))

    ax3 = fig.add_subplot(413)
    #ax3.semilogx(Pm_arr, np.abs(h_arr/a_arr), '.')
    ax3.semilogx(Pm_arr, h_arr/a_arr, '.')
    ax3.set_title("D = h/a: diffusion term")
    ax3.set_xlim(10**(-7.5), 10**(-1.5))

    ax4 = fig.add_subplot(414)
    ax4.semilogx(Pm_arr, (1j*Q_arr*b_arr - 1j*Q_arr**3*g_arr)/a_arr, '.')
    ax4.set_title("lambda = (i*Q*b - i*Q**3*g)/a: first linear term")
    ax4.set_xlim(10**(-7.5), 10**(-1.5))
    pylab.savefig("test_Uplus_nonalpha_coeffs.png")

    # alpha plot
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.loglog(Pm_arr, c_arr/a_arr, '.')
    ax1.set_title("alpha = c/a: nonlinear term")
    ax1.set_xlim(10**(-7.5), 10**(-1.5))
    ax1.set_xlabel("Pm")
    ax1.set_ylabel("alpha")

    ax2 = fig.add_subplot(122)
    ax2.loglog(Pm_arr, 1.0/(c_arr/a_arr), '.')
    ax2.set_title("1/alpha = 1/(c/a)")
    ax2.set_xlim(10**(-7.5), 10**(-1.5))
    ax2.set_xlabel("Pm")
    ax2.set_ylabel("1/alpha")
    pylab.savefig("test_Uplus_alpha_coeffs.png")
    
    
    
if __name__ == "__main__":
    gridnum = 64
    print(gridnum)
    x_basis = Chebyshev(gridnum)
    domain = Domain([x_basis], grid_dtype=np.complex128)

    coeffs = {}
    norm = True

    #Pm = 1E-2: Q = 0.757, Rm = 4.93

    #Pm = 1E-3: Q = 0.75, Rm = 4.8775
    onerun(Pm = 1E-3, Q = 0.75, Rm = 4.88, gridnum = 64, norm = norm)
    coeffs[0] = loader(Pm = 1E-3, Q = 0.75, Rm = 4.88, gridnum = 64, norm = norm)

    #Pm = 1E-4: Q = 0.747, Rm = 4.88
    onerun(Pm = 1E-4, Q = 0.75, Rm = 4.88, gridnum = 64, norm = norm)
    coeffs[1] = loader(Pm = 1E-4, Q = 0.75, Rm = 4.88, gridnum = 64, norm = norm)

    #Pm = 1E-5, Q = 0.747, Rm = 4.88
    onerun(Pm = 1E-5, Q = 0.75, Rm = 4.88, gridnum = 64, norm = norm)
    coeffs[2] = loader(Pm = 1E-5, Q = 0.75, Rm = 4.88, gridnum = 64, norm = norm)

    # Plotting
    plot_alpha_vs_T(coeffs)
    pylab.savefig("test_plot4.png")
    saturation_amp_from_coeffs(coeffs, mag=False)
    pylab.savefig("test_sat_amp_from_coeffs.png")
    sat_amp_from_IVP(coeffs)
    pylab.savefig("test_sat_amp_from_ivp.png")
    plot_U_coeffs()
    #pylab.savefig("test_Uplus_coeffs.png")


    # Pm = 1E-6: Q = 0.75, Rm = 4.88, beta = 25
    #Pm = 1E-7: Q = 0.745, Rm = 4.90
    #Pm = 1E-8: Q = 0.815, Rm = 4.75
