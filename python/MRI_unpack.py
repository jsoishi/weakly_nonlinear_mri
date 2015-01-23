import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, norm
import pylab
import copy
import pickle
import random

import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
matplotlib.rcParams.update({'figure.autolayout': True})

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


q = 1.5
beta = 25.0
norm = True

coeffs = {}

test = True

if test == True:

    # For false test
    gridnum = 64
    norm = False
    
    #Pm = 1E-4: Q = 0.747, Rm = 4.88
    Pm = 1.0E-4
    Q = 0.747
    Rm = 4.88
    coeffs[0] = pickle.load(open("pspace/test_coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+"_norm_"+str(norm)+".p", "rb"))
    

    #Pm = 1E-3, Q = 0.75, Rm = 4.8775
    Pm = 1.0E-3
    Q = 0.75
    Rm = 4.8775
    coeffs[1] = pickle.load(open("pspace/test_coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+"_norm_"+str(norm)+".p", "rb"))

    #Pm = 1E-2: Q = 0.757, Rm = 4.93
    Pm = 1.0E-2
    Q = 0.757
    Rm = 4.93
    coeffs[2] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+"_norm_"+str(norm)+".p", "rb"))


else:
    #Pm = 1E-7: Q = 0.745, Rm = 4.90
    """
    gridnum = 1024#512
    Pm = 1.0E-7
    Q = 0.745
    Rm = 4.90

    coeffs[0] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+".p", "rb"))
    """

    # Pm = 1E-6: Q = 0.75, Rm = 4.88, beta = 25
    gridnum = 1024
    Pm = 1.0E-6
    Q = 0.75
    Rm = 4.88
    coeffs[0] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+"_norm_"+str(norm)+".p", "rb"))


    #Pm = 1E-5, Q = 0.747, Rm = 4.88
    gridnum = 512
    Pm = 1.0E-5
    Q = 0.747
    Rm = 4.88
    coeffs[1] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+"_norm_"+str(norm)+".p", "rb"))


    #Pm = 1E-4: Q = 0.747, Rm = 4.88
    """
    Pm = 1.0E-4
    Q = 0.747
    Rm = 4.88
    coeffs[0] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+"_norm_"+str(norm)+".p", "rb"))
    """

    #Pm = 1E-3, Q = 0.75, Rm = 4.8775
    gridnum = 256
    Pm = 1.0E-3
    Q = 0.75
    Rm = 4.8775
    coeffs[2] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+"_norm_"+str(norm)+".p", "rb"))

    #Pm = 1E-2: Q = 0.757, Rm = 4.93
    gridnum =  256
    Pm = 1.0E-2
    Q = 0.757
    Rm = 4.93
    coeffs[3] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+"_norm_"+str(norm)+".p", "rb"))

num = len(coeffs)
print("Considering %d Pm's" % num)
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
    
def plot1():
    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    #ax1.semilogx(Pm_arr, np.abs(c_arr/a_arr), '.')
    ax1.semilogx(Pm_arr, np.abs(c_arr/a_arr), '.')
    ax1.set_title("c/a: nonlinear term")
    ax1.set_xlim(10**(-7.5), 10**(-1.5))

    ax2 = fig.add_subplot(412)
    #ax2.semilogx(Pm_arr, np.abs(b_arr/a_arr), '.')
    ax2.semilogx(Pm_arr, np.abs(1j*Q_arr*b_arr/a_arr), '.')
    ax2.set_title("b/a: linear term")
    ax2.set_xlim(10**(-7.5), 10**(-1.5))

    ax3 = fig.add_subplot(413)
    #ax3.semilogx(Pm_arr, np.abs(h_arr/a_arr), '.')
    ax3.semilogx(Pm_arr, np.abs(h_arr/a_arr), '.')
    ax3.set_title("h/a: diffusion term")
    ax3.set_xlim(10**(-7.5), 10**(-1.5))

    ax4 = fig.add_subplot(414)
    ax4.semilogx(Pm_arr, np.abs(1j*Q_arr**3*g_arr/a_arr), '.')
    ax4.set_title("g/a: additional linear term")
    ax4.set_xlim(10**(-7.5), 10**(-1.5))

def plot2():
    # Reproducing Umurhan+ Fig 3
    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    #ax1.semilogx(Pm_arr, np.abs(c_arr/a_arr), '.')
    ax1.semilogx(Pm_arr, Rm_arr, '.')
    ax1.set_title("Rm")
    ax1.set_xlim(10**(-7.5), 10**(-1.5))

    ax2 = fig.add_subplot(512)
    #ax2.semilogx(Pm_arr, np.abs(b_arr/a_arr), '.')
    ax2.semilogx(Pm_arr, Q_arr, '.')
    ax2.set_title("Q")
    ax2.set_xlim(10**(-7.5), 10**(-1.5))

    ax3 = fig.add_subplot(513)
    #ax3.semilogx(Pm_arr, np.abs(h_arr/a_arr), '.')
    ax3.semilogx(Pm_arr, h_arr/a_arr, '.')
    ax3.set_title("D = h/a: diffusion term")
    ax3.set_xlim(10**(-7.5), 10**(-1.5))

    ax4 = fig.add_subplot(514)
    ax4.semilogx(Pm_arr, 1j*Q_arr*b_arr/a_arr, '.')
    ax4.set_title("lambda = i*Q*b/a: first linear term")
    ax4.set_xlim(10**(-7.5), 10**(-1.5))

    ax5 = fig.add_subplot(515)
    ax5.semilogx(Pm_arr, np.abs(1j*Q_arr**3*g_arr/a_arr), '.')
    ax5.set_title("i*Q**3*g/a: additional linear term")
    ax5.set_xlim(10**(-7.5), 10**(-1.5))

def plot3():
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

# Amplitude alpha vs time
def plot4():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    for i in range(num):
        #if i == 1:
        #    print("i = 1")
        #else: 
            ax1.plot(coeffs[i]["t_array"], coeffs[i]["alpha_array"].real, '.', label = coeffs[i]["Pm"])
            #ax1.plot(coeffs[i]["t_array"], np.sqrt(coeffs[i]["alpha_array"].real**2 + coeffs[i]["alpha_array"].imag**2), '.', label = coeffs[i]["Pm"])
            
            #ax1.plot(coeffs[i]["t_array"], coeffs[i]["alpha_array"][:, 0], '.', label = coeffs[i]["Pm"])
            #ax1.loglog(coeffs[i]["Pm"], coeffs[i]["alpha_array"][-1][0], '.', label = coeffs[i]["Pm"])
            #ax1.semilogx(coeffs[i]["Pm"], coeffs[i]["c"]/coeffs[i]["a"], '.', label = coeffs[i]["Pm"])

    plt.legend()
    """
    print(coeffs[4]["Pm"])
    ax1.plot(coeffs[4]["t_array"], coeffs[4]["alpha_array"][:, 0], '.')
    """
    
# Width of the boundary layer from u_y first order
def boundary_layers():

    boundary_widths = np.zeros(len(coeffs))
    Pms = np.zeros(len(coeffs))
    for i in range(len(coeffs)):
        c = coeffs[i]["u_y first order"]
        x = coeffs[i]["x"]
        Pms[i] = coeffs[i]["Pm"]
        
        plt.figure()
        uy = c[40, :]
        plt.plot(x, uy)
        plt.plot(x, np.zeros(len(x)))
        plt.title("Pm "+str(Pms[i]))
        
        # Determine where u_y first dips below zero
        lt = np.where(uy < 0)
        first_crossing = lt[0]
        
        boundary_widths[i] = first_crossing[0]
        
    print(boundary_widths)
    print(Pms)
    #plt.plot(Pms, boundary_widths)
        
def plot_widths_byhand():

    Pms = [1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
    widths = [0.248, 0.112, 0.055, 0.026, 0.013]
    
    plt.figure()
    plt.loglog(Pms, widths, '.')
    plt.xlim(1E-8, 1E-2)
        
def saturation_amp_from_coeffs(sqmag=False):
    fig = plt.figure()
    Pms = np.zeros(len(coeffs))
    asat = np.zeros(len(coeffs), dtype = np.complex128)
    for i in range(len(coeffs)):
        Pms[i] = coeffs[i]["Pm"]
        asat[i] = np.sqrt((coeffs[i]["b"]*1j*coeffs[i]["Q"] - coeffs[i]["g"]*1j*coeffs[i]["Q"]**3)/coeffs[i]["c"])
        
    print(Pms, asat)
    
    if sqmag == True:
        plt.loglog(Pms, asat.real**2 + asat.imag**2, "+", markersize = 10)
    else:
        plt.loglog(Pms, asat, "+", markersize = 10)
    
    plt.xlim(min(Pms) - min(Pms)/2, max(Pms) + max(Pms)/2)
    plt.xlabel("Pm")
    plt.ylabel(r"$\alpha_{saturation}$")

    
def plot5():
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    
    cy1 = coeffs[0]["u_y first order"]
    cy2 = coeffs[0]["u_y second order"]
    x = coeffs[0]["x"]    
    jj = np.zeros(len(x))
    
    ax1.plot(x, jj, color="grey", ls='--')
    ax1.plot(x, 0.5*cy1[25, :].real, color="purple", lw=2)
    ax1.plot(x, 0.5**2*cy2[25, :].real, color="orange", lw=2)
    ax1.plot(x, 0.5*cy1[25, :].real + 0.5**2*cy2[25, :].real, color="black")
    
    ax2 = fig.add_subplot(312)
    ax2.plot(x, jj, color="grey", ls='--')
    ax2.plot(x, 0.5*cy1[128, :].real, color="purple", lw=2)
    ax2.plot(x, 0.5**2*cy2[128, :].real, color="orange", lw=2)
    ax2.plot(x, 0.5*cy1[128, :].real + 0.5**2*cy2[128, :].real, color="black")

    
    ax3 = fig.add_subplot(313)
    ax3.plot(x, jj, color="grey", ls='--')
    ax3.plot(x, 0.5*cy1[231, :].real, color="purple", lw=2)
    ax3.plot(x, 0.5**2*cy2[231, :].real, color="orange", lw=2)
    ax3.plot(x, 0.5*cy1[231, :].real + 0.5**2*cy2[231, :].real, color="black")
    

def plotstress():
    for i in range(num):
        if i > -1:#0:
        
            if i > 1:
                gridnum = 128#256
            else:
                gridnum = 256
        
            print(coeffs[i]["Pm"])
            """
            TR_firstorder = coeffs[i]["u_x first order"].real*coeffs[i]["u_y first order"].real
            TM_firstorder = -(2.0/coeffs[i]["beta"])*coeffs[i]["B_x first order"].real*coeffs[i]["B_y first order"].real

            TR_secondorder = coeffs[i]["u_x second order"].real*coeffs[i]["u_y second order"].real
            TM_secondorder = -(2.0/coeffs[i]["beta"])*coeffs[i]["B_x second order"].real*coeffs[i]["B_y second order"].real

            TR_firstorder = coeffs[i]["u_x first order"]*coeffs[i]["u_y first order"]
            TM_firstorder = -(2.0/coeffs[i]["beta"])*coeffs[i]["B_x first order"]*coeffs[i]["B_y first order"]

            TR_secondorder = coeffs[i]["u_x second order"]*coeffs[i]["u_y second order"]
            TM_secondorder = -(2.0/coeffs[i]["beta"])*coeffs[i]["B_x second order"]*coeffs[i]["B_y second order"]
            """
            TR = (0.5*coeffs[i]["u_x first order"].real + 0.5**2*coeffs[i]["u_x second order"].real)*(0.5*coeffs[i]["u_y first order"].real + 0.5**2*coeffs[i]["u_y second order"].real)
            TM = -(2.0/coeffs[i]["beta"])*(0.5*coeffs[i]["B_x first order"].real + 0.5**2*coeffs[i]["B_x second order"].real)*(0.5*coeffs[i]["B_y first order"].real + 0.5**2*coeffs[i]["B_y second order"].real) 

            T = TR + TM
            """
            T_firstorder = 0.5*TR_firstorder + 0.5*TM_firstorder
            T_secondorder = 0.5**2*TR_secondorder + 0.5**2*TM_secondorder
            """
            nz = gridnum
            Lz = 2*np.pi/coeffs[i]["Q"]
            z = np.linspace(0, Lz, nz, endpoint=False)
            zz = z.reshape(nz, 1)
            dz = z[1] - z[0]
            x = coeffs[i]["x"]
        
            fig = plt.figure()
            ax = fig.add_subplot(111)
            #info = ax.pcolormesh(x, z, T_firstorder + T_secondorder, cmap="RdBu_r")
            info = ax.pcolormesh(x, z, T, cmap="RdBu_r")
            #info = ax.contour(x, z, T_firstorder + T_secondorder, cmap="RdBu_r")
            #info = ax.pcolormesh(x, z, TM_firstorder + TM_secondorder, cmap="RdBu_r")
            cbar = plt.colorbar(info)
            cbar.ax.tick_params(labelsize = 20)
            #ax.set_title("Pm = "+str(coeffs[i]["Pm"]))

            ax.set_ylim(0, Lz - dz)
            ax.set_xlim(-1, 1)
        
            plt.tick_params(labelsize = 20)
            rr = np.zeros(len(x))
            plt.plot(x, rr+z[25], color="black", lw=3, ls='--')
            plt.plot(x, rr+z[128], color="black", lw=3, ls='--')
            plt.plot(x, rr+z[231], color="black", lw=3, ls='--')
        
            """
            fig = plt.figure()
            Tint = np.sum(T_firstorder + T_secondorder, axis=0)
            ax2 = fig2.add_subplot(2, 2, i)
            ax2.plot(x, Tint, label=coeffs[i]["Pm"])
            ax2.set_title("Pm = "+str(coeffs[i]["Pm"]))
            """
