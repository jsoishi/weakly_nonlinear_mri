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

coeffs = {}

#Pm = 1E-7: Q = 0.745, Rm = 4.90
gridnum = 1024#512
Pm = 1.0E-7
Q = 0.745
Rm = 4.90

coeffs[0] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+".p", "rb"))

# all the rest have gridnum 256
gridnum = 256

# Pm = 1E-6: Q = 0.75, Rm = 4.88, beta = 25
Pm = 1.0E-6
Q = 0.75
Rm = 4.88

coeffs[1] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+".p", "rb"))

#Pm = 1E-5, Q = 0.747, Rm = 4.88
Pm = 1.0E-5
Q = 0.747
Rm = 4.88

coeffs[2] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+".p", "rb"))

#Pm = 1E-4: Q = 0.747, Rm = 4.88
Pm = 1.0E-4
Q = 0.747
Rm = 4.88

coeffs[3] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+".p", "rb"))

#Pm = 1E-3, Q = 0.75, Rm = 4.8775
Pm = 1.0E-3
Q = 0.75
Rm = 4.8775
coeffs[4] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+".p", "rb"))


#Pm = 1E-2: Q = 0.757, Rm = 4.93
Pm = 1.0E-2
Q = 0.757
Rm = 4.93
coeffs[5] = pickle.load(open("pspace/coeffs_gridnum"+str(gridnum)+"_Pm_"+str(Pm)+"_Q_"+str(Q)+"_Rm_"+str(Rm)+"_q_"+str(q)+"_beta_"+str(beta)+".p", "rb"))


num = 6
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

"""
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
"""
