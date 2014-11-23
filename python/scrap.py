from multiprocessing import Pool
import multiprocessing as mp
import itertools
import numpy as np
import matplotlib.pyplot as plt
#from dedalus2.tools.config import config
#config['logging']['stdout_level'] = 'critical'
#from dedalus2.public import *
#from dedalus2.pde.solvers import LinearEigenvalue
import pylab
import pickle

path = "/Users/susanclark/weakly_nonlinear_mri/python/multirun/"

#data = pickle.load(open(path+"Pm_0.0001_Q_0.05_dQ_0.05_Rm_0.05_dRm_0.05.p", "rb"))
#data = pickle.load(open(path+"Pm_1e-05_Q_0.05_dQ_0.05_Rm_0.05_dRm_0.05.p", "rb"))
#data = pickle.load(open(path+"Pm_1e-06_Q_0.05_dQ_0.1_Rm_0.05_dRm_0.1.p", "rb"))
#data = pickle.load(open(path+"Pm_1e-06_Q_0.5_dQ_0.01_Rm_4.6_dRm_0.01.p", "rb"))
#data = pickle.load(open(path+"Pm_1e-07_Q_0.5_dQ_0.01_Rm_4.6_dRm_0.01.p", "rb"))
#data = pickle.load(open(path+"Pm_1e-08_Q_0.5_dQ_0.01_Rm_4.6_dRm_0.01.p", "rb"))
#data = pickle.load(open(path+"Pm_1e-09_Q_0.5_dQ_0.01_Rm_4.6_dRm_0.01.p", "rb"))

#data = pickle.load(open(path+"beta_2.5_Q_0.5_dQ_0.01_Rm_4.6_dRm_0.01.p", "rb"))
#data = pickle.load(open(path+"beta_0.0025_Pm_0.001_Q_0.2_dQ_0.05_Rm_5.0_dRm_0.05.p", "rb"))
#data = pickle.load(open(path+"hmri/Pm_0.001_beta_25_Q_0.001_dQ_0.005_Rm_0.05_dRm_0.05.p", "rb"))
#data = pickle.load(open(path+"hmri/Pm_0.001_beta_25_Q_0.2_dQ_0.005_Rm_0.05_dRm_0.05.p", "rb"))
#data = pickle.load(open(path+"hmri/Pm_0.01_beta_25_Q_0.001_dQ_0.005_Rm_0.05_dRm_0.05.p", "rb"))
#data = pickle.load(open(path+"hmri/Pm_0.01_beta_25_Q_0.0001_dQ_0.005_Rm_0.005_dRm_0.005.p", "rb"))

#data = pickle.load(open(path+"hmri/Pm_0.001_beta_0.025_Q_0.0001_dQ_0.005_Rm_0.005_dRm_0.005.p", "rb"))
#data = pickle.load(open(path+"hmri/Pm_0.001_beta_2.5_Q_0.0001_dQ_0.005_Rm_0.005_dRm_0.005.p", "rb"))
#data = pickle.load(open(path+"hmri/Pm_0.1_beta_25_Q_0.0001_dQ_0.005_Rm_0.005_dRm_0.005.p", "rb"))

#data = pickle.load(open(path+"hmri/Pm_5e-06_beta_0.0057_Q_0.0_dQ_0.005_Rm_0.015_dRm_0.5.p", "rb"))

#data = pickle.load(open(path+"hmri/Pm_5e-06_beta_0.0057_Q_0.0_dQ_0.05_Rm_0.005_dRm_0.005.p", "rb"))

#data = pickle.load(open(path+"hmri/Pm_5e-06_beta_0.0057_Q_0.0_dQ_0.05_Rm_0.005_dRm_0.05.p", "rb"))

#Pm 1E-4 standard MRI
#data = pickle.load(open(path+"Pm_0.0001_Q_0.5_dQ_0.005_Rm_4.7_dRm_0.01.p", "rb"))

#Pm 1E-5
#data = pickle.load(open(path+"Pm_1e-05_Q_0.5_dQ_0.005_Rm_4.7_dRm_0.01.p", "rb"))

#Pm 1E-2
data = pickle.load(open(path+"Pm_0.01_Q_0.5_dQ_0.005_Rm_4.7_dRm_0.01.p", "rb"))

ids = np.zeros(len(data))
eval = np.zeros(len(data), np.complex128)
eval_pos = np.zeros(len(data), np.complex128)
eval_neg = np.zeros(len(data), np.complex128)
hmri = False

Pm = 0.00001 #Pm = Rm/R
q = 3/2.
Co = 0.08

Rmsearch = np.arange(4.6, 5.1, 0.01)
Qsearch = np.arange(0.5, 1.0, 0.01)

dQ = 0.005
dRm = 0.5

#big hmri search...
#Rmsearch = np.arange(0.005, 0.5, dRm)
#Qsearch = np.arange(0.0001, 0.4, dQ)

#hmri liu values
Rmsearch = np.arange(0.015, 0.016, dRm)
Qsearch = np.arange(0.0, 10.0, dQ)

#dQ = 0.05
#dRm = 0.005

#Rmsearch = np.arange(0.015, 0.016, dRm)
#Rmsearch = np.arange(0.005, 0.1, dRm) 
#Qsearch = np.arange(0.0, 10.0, dQ)

#Rmsearch = Rmsearch[0:len(data)]
#Qsearch = Qsearch[0:len(data)]

Rmsearch = np.arange(0.005, 0.5, 0.05)
Qsearch = np.arange(0.0, 10.0, 0.05)


Qsearch = np.arange(0.5, 1.0, 0.005)
Rmsearch = np.arange(4.7, 5.1, 0.01)

QRm = np.array(list(itertools.product(Qsearch, Rmsearch)))
Qs = QRm[:, 0]
Rms = QRm[:, 1]

for i in range(len(data)):
    jj = data.popitem()
    
    if hmri == False:
        eval[i] = jj[1]
        ids[i] = jj[0]
    
    else:
        qq = jj[1]
        q0 = qq[0]
        q1 = qq[1]
        ids[i] = jj[0]

        if np.isnan(np.sum(q0)):
            eval_pos[i] = q0
        else:
            eval_pos[i] = q0[0]
    
        if np.isnan(np.sum(q1)):
            eval_neg[i] = q1
        else:
            eval_neg[i] = q1[0]
            

Q = Qs[ids.astype(int)]
Rm = Rms[ids.astype(int)]
eval_neg = eval_neg[ids.astype(int)]
eval_pos = eval_pos[ids.astype(int)]

if hmri == True:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cb = ax1.scatter(Qs, Rms, c=np.real(eval_neg), marker="s", s=40)#, vmin=-1E-7, vmax=1E-7, cmap="bone")
    fig.colorbar(cb)
    ax1.set_title("Real")
    ax1.set_xlabel("Q")
    ax1.set_ylabel("Rm")
    ax1.set_xlim(Q[0], Q[-1])
    ax1.set_ylim(Rm[0], Rm[-1])
    
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    cb2 = ax2.scatter(Qs, Rms, c=np.real(eval_pos), marker="s", s=40)
    fig.colorbar(cb2)
    ax2.set_title("Imaginary")
    ax2.set_xlabel("Q")
    ax2.set_ylabel("Rm")
    ax2.set_xlim(Q[0], Q[-1])
    ax2.set_ylim(Rm[0], Rm[-1])
    

if hmri == False:

    e = eval[ids.astype(int)]
    e = -1*e

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cb = ax1.scatter(Qs, Rms, c=np.real(e), marker="s", s=40, vmin=-1E-7, vmax=1E-7, cmap="bone")
    fig.colorbar(cb)
    ax1.set_title("Real")
    ax1.set_xlabel("Q")
    ax1.set_ylabel("Rm")
    ax1.set_xlim(Q[0], Q[-1])
    ax1.set_ylim(Rm[0], Rm[-1])
    
  
    ax2 = fig.add_subplot(122)
    cb2 = ax2.scatter(Qs, Rms, c=np.imag(e), marker="s", s=40)
    fig.colorbar(cb2)
    ax2.set_title("Imaginary")
    ax2.set_xlabel("Q")
    ax2.set_ylabel("Rm")
    ax2.set_xlim(Q[0], Q[-1])
    ax2.set_ylim(Rm[0], Rm[-1])
    

    plt.show()

def test():
    A = np.arange(4, 8, 1)
    B = np.arange(2, 5, 1)

    def run_solver(A, B):
        try:
           result = A + B
        except np.linalg.LinAlgError:
           result = np.nan
        return (result, B)

    results = []
    with Pool(processes=15) as pool:
        params = (zip(A, B))
        #print(*params)
        #for result in pool.starmap_async(run_solver, params):
        #r = dict(pool.starmap_async(run_solver, params))
        #for p, q in params:
        """
        try:
            results[p, q] = r.get(15)
        except mp.context.TimeoutError:  
            # do desired action on timeout
            print("timeouterror")
            results[p, q] = np.nan
        """
        #ids = np.arange(len(Qs))
    
        result_pool = [pool.apply_async(run_solver, args) for args in params]
    
        for result in result_pool:
            print(result)
            try:
                print("appending")
                results.append(result.get(15))
                print(results)
            except mp.context.TimeoutError:
                # do desired action on timeout
                print("timeout error")
                results.append(None)

    jj = {}    
    for x in results:
        jj[x[0]] = x[1]
    
    print(jj)
        

