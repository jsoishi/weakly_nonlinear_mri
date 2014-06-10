import numpy as np
import matplotlib.pyplot as plt
import pylab
import pickle

testdata = pickle.load(open("results_test.p", "rb"))

Rms = np.zeros(len(testdata))
Qs = np.zeros(len(testdata))
eval = np.zeros(len(testdata), np.complex128)

for i in range(len(testdata)):
    jj = testdata.popitem()
    RmQ = jj[0]
    Rms[i] = RmQ[0]
    Qs[i] = RmQ[1]
    eval[i] = jj[1]
    
eval = -1*eval

fig = plt.figure()
ax1 = fig.add_subplot(121)
cb = ax1.scatter(Qs, Rms, c=np.real(eval), marker="s", s=40, vmin=-1E-7, vmax=1E-7, cmap="bone")
fig.colorbar(cb)
ax1.set_title("Real")
ax1.set_xlabel("Q")
ax1.set_ylabel("Rm")

ax2 = fig.add_subplot(122)
cb2 = ax2.scatter(Qs, Rms, c=np.imag(eval), marker="s", s=40)
fig.colorbar(cb2)
ax2.set_title("Imaginary")
ax2.set_xlabel("Q")
ax2.set_ylabel("Rm")