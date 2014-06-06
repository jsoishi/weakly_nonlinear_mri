import numpy as np
import matplotlib.pyplot as plt
import pylab

esearch2 = np.load("esearch_2.npy")
Q2 = np.load("Qsearch_2.npy")
Rm2 = np.load("Rmsearch_2.npy")

esearch3 = np.load("esearch_3.npy")
Q3 = np.load("Qsearch_3.npy")
Rm3 = np.load("Rmsearch_3.npy")

esearch4 = np.load("esearch_4.npy")
Q4 = np.load("Qsearch_4.npy")
Rm4 = np.load("Rmsearch_4.npy")

Q2 = Q2.reshape(1, Q2.shape[0]*Q2.shape[1])
Q3 = Q3.reshape(1, Q3.shape[0]*Q3.shape[1])
Q4 = Q4.reshape(1, Q4.shape[0]*Q4.shape[1])

esearch2 = esearch2.reshape(Q2.shape)
Rm2 = Rm2.reshape(Q2.shape)

esearch3 = esearch3.reshape(Q3.shape)
Rm3 = Rm3.reshape(Q3.shape)

esearch4 = esearch4.reshape(Q4.shape)
Rm4 = Rm4.reshape(Q4.shape)

allQ = np.hstack((Q2, Q3, Q4))
allesearch = np.hstack((esearch2, esearch3, esearch4))
allRm = np.hstack((Rm2, Rm3, Rm4))

print(allQ.shape)

fig = plt.figure()
ax1 = fig.add_subplot(121)
cb = ax1.scatter(allRm, allQ, c=np.real(allesearch), marker="s", s=40, vmin=-0.08, vmax=0.08)
fig.colorbar(cb)
ax1.set_title("Real")
ax1.set_xlabel("Rm")
ax1.set_ylabel("Q")

ax2 = fig.add_subplot(122)
cb2 = ax2.scatter(allRm, allQ, c=np.imag(allesearch), marker="s", s=40)
fig.colorbar(cb2)
ax2.set_title("Imaginary")
ax2.set_xlabel("Rm")
ax2.set_ylabel("Q")