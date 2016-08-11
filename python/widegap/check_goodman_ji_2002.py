import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import h5py

import dedalus.public as de
from allorders2_widegap import OrderE

# Parameters from Goodman + Ji (2002)
Omega1 = 313.55
Omega2 = 37.9#56.43
xi = 0
R1 = 5.
R2 = 15.
h = 10.
B0 = 3000. # G
Q = np.pi/h
eta = 2000. # in cm^2/s
rho = 6. # g/cm^3
Pm = 1.6e-6

B = (Omega2 - Omega1)/(1/R2**2 - 1/R1**2)
A = Omega1 - B/R1**2
r0 = (R1 + R2)/2.
Omega0 = A + B/r0**2
Rmc = (Omega0 * r0**2)/eta #3e5*Pm
beta = (Omega0**2 * r0**2 *4*np.pi*rho)/B0**2
# solve EVP
gridnum = 256
r_basis = de.Chebyshev('r', gridnum, interval=(R1, R2))
domain = de.Domain([r_basis], np.complex128, comm=MPI.COMM_SELF)
print("running at gridnum", gridnum)

e1 = OrderE(domain, Q = Q, Rm= Rmc, Pm=Pm,beta=beta,Omega1=Omega1, Omega2=Omega2)

# construct primative variables from streamfunction/vector potential
r_g = domain.grid(0)

u = domain.new_field()
w = domain.new_field()
Br = domain.new_field()
Bz = domain.new_field()

u['g'] = (e1.psi['g']*1j*Q - e1.psi['g'].conj()*1j*Q)/r_g
w['g'] = -(e1.psi_r['g'] + e1.psi_r['g'].conj())/r_g

Br['g'] = (e1.A['g']*1j*Q - e1.A['g'].conj()*1j*Q)/r_g
Bz['g'] = -(e1.A_r['g'] + e1.A_r['g'].conj())/r_g

Bphi = e1.B
v = e1.u

# plot
scale_factor = 1

fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(r_g,scale_factor*v['g'],'k',ls='--')
ax.plot(r_g,scale_factor*u['g']/3.,'k',ls='dotted')
ax.plot(r_g,scale_factor*w['g']*0.07,'k',ls='-.')
ax.plot(r_g,scale_factor*Br['g'],'k')
ax.plot(r_g,scale_factor*Bphi['g']*5,'b',dashes=[2,4,8,4])
ax.plot(r_g,scale_factor*Bz['g'],'k',dashes=[4,4])
ax.set_xlabel('r', fontsize=18)
ax.set_ylabel('f', fontsize=18)
ax.axvspan(4, 5, alpha=0.5, color='red')
ax.axvspan(15, 16, alpha=0.5, color='red')

ax.set_xlim(4,16)
#ax.set_ylim(-0.4,0.4)

ax2 = fig.add_subplot(122)
ax2.plot(r_g,scale_factor*v['g'],'k',ls='--')
ax2.plot(r_g,scale_factor*u['g']/3,'k',ls='dotted')
ax2.plot(r_g,scale_factor*w['g']*0.07,'k',ls='-.')
ax2.plot(r_g,scale_factor*Br['g'],'k')
ax2.plot(r_g,scale_factor*Bphi['g']*5,'b',dashes=[2,4,8,4])
ax2.plot(r_g,scale_factor*Bz['g'],'k',dashes=[4,4])
ax2.set_xlabel('r', fontsize=18)
ax2.set_ylabel('f', fontsize=18)
ax2.axvspan(4.8, 5, alpha=0.5, color='red')

ax2.set_xlim(4.8,6)
#ax2.set_ylim(-0.5,3)

fig.savefig('../../figs/goodman_ji_check.png')


