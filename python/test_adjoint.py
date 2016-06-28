import matplotlib.pyplot as plt

from allorders_2 import AdjointHomogenous

AH = AdjointHomogenous()

scale = (0.18+3/11.*0.02)/AH.B['g'][0].imag

x = AH.EP.solver.domain.grid(0)

plt.subplot(141)
plt.plot(x,scale*AH.psi['g'].imag)
plt.xlabel('x')
plt.ylabel('Imag(psi)')

plt.ylim(-0.15,0.05)

plt.subplot(142)
plt.plot(x,scale*AH.u['g'].real)
plt.xlabel('x')
plt.ylabel('Real(u)')

plt.subplot(143)
plt.plot(x,scale*AH.A['g'].real)
plt.xlabel('x')
plt.ylabel('Real(A)')

plt.subplot(144)
plt.plot(x,scale*AH.B['g'].imag)
plt.ylim(0.18,0.24)
plt.xlabel('x')
plt.ylabel('Imag(B)')

plt.savefig('ah_pm1e-3.png')
