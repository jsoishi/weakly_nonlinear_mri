import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
from scipy.linalg import eig, norm
import pylab
import copy


def setup():
    x_basis = de.Chebyshev('x', 64)
    domain = de.Domain([x_basis], np.complex128)#,grid_dtype=np.float64)
    lv1 = de.EVP(domain,
                 variables=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                 eigenvalue='sigma')

    #Solve equations for fixed z eigenvalue: V+ = V+ e^(iQz)
    #Parameter values from Umurhan+:
    Pm = 0.001 #Pm = Rm/R
    q = 3/2.
    Co = 0.08

    #Parameter values found by solving the linear MRI to find most unstable mode
    Rm = 4.8775
    #Rm = 4.89
    #Rm = 4.9
    #Rm = 4.88
    #Rm = 4.877
    #Rm = 4.878
    #Rm = 4.87
    iRm = 1./Rm
    Q = 0.75

    R = Rm/Pm
    iR = 1./R
    lv1.parameters['Q'] = Q
    lv1.parameters['iR'] = iR
    lv1.parameters['iRm'] = iRm
    lv1.parameters['q'] = q
    lv1.parameters['Co'] = Co

    #Correct equations
    lv1.add_equation("-sigma*Q**2*psi + sigma*psixx + 1j*Q*A + 1j*(q - 2)*Q*u + iR*Q**4*psi - 2*iR*Q**2*psixx + iR*dx(psixxx) = 0")
    lv1.add_equation("sigma*u + 1j*Q*B + 2*1j*Q*psi - iR*Q**2*u + iR*dx(ux) = 0")
    lv1.add_equation("sigma*A - iRm*Q**2*A + iRm*dx(Ax) - 1j*Q*q*B - 1j*Co*Q**3*psi + 1j*Co*Q*psixx = 0")
    lv1.add_equation("sigma*B - iRm*Q**2*B + iRm*dx(Bx) + 1j*Co*Q*u = 0")
    
    #without iQ's
    #lv1.add_equation("1j*dt(psixx) + A + iR*dx(psixxx) + iR*2*psixx + iR*psi + (q-2)*u = 0")
    #lv1.add_equation("1j*dt(u) + B + 2*psi + iR*dx(ux) + iR*u = 0")
    #lv1.add_equation("1j*dt(A) + iRm*dx(Ax) + iRm*A - q*B + Co*dx(psix) + Co*psi = 0")
    #lv1.add_equation("1j*dt(B) + iRm*dx(Bx) + iRm*B + Co*u = 0")

    lv1.add_equation("dx(psi) - psix = 0")
    lv1.add_equation("dx(psix) - psixx = 0")
    lv1.add_equation("dx(psixx) - psixxx = 0")
    lv1.add_equation("dx(u) - ux = 0")
    lv1.add_equation("dx(A) - Ax = 0")
    lv1.add_equation("dx(B) - Bx = 0")

    #Boundary conditions
    lv1.add_bc("left(u) = 0")
    lv1.add_bc("right(u) = 0")
    lv1.add_bc("left(psi) = 0")
    lv1.add_bc("right(psi) = 0")
    lv1.add_bc("left(A) = 0")
    lv1.add_bc("right(A) = 0")
    lv1.add_bc("left(psix) = 0")
    lv1.add_bc("right(psix) = 0")
    lv1.add_bc("left(Bx) = 0")
    lv1.add_bc("right(Bx) = 0")

    #Parameters
    lv1.parameters['Q'] = Q
    lv1.parameters['iR'] = iR
    lv1.parameters['iRm'] = iRm
    lv1.parameters['q'] = q
    lv1.parameters['Co'] = Co

    LEV = lv1.build_solver()
    LEV.solve(LEV.pencils[0])

    #Find the eigenvalue that is closest to zero. This should be the adjoint homogenous solution.
    evals = LEV.eigenvalues
    indx = np.arange(len(evals))
    #e0 = indx[np.abs(evals.imag) == np.nanmin(np.abs(evals.imag))]
    e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
    print(evals[e0])
    
    """
    #second closest to zero...
    indx = np.delete(indx, e0)
    evals = np.delete(evals, e0)
    e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
    print(evals[e0])
    
    #third closest to zero...
    indx = np.delete(indx, e0)
    evals = np.delete(evals, e0)
    e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
    print(evals[e0])
    """

    #Plot
    x = domain.grid(0)
    LEV.set_state(e0[0])

    #L = LEV.eigenvalue_pencil.L.todense()
    #M = LEV.eigenvalue_pencil.M.todense()
    #b = np.zeros_like(LEV.eigenvectors[0])
    #lsolve = np.linalg.solve(L, b)

    return x, LEV


#Plotting --------------------------------------------

def plot_all(x, LEV):

    #Currently normalized so that ee = 1
    #ee = np.abs(np.real(LEV.eigenvectors)) + np.abs(np.imag(LEV.eigenvectors))

    #Normalized to resemble Umurhan+
    norm1 = -0.9/np.min(LEV.state['u']['g'].imag) #norm(LEV.eigenvectors)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(x, LEV.state['psi']['g'].imag*norm1, color="red")
    ax1.plot(x, LEV.state['psi']['g'].real*norm1, color="black")
    
    #psi_cc = LEV.state['psi']['g'] + LEV.state['psi']['g'].conj()
    #ax1.plot(x, psi_cc.imag, color="red")
    #ax1.plot(x, psi_cc.real, color="black")
    
    ax1.set_title(r"Im($\psi^\dagger$)")

    ax2 = fig.add_subplot(222)
    ax2.plot(x, LEV.state['u']['g'].imag*norm1, color="black")
    ax2.plot(x, LEV.state['u']['g'].real*norm1, color="red")
    
    #u_cc = LEV.state['u']['g'] + LEV.state['u']['g'].conj()
    #ax2.plot(x, u_cc.imag, color="black")
    #ax2.plot(x, u_cc.real, color="red")
    
    ax2.set_title("Re($u^\dagger$)")

    ax3 = fig.add_subplot(223)
    ax3.plot(x, LEV.state['A']['g'].imag*norm1, color="black")
    ax3.plot(x, LEV.state['A']['g'].real*norm1, color="red")
    
    #A_cc = LEV.state['A']['g'] + LEV.state['A']['g'].conj()
    #ax3.plot(x, A_cc.imag, color="black")
    #ax3.plot(x, A_cc.real, color="red")
    
    ax3.set_title("Re($A^\dagger$)")

    ax4 = fig.add_subplot(224)
    ax4.plot(x, LEV.state['B']['g'].imag*norm1, color="red")
    ax4.plot(x, LEV.state['B']['g'].real*norm1, color="black")
    
    #B_cc = LEV.state['B']['g'] + LEV.state['B']['g'].conj()
    #ax4.plot(x, B_cc.imag, color="red")
    #ax4.plot(x, B_cc.real, color="black")
    
    ax4.set_title("Im($B^\dagger$)")
    fig.savefig("ah1.png")

if __name__ == '__main__':
    x, LEV = setup()
    plot_all(x, LEV)


"""
for i in range(len(evals)):
    LEV.set_state(i)

    ax1 = fig.add_subplot(221)
    ax1.plot(x, LEV.state['psi']['g'].imag)
    ax1.set_title(r"$\psi$")
    ax1 = fig.add_subplot(222)
    ax1.plot(x, LEV.state['u']['g'].real)
    ax1.set_title("u")
    ax1 = fig.add_subplot(223)
    ax1.plot(x, LEV.state['A']['g'].real)
    ax1.set_title("A")
    ax1 = fig.add_subplot(224)
    ax1.plot(x, LEV.state['B']['g'].imag)
    ax1.set_title("B")

    pylab.savefig('AH_'+ '%03d' % i +'.png')

    fig.clf()
"""
