import numpy as np
import matplotlib.pyplot as plt
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue, LinearBVP
from scipy.linalg import eig, norm
import pylab
import copy
import pickle

import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
matplotlib.rcParams.update({'figure.autolayout': True})

gridnum = 64
x_basis = Chebyshev(gridnum)
domain = Domain([x_basis], grid_dtype=np.complex128)

class AdjointHomogenous():

    """
    Solves the adjoint homogenous equation L^dagger V^dagger = 0
    Returns V^dagger
    
    """
    
    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0):
        
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.q = q
        self.beta = beta
        
    
    def solve(self, gridnum = gridnum, save = False, norm = True):

        self.gridnum = gridnum

        lv1 = ParsedProblem(['x'],
                              field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                              param_names=['Q', 'iR', 'iRm', 'q', 'beta'])

        # inverse magnetic reynolds number
        iRm = 1./self.Rm

        # rayleigh number defined from prandtl number
        R = self.Rm/self.Pm
        iR = 1./R
        
        beta = self.beta
        Q = self.Q
        q = self.q

        # equations to solve
        lv1.add_equation("-1j*Q**2*dt(psi) + 1j*dt(psixx) + 1j*Q*A + 1j*(q - 2)*Q*u + iR*Q**4*psi - 2*iR*Q**2*psixx + iR*dx(psixxx) = 0")
        lv1.add_equation("1j*dt(u) + 1j*Q*B + 2*1j*Q*psi - iR*Q**2*u + iR*dx(ux) = 0")
        lv1.add_equation("1j*dt(A) - iRm*Q**2*A + iRm*dx(Ax) - 1j*Q*q*B - 1j*(2/beta)*Q**3*psi + 1j*(2/beta)*Q*psixx = 0")
        lv1.add_equation("1j*dt(B) - iRm*Q**2*B + iRm*dx(Bx) + 1j*(2/beta)*Q*u = 0")

        lv1.add_equation("dx(psi) - psix = 0")
        lv1.add_equation("dx(psix) - psixx = 0")
        lv1.add_equation("dx(psixx) - psixxx = 0")
        lv1.add_equation("dx(u) - ux = 0")
        lv1.add_equation("dx(A) - Ax = 0")
        lv1.add_equation("dx(B) - Bx = 0")

        # boundary conditions
        lv1.add_left_bc("u = 0")
        lv1.add_right_bc("u = 0")
        lv1.add_left_bc("psi = 0")
        lv1.add_right_bc("psi = 0")
        lv1.add_left_bc("A = 0")
        lv1.add_right_bc("A = 0")
        lv1.add_left_bc("psix = 0")
        lv1.add_right_bc("psix = 0")
        lv1.add_left_bc("Bx = 0")
        lv1.add_right_bc("Bx = 0")

        # parameters
        lv1.parameters['Q'] = self.Q
        lv1.parameters['iR'] = iR
        lv1.parameters['iRm'] = iRm
        lv1.parameters['q'] = self.q
        lv1.parameters['beta'] = self.beta

        lv1.expand(domain)
        LEV = LinearEigenvalue(lv1,domain)
        LEV.solve(LEV.pencils[0])
        
        self.LEV = LEV

        # the eigenvalue that is closest to zero is the adjoint homogenous solution.
        evals = LEV.eigenvalues
        indx = np.arange(len(evals))
        e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
        print('eigenvalue', evals[e0])
       
        # set state
        self.x = domain.grid(0)
        LEV.set_state(e0[0])
        
        self.psi = LEV.state['psi']
        self.u = LEV.state['u']
        self.A = LEV.state['A']
        self.B = LEV.state['B']
        
        if norm == True:
            n = np.abs(self.LEV.state['psi']['g'])[13]
            a = self.LEV.state['psi']['g'].real[13]/n
            b = self.LEV.state['psi']['g'].imag[13]/n
            scale = 1j*a/(b*(a**2/b+b)) + 1./(a**2/b +b)
            #scale *= -664.4114817
            
            psinorm = LEV.state['psi']*scale
            self.psi = psinorm.evaluate()
            unorm = LEV.state['u']*scale
            self.u = unorm.evaluate()
            Anorm = LEV.state['A']*scale
            self.A = Anorm.evaluate()
            Bnorm = LEV.state['B']*scale
            self.B = Bnorm.evaluate()
            
            self.scale = scale
        

        if save == True:
            pickle.dump(psi, open("V_adj_psi.p", "wb"), protocol=-1)
            pickle.dump(LEV.state['u'], open("V_adj_u.p", "wb"))
            pickle.dump(LEV.state['A'], open("V_adj_A.p", "wb"))
            pickle.dump(LEV.state['B'], open("V_adj_B.p", "wb"))
            


    def plot(self):
        
        #Normalized to resemble Umurhan+
        #norm1 = -0.9/np.min(self.u.imag) #norm(LEV.eigenvectors)
        norm1 = 1
    
        fig = plt.figure()
    
        ax1 = fig.add_subplot(221)
        ax1.plot(self.x, self.psi['g'].imag*norm1, color="black")
        ax1.plot(self.x, self.psi['g'].real*norm1, color="red")
        ax1.set_title(r"Im($\psi^\dagger$)")

        ax2 = fig.add_subplot(222)
        ax2.plot(self.x, self.u['g'].imag*norm1, color="red")
        ax2.plot(self.x, self.u['g'].real*norm1, color="black")
        ax2.set_title("Re($u^\dagger$)")

        ax3 = fig.add_subplot(223)
        ax3.plot(self.x, self.A['g'].imag*norm1, color="red")
        ax3.plot(self.x, self.A['g'].real*norm1, color="black")
        ax3.set_title("Re($A^\dagger$)")

        ax4 = fig.add_subplot(224)
        ax4.plot(self.x, self.B['g'].imag*norm1, color="black")
        ax4.plot(self.x, self.B['g'].real*norm1, color="red")
        ax4.set_title("Im($B^\dagger$)")
        fig.savefig("ah1.png")
        
        
        
class OrderE():

    """
    Solves the order(epsilon) equation L V_1 = 0
    This is simply the linearized MRI.
    Returns V_1
    
    """
    
    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0):
        
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.q = q
        self.beta = beta
        
    
    def solve(self, gridnum = gridnum, save = False, norm = True):

        self.gridnum = gridnum


        lv1 = ParsedProblem(['x'],
                              field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                              param_names=['Q', 'iR', 'iRm', 'q', 'beta'])
                  

        # inverse magnetic reynolds number
        iRm = 1./self.Rm

        # rayleigh number defined from prandtl number
        R = self.Rm/self.Pm
        iR = 1./R
        
        beta = self.beta
        Q = self.Q
        q = self.q

        # linear MRI equations
        
        # bvp way
        #lv1.add_equation("-iR*dx(psixxx) + 2*iR*Q**2*psixx - iR*Q**4*psi - 2*1j*Q*u - (2/beta)*1j*Q*dx(Ax) + (2/beta)*Q**3*1j*A = 0")
        #lv1.add_equation("-iR*dx(ux) + iR*Q**2*u + (2-q)*1j*Q*psi - (2/beta)*1j*Q*B = 0") 
        #lv1.add_equation("-iRm*dx(Ax) + iRm*Q**2*A - 1j*Q*psi = 0") 
        #lv1.add_equation("-iRm*dx(Bx) + iRm*Q**2*B - 1j*Q*u + q*1j*Q*A = 0")
        
        lv1.add_equation("1j*dt(psixx) - 1j*Q**2*dt(psi) - iR*dx(psixxx) + 2*iR*Q**2*psixx - iR*Q**4*psi - 2*1j*Q*u - (2/beta)*1j*Q*dx(Ax) + (2/beta)*Q**3*1j*A = 0")
        lv1.add_equation("1j*dt(u) - iR*dx(ux) + iR*Q**2*u + (2-q)*1j*Q*psi - (2/beta)*1j*Q*B = 0") 
        lv1.add_equation("1j*dt(A) - iRm*dx(Ax) + iRm*Q**2*A - 1j*Q*psi = 0") 
        lv1.add_equation("1j*dt(B) - iRm*dx(Bx) + iRm*Q**2*B - 1j*Q*u + q*1j*Q*A = 0")
        
        lv1.add_equation("dx(psi) - psix = 0")
        lv1.add_equation("dx(psix) - psixx = 0")
        lv1.add_equation("dx(psixx) - psixxx = 0")
        lv1.add_equation("dx(u) - ux = 0")
        lv1.add_equation("dx(A) - Ax = 0")
        lv1.add_equation("dx(B) - Bx = 0")

        # boundary conditions
        lv1.add_left_bc("u = 0")
        lv1.add_right_bc("u = 0")
        lv1.add_left_bc("psi = 0")
        lv1.add_right_bc("psi = 0")
        lv1.add_left_bc("A = 0")
        lv1.add_right_bc("A = 0")
        lv1.add_left_bc("psix = 0")
        lv1.add_right_bc("psix = 0")
        lv1.add_left_bc("Bx = 0")
        lv1.add_right_bc("Bx = 0")

        # parameters
        lv1.parameters['Q'] = Q
        lv1.parameters['iR'] = iR
        lv1.parameters['iRm'] = iRm
        lv1.parameters['q'] = q
        lv1.parameters['beta'] = beta

        lv1.expand(domain, order = gridnum)
        LEV = LinearEigenvalue(lv1, domain)
        LEV.solve(LEV.pencils[0])
        #LEV = LinearBVP(lv1, domain)
        #LEV.solvedense()
        
        self.LEV = LEV

        #Find the eigenvalue that is closest to zero.
        evals = LEV.eigenvalues
        indx = np.arange(len(evals))
        e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
        print('eigenvalue', evals[e0])
       
        # set state
        self.x = domain.grid(0)
        LEV.set_state(e0[0])
        
        self.psi = LEV.state['psi']['g']
        self.u = LEV.state['u']['g']
        self.A = LEV.state['A']['g']
        self.B = LEV.state['B']['g']
        
        if norm == True:
            n = np.abs(self.LEV.state['psi']['g'])[13]
            a = self.LEV.state['psi']['g'].real[13]/n
            b = self.LEV.state['psi']['g'].imag[13]/n
            scale = 1j*a/(b*(a**2/b+b)) + 1./(a**2/b +b)
            scale *= -664.4114817
            
            self.psi = LEV.state['psi']['g']*scale
            self.u = LEV.state['u']['g']*scale
            self.A = LEV.state['A']['g']*scale
            self.B = LEV.state['B']['g']*scale
            
            self.scale = scale

        if save == True:
            pickle.dump(LEV.state['psi'], open("V_1_psi.p", "wb"))
            pickle.dump(LEV.state['u'], open("V_1_u.p", "wb"))
            pickle.dump(LEV.state['A'], open("V_1_A.p", "wb"))
            pickle.dump(LEV.state['B'], open("V_1_B.p", "wb"))
            
        

    def plot(self):
    
        fig = plt.figure()
        
        ax1 = fig.add_subplot(221)
        ax1.plot(self.x, self.psi.imag, color="black")
        ax1.plot(self.x, self.psi.real, color="red")
        ax1.set_title(r"Im($\psi_{1}$)")

        ax2 = fig.add_subplot(222)
        ax2.plot(self.x, self.u.real, color="black")
        ax2.plot(self.x, self.u.imag, color="red")
        ax2.set_title("Re($u_{1}$)")

        ax3 = fig.add_subplot(223)
        ax3.plot(self.x, self.A.real, color="black")
        ax3.plot(self.x, self.A.imag, color="red")
        ax3.set_title("Re($A_{1}$)")

        ax4 = fig.add_subplot(224)
        ax4.plot(self.x, self.B.imag, color="black")
        ax4.plot(self.x, self.B.real, color="red")
        ax4.set_title("Im($B_{1}$)")
        fig.savefig("v1.png")

class N2():

    """
    Solves the nonlinear term N2
    Returns N2
    
    """
    
    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, run = True, norm = True):
        
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.q = q
        self.beta = beta
        
        # either run or load the data in from saved files (needs implementing)
        if run == True:
            v1 = OrderE()
            v1.solve(save=False)
            
            if norm == True:
            
                psi_1 = v1.LEV.state['psi']*v1.scale
                u_1 = v1.LEV.state['u']*v1.scale
                A_1 = v1.LEV.state['A']*v1.scale
                B_1 = v1.LEV.state['B']*v1.scale
            
                self.psi_1 = psi_1.evaluate()
                self.u_1 = u_1.evaluate()
                self.A_1 = A_1.evaluate()
                self.B_1 = B_1.evaluate()
                
                
            else:
            
                self.psi_1 = v1.LEV.state['psi']
                self.u_1 = v1.LEV.state['u']
                self.A_1 = v1.LEV.state['A']
                self.B_1 = v1.LEV.state['B']
        
    
    def solve(self, gridnum = gridnum, save = True, norm = True):

        self.gridnum = gridnum
        self.x = domain.grid(0)
        
        # inverse magnetic reynolds number
        iRm = 1./self.Rm

        # rayleigh number defined from prandtl number
        R = self.Rm/self.Pm
        iR = 1./R
        
        beta = self.beta
        Q = self.Q
        q = self.q
        
        self.psi_1_x = self.psi_1.differentiate(0)
        self.psi_1_xx = self.psi_1_x.differentiate(0)
        self.psi_1_xxx = self.psi_1_xx.differentiate(0)
        
        self.u_1_x = self.u_1.differentiate(0)
        
        self.A_1_x = self.A_1.differentiate(0)
        self.A_1_xx = self.A_1_x.differentiate(0)
        self.A_1_xxx = self.A_1_xx.differentiate(0)
        
        self.B_1_x = self.B_1.differentiate(0)
        
        # complex conjugates 
        self.psi_1_star = domain.new_field()
        self.psi_1_star.name = 'psi_1_star'
        self.psi_1_star['g'] = self.psi_1['g'].conj()
        
        self.psi_1_star_x = self.psi_1_star.differentiate(0)
        self.psi_1_star_xx = self.psi_1_star_x.differentiate(0)
        self.psi_1_star_xxx = self.psi_1_star_xx.differentiate(0)
        
        self.u_1_star = domain.new_field()
        self.u_1_star.name = 'u_1_star'
        self.u_1_star['g'] = self.u_1['g'].conj()
        
        self.u_1_star_x = self.u_1_star.differentiate(0)
   
        self.A_1_star = domain.new_field()
        self.A_1_star.name = 'A_1_star'
        self.A_1_star['g'] = self.A_1['g'].conj()
        
        self.A_1_star_x = self.A_1_star.differentiate(0)
        self.A_1_star_xx = self.A_1_star_x.differentiate(0)
        self.A_1_star_xxx = self.A_1_star_xx.differentiate(0)
        
        self.B_1_star = domain.new_field()
        self.B_1_star.name = 'B_1_star'
        self.B_1_star['g'] = self.B_1['g'].conj()
        
        self.B_1_star_x = self.B_1_star.differentiate(0)
        
        # define nonlinear terms N22 and N20
        N22psi = 1j*Q*self.psi_1*(self.psi_1_xxx - Q**2*self.psi_1_x) - self.psi_1_x*(1j*Q*self.psi_1_xx - 1j*Q**3*self.psi_1) + (2/self.beta)*self.A_1_x*(1j*Q*self.A_1_xx - 1j*Q**3*self.A_1) - (2/self.beta)*1j*Q*self.A_1*(self.A_1_xxx - Q**2*self.A_1_x)
        self.N22_psi = N22psi.evaluate()
        
        N20psi = 1j*Q*self.psi_1*(self.psi_1_star_xxx - Q**2*self.psi_1_star_x) - self.psi_1_x*(-1j*Q*self.psi_1_star_xx + 1j*Q**3*self.psi_1_star) + (2/self.beta)*self.A_1_x*(-1j*Q*self.A_1_star_xx + 1j*Q**3*self.A_1_star) - (2/self.beta)*1j*Q*self.A_1*(self.A_1_star_xxx - Q**2*self.A_1_star_x)
        self.N20_psi = N20psi.evaluate()
        
        N22u = 1j*Q*self.psi_1*self.u_1_x - self.psi_1_x*1j*Q*self.u_1 - (2/self.beta)*1j*Q*self.A_1*self.B_1_x + (2/self.beta)*self.A_1_x*1j*Q*self.B_1
        self.N22_u = N22u.evaluate()
        
        N20u = 1j*Q*self.psi_1*self.u_1_star_x + self.psi_1_x*1j*Q*self.u_1_star - (2/self.beta)*1j*Q*self.A_1*self.B_1_star_x - (2/self.beta)*self.A_1_x*1j*Q*self.B_1_star
        self.N20_u = N20u.evaluate()
        
        N22A = -1j*Q*self.A_1*self.psi_1_x + self.A_1_x*1j*Q*self.psi_1
        self.N22_A = N22A.evaluate()
        
        N20A = -1j*Q*self.A_1*self.psi_1_star_x - self.A_1_x*1j*Q*self.psi_1_star
        self.N20_A = N20A.evaluate()
        
        N22B = 1j*Q*self.psi_1*self.B_1_x - self.psi_1_x*1j*Q*self.B_1 - 1j*Q*self.A_1*self.u_1_x + self.A_1_x*1j*Q*self.u_1
        self.N22_B = N22B.evaluate()
        
        N20B = 1j*Q*self.psi_1*self.B_1_star_x + self.psi_1_x*1j*Q*self.B_1_star - 1j*Q*self.A_1*self.u_1_star_x - self.A_1_x*1j*Q*self.u_1_star
        self.N20_B = N20B.evaluate()
        
        if norm == True:
        
            self.n20scale = 0.2/0.087941764519131521
            
            N20psi = self.N20_psi*self.n20scale
            self.N20_psi = N20psi.evaluate()
            
            N20u = self.N20_u*self.n20scale
            self.N20_u = N20u.evaluate()
            
            N20A = self.N20_A*self.n20scale
            self.N20_A = N20A.evaluate()
            
            N20B = self.N20_B*self.n20scale
            self.N20_B = N20B.evaluate()
        
    def plot(self):
    
        fig = plt.figure()
        
        ax1 = fig.add_subplot(241)
        ax1.plot(self.x, self.N20_psi['g'].imag, color="black")
        ax1.plot(self.x, self.N20_psi['g'].real, color="red")
        ax1.set_title(r"Im($\psi_{N20}$)")

        ax2 = fig.add_subplot(242)
        ax2.plot(self.x, self.N20_u['g'].real, color="black")
        ax2.plot(self.x, self.N20_u['g'].imag, color="red")
        ax2.set_title("Re($u_{N20}$)")

        ax3 = fig.add_subplot(243)
        ax3.plot(self.x, self.N20_A['g'].real, color="black")
        ax3.plot(self.x, self.N20_A['g'].imag, color="red")
        ax3.set_title("Re($A_{N20}$)")

        ax4 = fig.add_subplot(244)
        ax4.plot(self.x, self.N20_B['g'].imag, color="black")
        ax4.plot(self.x, self.N20_B['g'].real, color="red")
        ax4.set_title("Im($B_{N20}$)")
        
        ax1 = fig.add_subplot(245)
        ax1.plot(self.x, self.N22_psi['g'].imag, color="black")
        ax1.plot(self.x, self.N22_psi['g'].real, color="red")
        ax1.set_title(r"Im($\psi_{N22}$)")

        ax2 = fig.add_subplot(246)
        ax2.plot(self.x, self.N22_u['g'].real, color="black")
        ax2.plot(self.x, self.N22_u['g'].imag, color="red")
        ax2.set_title("Re($u_{N22}$)")

        ax3 = fig.add_subplot(247)
        ax3.plot(self.x, self.N22_A['g'].real, color="black")
        ax3.plot(self.x, self.N22_A['g'].imag, color="red")
        ax3.set_title("Re($A_{N22}$)")

        ax4 = fig.add_subplot(248)
        ax4.plot(self.x, self.N22_B['g'].imag, color="black")
        ax4.plot(self.x, self.N22_B['g'].real, color="red")
        ax4.set_title("Im($B_{N22}$)")
        fig.savefig("n2.png")

        
class OrderE2():

    """
    Solves the second order equation L V2 = N2 - Ltwiddle V1
    Returns V2
    
    """
    
    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, run = True, norm = True):
        
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.q = q
        self.beta = beta
        
        if run == True:
            v1 = OrderE()
            v1.solve(save=False)
            
            if norm == True:
            
                psi_1 = v1.LEV.state['psi']*v1.scale
                u_1 = v1.LEV.state['u']*v1.scale
                A_1 = v1.LEV.state['A']*v1.scale
                B_1 = v1.LEV.state['B']*v1.scale
            
                self.psi_1 = psi_1.evaluate()
                self.u_1 = u_1.evaluate()
                self.A_1 = A_1.evaluate()
                self.B_1 = B_1.evaluate()
                
                
            else:
            
                self.psi_1 = v1.LEV.state['psi']
                self.u_1 = v1.LEV.state['u']
                self.A_1 = v1.LEV.state['A']
                self.B_1 = v1.LEV.state['B']
            
            n2 = N2()
            n2.solve()
            self.n22_psi = n2.N22_psi
            self.n22_u = n2.N22_u
            self.n22_A = n2.N22_A
            self.n22_B = n2.N22_B
            
            self.n20_psi = n2.N20_psi
            self.n20_u = n2.N20_u
            self.n20_A = n2.N20_A
            self.n20_B = n2.N20_B
            
    def solve20(self, gridnum = gridnum, save = False):
        # inverse magnetic reynolds number
        iRm = 1./self.Rm

        # rayleigh number defined from prandtl number
        R = self.Rm/self.Pm
        iR = 1./R
        
        beta = self.beta
        Q = self.Q
        q = self.q
        
        # righthand side for the 20 terms (e^0)
        rhs20_psi = self.n20_psi['g'] #+ self.n20_psi['g'].conj()
        rhs20_u = self.n20_u['g'] #+ self.n20_u['g'].conj()
        rhs20_A = self.n20_A['g'] #+ self.n20_A['g'].conj()
        rhs20_B = self.n20_B['g'] #+ self.n20_B['g'].conj()
        
        lv20psi = ParsedProblem(['x'],
                              field_names=['psi20', 'psi20x', 'psi20xx', 'psi20xxx'],
                              param_names=['iR', 'rhs20_psi'])
        lv20psi.add_equation("iR*dx(psi20xxx) = rhs20_psi")
        lv20psi.add_equation("dx(psi20) - psi20x = 0")
        lv20psi.add_equation("dx(psi20x) - psi20xx = 0")
        lv20psi.add_equation("dx(psi20xx) - psi20xxx = 0")
        lv20psi.parameters['iR'] = iR
        lv20psi.parameters['rhs20_psi'] = rhs20_psi
        lv20psi.add_left_bc("psi20 = 0")
        lv20psi.add_right_bc("psi20 = 0")
        lv20psi.add_left_bc("psi20x = 0")
        lv20psi.add_right_bc("psi20x = 0")
        
        lv20u = ParsedProblem(['x'],
                              field_names=['u20', 'u20x'],
                              param_names=['iR', 'rhs20_u'])
        lv20u.add_equation("iR*dx(u20x) = rhs20_u")
        lv20u.add_equation("dx(u20) - u20x = 0")
        lv20u.parameters['iR'] = iR
        lv20u.parameters['rhs20_u'] = rhs20_u
        lv20u.add_left_bc("u20 = 0")
        lv20u.add_right_bc("u20 = 0")
        
        lv20A = ParsedProblem(['x'],
                              field_names=['A20', 'A20x'],
                              param_names=['iRm', 'rhs20_A'])
        lv20A.add_equation("iRm*dx(A20x) = rhs20_A")
        lv20A.add_equation("dx(A20) - A20x = 0")
        lv20A.parameters['iRm'] = iRm
        lv20A.parameters['rhs20_A'] = rhs20_A
        lv20A.add_left_bc("A20 = 0")
        lv20A.add_right_bc("A20 = 0")
        
        lv20B = ParsedProblem(['x'],
                              field_names=['B20', 'B20x'],
                              param_names=['iRm', 'rhs20_B'])
        lv20B.add_equation("iRm*dx(B20x) = rhs20_B")
        lv20B.add_equation("dx(B20) - B20x = 0")
        lv20B.parameters['iRm'] = iRm
        lv20B.parameters['rhs20_B'] = rhs20_B
        lv20B.add_left_bc("B20x = 0")
        lv20B.add_right_bc("B20x = 0")
        
        lv20psi.expand(domain, order = gridnum)
        lv20u.expand(domain, order = gridnum)
        lv20A.expand(domain, order = gridnum)
        lv20B.expand(domain, order = gridnum)
        
        LEV20psi = LinearBVP(lv20psi, domain)
        LEV20psi.solve()
        
        LEV20u = LinearBVP(lv20u, domain)
        LEV20u.solve()
        
        LEV20A = LinearBVP(lv20A, domain)
        LEV20A.solve()
        
        LEV20B = LinearBVP(lv20B, domain)
        LEV20B.solve()
        
        self.lv20psi = lv20psi
        self.LEV20psi = LEV20psi
        
        self.lv20u = lv20u
        self.LEV20u = LEV20u
        
        self.lv20A = lv20A
        self.LEV20A = LEV20A
        
        self.lv20B = lv20B
        self.LEV20B = LEV20B

        # set state
        self.x = domain.grid(0)
        
        self.psi20 = LEV20psi.state['psi20']['g']
        self.u20 = LEV20u.state['u20']['g'] #+ beta*R*(self.x**2 - 1) #+ LEV20utwiddle.state['u20twiddle']['g']
        self.A20 = LEV20A.state['A20']['g']
        self.B20 = LEV20B.state['B20']['g']
            
    def solve21(self, gridnum = gridnum, save = True, norm = True):
    
        # inverse magnetic reynolds number
        iRm = 1./self.Rm

        # rayleigh number defined from prandtl number
        R = self.Rm/self.Pm
        iR = 1./R
        
        beta = self.beta
        Q = self.Q
        q = self.q
        
        # derivatives
        self.psi_1_x = self.psi_1.differentiate(0)
        self.psi_1_xx = self.psi_1_x.differentiate(0)
        
        self.A_1_x = self.A_1.differentiate(0)
        self.A_1_xx = self.A_1_x.differentiate(0)
        
        # second term: L1twiddle V1
        term2_psi = -3*(2/self.beta)*Q**2*self.A_1 + (2/self.beta)*self.A_1_xx - 4*iR*1j*Q**3*self.psi_1 + 4*iR*1j*Q*self.psi_1_xx + 2*self.u_1
        self.term2_psi = term2_psi.evaluate()
        
        term2_u = (2/self.beta)*self.B_1 + 2*iR*Q*self.u_1 + (q - 2)*self.psi_1
        self.term2_u = term2_u.evaluate()
        
        term2_A = 2*iRm*1j*Q*self.A_1 + self.psi_1
        self.term2_A = term2_A.evaluate()
        
        term2_B = -q*self.A_1 + 2*iRm*1j*Q*self.B_1 + self.u_1
        self.term2_B = term2_B.evaluate()
        
        # righthand side for the 21 terms (e^iQz)
        rhs21_psi = self.term2_psi['g']
        rhs21_u = self.term2_u['g']
        rhs21_A = self.term2_A['g']
        rhs21_B = self.term2_B['g']
                
        # define problem using righthand side as nonconstant coefficients
        
        lv21 = ParsedProblem(['x'],
                              field_names=['psi21', 'psi21x', 'psi21xx', 'psi21xxx', 'u21', 'u21x', 'A21', 'A21x', 'B21', 'B21x'],
                              param_names=['Q', 'iR', 'iRm', 'q', 'beta', 'rhs20_psi', 'rhs20_u', 'rhs20_A', 'rhs20_B', 'rhs21_psi', 'rhs21_u', 'rhs21_A', 'rhs21_B', 'rhs22_psi', 'rhs22_u', 'rhs22_A', 'rhs22_B'])
          
        # equations for V21
        
        #lv21.add_equation("1j*Q**2*dt(psi21) - 1j*dt(psi21xx) + 1j*(2/beta)*Q**3*A21 - 1j*(2/beta)*Q*dx(A21x) - 2*1j*Q*u21 - iR*Q**4*psi21 + 2*iR*Q**2*psi21xx - iR*dx(psi21xxx) = rhs21_psi")
        #lv21.add_equation("-1j*dt(u21) + -1j*(2/beta)*Q*B21 - 1j*Q*(q - 2)*psi21 + iR*Q**2*u21 - iR*dx(u21x) = rhs21_u")
        #lv21.add_equation("-1j*dt(A21) + iRm*Q**2*A21 - iRm*dx(A21x) - 1j*Q*psi21 = rhs21_A")
        #lv21.add_equation("-1j*dt(B21) + 1j*Q*q*A21 + iRm*Q**2*B21 - iRm*dx(B21x) - 1j*Q*u21 = rhs21_B")
        lv21.add_equation("1j*(2/beta)*Q**3*A21 - 1j*(2/beta)*Q*dx(A21x) - 2*1j*Q*u21 - iR*Q**4*psi21 + 2*iR*Q**2*psi21xx - iR*dx(psi21xxx) = rhs21_psi")
        lv21.add_equation("-1j*(2/beta)*Q*B21 - 1j*Q*(q - 2)*psi21 + iR*Q**2*u21 - iR*dx(u21x) = rhs21_u")
        lv21.add_equation("iRm*Q**2*A21 - iRm*dx(A21x) - 1j*Q*psi21 = rhs21_A")
        lv21.add_equation("1j*Q*q*A21 + iRm*Q**2*B21 - iRm*dx(B21x) - 1j*Q*u21 = rhs21_B")

        lv21.add_equation("dx(psi21) - psi21x = 0")
        lv21.add_equation("dx(psi21x) - psi21xx = 0")
        lv21.add_equation("dx(psi21xx) - psi21xxx = 0")
        
        lv21.add_equation("dx(u21) - u21x = 0")
        
        lv21.add_equation("dx(A21) - A21x = 0")
        
        lv21.add_equation("dx(B21) - B21x = 0")

        # boundary conditions
        lv21.add_left_bc("psi21 = 0")
        lv21.add_right_bc("psi21 = 0")

        lv21.add_left_bc("u21 = 0")
        lv21.add_right_bc("u21 = 0")
       
        lv21.add_left_bc("A21 = 0")
        lv21.add_right_bc("A21 = 0")
  
        lv21.add_left_bc("psi21x = 0")
        lv21.add_right_bc("psi21x = 0")
   
        lv21.add_left_bc("B21x = 0")
        lv21.add_right_bc("B21x = 0")

        # parameters
        lv21.parameters['Q'] = Q
        lv21.parameters['iR'] = iR
        lv21.parameters['iRm'] = iRm
        lv21.parameters['q'] = q
        lv21.parameters['beta'] = beta
        lv21.parameters['rhs21_psi'] = rhs21_psi
        lv21.parameters['rhs21_u'] = rhs21_u
        lv21.parameters['rhs21_A'] = rhs21_A
        lv21.parameters['rhs21_B'] = rhs21_B

        # expand domain to gridnum points
        lv21.expand(domain, order = gridnum)
        #LEV21 = LinearEigenvalue(lv21, domain)
        #LEV21.solve(LEV21.pencils[0])
        LEV21 = LinearBVP(lv21, domain)
        LEV21.solve()
        
        self.lv21 = lv21
        self.LEV21 = LEV21

        #Find the eigenvalue that is closest to zero.
        #evals = LEV21.eigenvalues
        #indx = np.arange(len(evals))
        #e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
        #print('eigenvalue', evals[e0])
       
        # set state
        self.x = domain.grid(0)
        #LEV21.set_state(e0[0])
        
        self.psi21 = LEV21.state['psi21']['g']
        self.u21 = LEV21.state['u21']['g']
        self.A21 = LEV21.state['A21']['g']
        self.B21 = LEV21.state['B21']['g']
        
        if norm == True:
            n = np.abs(self.LEV21.state['psi21']['g'])[13]
            a = self.LEV21.state['psi21']['g'].real[13]/n
            b = self.LEV21.state['psi21']['g'].imag[13]/n
            scale = 1j*a/(b*(a**2/b+b)) + 1./(a**2/b +b)
            tt = 1j*132/29.097408658719342
            scale *= tt
            
            self.psi21 = LEV21.state['psi21']['g']*scale
            self.u21 = LEV21.state['u21']['g']*scale
            self.A21 = LEV21.state['A21']['g']*scale
            self.B21 = LEV21.state['B21']['g']*scale
            
            self.scale = scale

    def solve22(self, gridnum = gridnum, save = True):

        # inverse magnetic reynolds number
        iRm = 1./self.Rm

        # rayleigh number defined from prandtl number
        R = self.Rm/self.Pm
        iR = 1./R
        
        beta = self.beta
        Q = self.Q
        q = self.q
        
        # righthand side for the 22 terms (e^2iQz)
        rhs22_psi = self.n22_psi['g'] #+ self.n22_psi['g'].conj()
        rhs22_u = self.n22_u['g'] #+ self.n22_u['g'].conj()
        rhs22_A = self.n22_A['g'] #+ self.n22_A['g'].conj()
        rhs22_B = self.n22_B['g'] #+ self.n22_B['g'].conj()
        
        self.rhs22_psi = rhs22_psi
        self.rhs22_u = rhs22_u
        self.rhs22_A = rhs22_A
        self.rhs22_B = rhs22_B
                
        # define problem using righthand side as nonconstant coefficients
        
        lv22 = ParsedProblem(['x'],
                              field_names=['psi22', 'psi22x', 'psi22xx', 'psi22xxx', 'u22', 'u22x', 'A22', 'A22x', 'B22', 'B22x'],
                              param_names=['Q', 'iR', 'iRm', 'q', 'beta', 'rhs22_psi', 'rhs22_u', 'rhs22_A', 'rhs22_B'])
        
        lv22.add_equation("-8*1j*(2/beta)*Q**3*A22 + 2*1j*(2/beta)*Q*dx(A22x) + 4*1j*Q*u22 + 16*iR*Q**4*psi22 - 8*iR*Q**2*psi22xx + iR*dx(psi22xxx) = rhs22_psi")
        lv22.add_equation("2*1j*(2/beta)*Q*B22 + 2*1j*Q*(q-2)*psi22 - 4*iR*Q**2*u22 + iR*dx(u22x) = rhs22_u")
        lv22.add_equation("-iRm*4*Q**2*A22 + iRm*dx(A22x) + 2*1j*Q*psi22 = rhs22_A")
        lv22.add_equation("-2*1j*Q*q*A22 - iRm*4*Q**2*B22 + iRm*dx(B22x) + 2*1j*Q*u22 = rhs22_B")
        
        lv22.add_equation("dx(psi22) - psi22x = 0")
        lv22.add_equation("dx(psi22x) - psi22xx = 0")
        lv22.add_equation("dx(psi22xx) - psi22xxx = 0")
        
        lv22.add_equation("dx(u22) - u22x = 0")

        lv22.add_equation("dx(A22) - A22x = 0")

        lv22.add_equation("dx(B22) - B22x = 0")

        # boundary conditions
        lv22.add_left_bc("psi22 = 0")
        lv22.add_right_bc("psi22 = 0")
        lv22.add_left_bc("u22 = 0")
        lv22.add_right_bc("u22 = 0")
        lv22.add_left_bc("A22 = 0")
        lv22.add_right_bc("A22 = 0")
        lv22.add_left_bc("psi22x = 0")
        lv22.add_right_bc("psi22x = 0")
        lv22.add_left_bc("B22x = 0")
        lv22.add_right_bc("B22x = 0")

        # parameters
        lv22.parameters['Q'] = Q
        lv22.parameters['iR'] = iR
        lv22.parameters['iRm'] = iRm
        lv22.parameters['q'] = q
        lv22.parameters['beta'] = beta
        lv22.parameters['rhs22_psi'] = rhs22_psi
        lv22.parameters['rhs22_u'] = rhs22_u
        lv22.parameters['rhs22_A'] = rhs22_A
        lv22.parameters['rhs22_B'] = rhs22_B

        # expand domain to gridnum points
        lv22.expand(domain, order = gridnum)
        #LEV22 = LinearEigenvalue(lv22, domain)
        #LEV22.solve(LEV22.pencils[0])
        LEV22 = LinearBVP(lv22, domain)
        LEV22.solve()
        
        self.lv22 = lv22
        self.LEV22 = LEV22

        #Find the eigenvalue that is closest to zero.
        #evals = LEV22.eigenvalues
        #indx = np.arange(len(evals))
        #e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
        #print('eigenvalue', evals[e0])
       
        # set state
        self.x = domain.grid(0)
        #LEV22.set_state(e0[0])
        
        self.psi22 = LEV22.state['psi22']['g']
        self.u22 = LEV22.state['u22']['g']
        self.A22 = LEV22.state['A22']['g']
        self.B22 = LEV22.state['B22']['g']
        
        
    def plot(self):
    
        fig = plt.figure()
        
        # plot 20
        ax1 = fig.add_subplot(3, 4, 1)
        #ax1.plot(self.x, self.psi20.imag, color="black")
        #ax1.plot(self.x, self.psi20.real, color="red")
        ax1.plot(self.x, self.psi20, color="black")
        ax1.set_title(r"$\psi_{20}$")

        ax2 = fig.add_subplot(3, 4, 2)
        #ax2.plot(self.x, self.u20.real, color="black")
        #ax2.plot(self.x, self.u20.imag, color="red")
        ax2.plot(self.x, self.u20, color="black")
        ax2.set_title(r"$u_{20}$")

        ax3 = fig.add_subplot(3, 4, 3)
        #ax3.plot(self.x, self.A20.real, color="black")
        #ax3.plot(self.x, self.A20.imag, color="red")
        ax3.plot(self.x, self.A20, color="black")
        ax3.set_title(r"$A_{20}$")

        ax4 = fig.add_subplot(3, 4, 4)
        #ax4.plot(self.x, self.B20.imag, color="black")
        #ax4.plot(self.x, self.B20.real, color="red")
        ax4.plot(self.x, self.B20, color="black")
        ax4.set_title(r"$B_{20}$")
        
        # plot 21
        ax1 = fig.add_subplot(3, 4, 5)
        ax1.plot(self.x, self.psi21.real, color="black")
        ax1.plot(self.x, self.psi21.imag, color="red")
        ax1.set_title(r"$\psi_{21}$")

        ax2 = fig.add_subplot(3, 4, 6)
        ax2.plot(self.x, self.u21.imag, color="black")
        ax2.plot(self.x, self.u21.real, color="red")
        ax2.set_title(r"$u_{21}$")

        ax3 = fig.add_subplot(3, 4, 7)
        ax3.plot(self.x, self.A21.imag, color="black")
        ax3.plot(self.x, self.A21.real, color="red")
        ax3.set_title(r"$A_{21}$")

        ax4 = fig.add_subplot(3, 4, 8)
        ax4.plot(self.x, self.B21.real, color="black")
        ax4.plot(self.x, self.B21.imag, color="red")
        ax4.set_title(r"$B_{21}$")
        
        # plot 22
        ax1 = fig.add_subplot(3, 4, 9)
        ax1.plot(self.x, self.psi22.imag, color="black")
        ax1.plot(self.x, self.psi22.real, color="red")
        ax1.set_title(r"$\psi_{22}$")

        ax2 = fig.add_subplot(3, 4, 10)
        ax2.plot(self.x, self.u22.real, color="black")
        ax2.plot(self.x, self.u22.imag, color="red")
        ax2.set_title(r"$u_{22}$")

        ax3 = fig.add_subplot(3, 4, 11)
        ax3.plot(self.x, self.A22.real, color="black")
        ax3.plot(self.x, self.A22.imag, color="red")
        ax3.set_title(r"$A_{22}$")

        ax4 = fig.add_subplot(3, 4, 12)
        ax4.plot(self.x, self.B22.imag, color="black")
        ax4.plot(self.x, self.B22.real, color="red")
        ax4.set_title(r"$B_{22}$")
        
        fig.savefig("v2.png")
           
            
class N3():

    """
    Solves the nonlinear vector N3
    Returns N3
    
    """
    
    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, run = True, norm = True):
        
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.q = q
        self.beta = beta
        
        if run == True:
            v1 = OrderE()
            v1.solve(save=False)
            
            if norm == True:
            
                psi_1 = v1.LEV.state['psi']*v1.scale
                u_1 = v1.LEV.state['u']*v1.scale
                A_1 = v1.LEV.state['A']*v1.scale
                B_1 = v1.LEV.state['B']*v1.scale
            
                self.v11_psi = psi_1.evaluate()
                self.v11_u = u_1.evaluate()
                self.v11_A = A_1.evaluate()
                self.v11_B = B_1.evaluate()
                
                
            else:
            
                self.v11_psi = v1.LEV.state['psi']
                self.v11_u = v1.LEV.state['u']
                self.v11_A = v1.LEV.state['A']
                self.v11_B = v1.LEV.state['B']
            
            o2 = OrderE2()
            o2.solve20()
            o2.solve21(norm = norm)
            o2.solve22()
            
            self.v20_psi = o2.LEV20psi.state['psi20']
            self.v20_u = o2.LEV20u.state['u20']
            self.v20_A = o2.LEV20A.state['A20']
            self.v20_B = o2.LEV20B.state['B20']  
            
            if norm == True:
                v21_psi = o2.LEV21.state['psi21']*o2.scale   
                v21_u = o2.LEV21.state['u21']*o2.scale 
                v21_A = o2.LEV21.state['A21']*o2.scale 
                v21_B = o2.LEV21.state['B21']*o2.scale 
                
                self.v21_psi = v21_psi.evaluate()     
                self.v21_u = v21_u.evaluate() 
                self.v21_A = v21_A.evaluate() 
                self.v21_B = v21_B.evaluate() 
            
            else: 
                
                self.v21_psi = o2.LEV21.state['psi21']
                self.v21_u = o2.LEV21.state['u21']
                self.v21_A = o2.LEV21.state['A21']
                self.v21_B = o2.LEV21.state['B21']
                
            self.v22_psi = o2.LEV22.state['psi22']
            self.v22_u = o2.LEV22.state['u22']
            self.v22_A = o2.LEV22.state['A22']
            self.v22_B = o2.LEV22.state['B22']
            
    def solve31(self, gridnum = gridnum, save = False):
    
        self.gridnum = gridnum
        self.x = domain.grid(0)
    
        # V1 derivatives
        self.v11_psi_x = self.v11_psi.differentiate(0)
        self.v11_psi_xx = self.v11_psi_x.differentiate(0)
        self.v11_psi_xxx = self.v11_psi_xx.differentiate(0)
        
        self.v11_u_x = self.v11_u.differentiate(0)
        
        self.v11_A_x = self.v11_A.differentiate(0)
        self.v11_A_xx = self.v11_A_x.differentiate(0)
        self.v11_A_xxx = self.v11_A_xx.differentiate(0)
        
        self.v11_B_x = self.v11_B.differentiate(0)
        
        # V1 complex conjugates 
        self.v11_psi_star = domain.new_field()
        self.v11_psi_star.name = 'v11_psi_star'
        self.v11_psi_star['g'] = self.v11_psi['g'].conj()
        
        self.v11_psi_star_x = self.v11_psi_star.differentiate(0)
        self.v11_psi_star_xx = self.v11_psi_star_x.differentiate(0)
        self.v11_psi_star_xxx = self.v11_psi_star_xx.differentiate(0)
        
        self.v11_u_star = domain.new_field()
        self.v11_u_star.name = 'v11_u_star'
        self.v11_u_star['g'] = self.v11_u['g'].conj()
        
        self.v11_u_star_x = self.v11_u_star.differentiate(0)
   
        self.v11_A_star = domain.new_field()
        self.v11_A_star.name = 'v11_A_star'
        self.v11_A_star['g'] = self.v11_A['g'].conj()
        
        self.v11_A_star_x = self.v11_A_star.differentiate(0)
        self.v11_A_star_xx = self.v11_A_star_x.differentiate(0)
        self.v11_A_star_xxx = self.v11_A_star_xx.differentiate(0)
        
        self.v11_B_star = domain.new_field()
        self.v11_B_star.name = 'v11_B_star'
        self.v11_B_star['g'] = self.v11_B['g'].conj()
        
        self.v11_B_star_x = self.v11_B_star.differentiate(0)
        
        # V22 derivatives
        self.v22_psi_x = self.v22_psi.differentiate(0)
        self.v22_psi_xx = self.v22_psi_x.differentiate(0)
        self.v22_psi_xxx = self.v22_psi_xx.differentiate(0)
        
        self.v22_u_x = self.v22_u.differentiate(0)
        
        self.v22_A_x = self.v22_A.differentiate(0)
        self.v22_A_xx = self.v22_A_x.differentiate(0)
        self.v22_A_xxx = self.v22_A_xx.differentiate(0)
        
        self.v22_B_x = self.v22_B.differentiate(0)
        
        # V22 complex conjugates 
        self.v22_psi_star = domain.new_field()
        self.v22_psi_star.name = 'v22_psi_star'
        self.v22_psi_star['g'] = self.v22_psi['g'].conj()
        
        self.v22_psi_star_x = self.v22_psi_star.differentiate(0)
        self.v22_psi_star_xx = self.v22_psi_star_x.differentiate(0)
        self.v22_psi_star_xxx = self.v22_psi_star_xx.differentiate(0)
        
        self.v22_u_star = domain.new_field()
        self.v22_u_star.name = 'v22_u_star'
        self.v22_u_star['g'] = self.v22_u['g'].conj()
        
        self.v22_u_star_x = self.v22_u_star.differentiate(0)
   
        self.v22_A_star = domain.new_field()
        self.v22_A_star.name = 'v22_A_star'
        self.v22_A_star['g'] = self.v22_A['g'].conj()
        
        self.v22_A_star_x = self.v22_A_star.differentiate(0)
        self.v22_A_star_xx = self.v22_A_star_x.differentiate(0)
        self.v22_A_star_xxx = self.v22_A_star_xx.differentiate(0)
        
        self.v22_B_star = domain.new_field()
        self.v22_B_star.name = 'v22_B_star'
        self.v22_B_star['g'] = self.v22_B['g'].conj()
        
        self.v22_B_star_x = self.v22_B_star.differentiate(0)
        
        # V20 derivatives
        self.v20_psi_x = self.v20_psi.differentiate(0)
        self.v20_psi_xx = self.v20_psi_x.differentiate(0)
        self.v20_psi_xxx = self.v20_psi_xx.differentiate(0)
        
        self.v20_u_x = self.v20_u.differentiate(0)
        
        self.v20_A_x = self.v20_A.differentiate(0)
        self.v20_A_xx = self.v20_A_x.differentiate(0)
        self.v20_A_xxx = self.v20_A_xx.differentiate(0)
        
        self.v20_B_x = self.v20_B.differentiate(0)
        
        # V20 complex conjugates 
        self.v20_psi_star = domain.new_field()
        self.v20_psi_star.name = 'v20_psi_star'
        self.v20_psi_star['g'] = self.v20_psi['g'].conj()
        
        self.v20_psi_star_x = self.v20_psi_star.differentiate(0)
        self.v20_psi_star_xx = self.v20_psi_star_x.differentiate(0)
        self.v20_psi_star_xxx = self.v20_psi_star_xx.differentiate(0)
        
        self.v20_u_star = domain.new_field()
        self.v20_u_star.name = 'v20_u_star'
        self.v20_u_star['g'] = self.v20_u['g'].conj()
        
        self.v20_u_star_x = self.v20_u_star.differentiate(0)
   
        self.v20_A_star = domain.new_field()
        self.v20_A_star.name = 'v20_A_star'
        self.v20_A_star['g'] = self.v20_A['g'].conj()
        
        self.v20_A_star_x = self.v20_A_star.differentiate(0)
        self.v20_A_star_xx = self.v20_A_star_x.differentiate(0)
        self.v20_A_star_xxx = self.v20_A_star_xx.differentiate(0)
        
        self.v20_B_star = domain.new_field()
        self.v20_B_star.name = 'v20_B_star'
        self.v20_B_star['g'] = self.v20_B['g'].conj()
        
        self.v20_B_star_x = self.v20_B_star.differentiate(0)
    
        N31_psi_my1 = 1j*self.Q*(self.v11_psi*self.v20_psi_xxx) + 1j*self.Q*(self.v11_psi*self.v20_psi_star_xxx) - 1j*self.Q*(self.v11_psi_star*self.v22_psi_xxx) - 1j*2*self.Q*(self.v11_psi_star_x*self.v22_psi_xx) + 1j*8*self.Q**3*(self.v11_psi_star_x*self.v22_psi) + 1j*4*self.Q**3*(self.v11_psi_star*self.v22_psi_x)
        N31_psi_my2 = -1j*self.Q*(2/self.beta)*(self.v11_A*self.v20_A_xxx) - 1j*self.Q*(2/self.beta)*(self.v11_A*self.v20_A_star_xxx) + 1j*self.Q*(2/self.beta)*(self.v11_A_star*self.v22_A_xxx) + 1j*2*self.Q*(2/self.beta)*(self.v11_A_star_x*self.v22_A_xx) - 1j*8*self.Q**3*(2/self.beta)*(self.v11_A_star_x*self.v22_A) - 1j*4*self.Q**3*(2/self.beta)*(self.v11_A_star*self.v22_A_x)
        N31_psi_my3 = 1j*2*self.Q*(self.v22_psi*self.v11_psi_star_xxx) - 1j*2*self.Q**3*(self.v22_psi*self.v11_psi_star_x) - 1j*self.Q*(self.v20_psi_x*self.v11_psi_xx) + 1j*self.Q*(self.v22_psi_x*self.v11_psi_star_xx) - 1j*self.Q*(self.v20_psi_star_x*self.v11_psi_xx) + 1j*self.Q**3*(self.v20_psi_x*self.v11_psi) + 1j*self.Q**3*(self.v20_psi_star_x*self.v11_psi) - 1j*self.Q**3*(self.v22_psi_x*self.v11_psi_star)
        N31_psi_my4 = -1j*2*self.Q*(2/self.beta)*(self.v22_A*self.v11_A_star_xxx) + 1j*2*self.Q**3*(2/self.beta)*(self.v22_A*self.v11_A_star_x) + 1j*self.Q*(2/self.beta)*(self.v20_A_x*self.v11_A_xx) - 1j*self.Q*(2/self.beta)*(self.v22_A_x*self.v11_A_star_xx) + 1j*self.Q*(2/self.beta)*(self.v20_A_star_x*self.v11_A_xx) - 1j*self.Q**3*(2/self.beta)*(self.v20_A_x*self.v11_A) - 1j*self.Q**3*(2/self.beta)*(self.v20_A_star_x*self.v11_A) + 1j*self.Q**3*(2/self.beta)*(self.v22_A_x*self.v11_A_star)
        
        N31_psi = N31_psi_my1 + N31_psi_my2 + N31_psi_my3 +  N31_psi_my4
        
        self.N31_psi = N31_psi.evaluate()
        
        N31_u_my1 = 1j*self.Q*(self.v11_psi*self.v20_u_x) + 1j*self.Q*(self.v11_psi*self.v20_u_star_x) - 1j*self.Q*(self.v11_psi_star*self.v22_u_x) - 1j*2*self.Q*(self.v11_psi_star_x*self.v22_u)
        N31_u_my2 = -1j*self.Q*(self.v11_u*self.v20_psi_x) - 1j*self.Q*(self.v11_u*self.v20_psi_star_x) + 1j*self.Q*(self.v11_u_star*self.v22_psi_x) + 1j*2*self.Q*(self.v11_u_star_x*self.v22_psi)
        N31_u_my3 = -1j*self.Q*(2/self.beta)*(self.v11_A*self.v20_B_x) - 1j*self.Q*(2/self.beta)*(self.v11_A*self.v20_B_star_x) + 1j*self.Q*(2/self.beta)*(self.v11_A_star*self.v22_B_x) + 1j*2*self.Q*(2/self.beta)*(self.v11_A_star_x*self.v22_B)
        N31_u_my4 = 1j*self.Q*(2/self.beta)*(self.v11_B*self.v20_A_x) + 1j*self.Q*(2/self.beta)*(self.v11_B*self.v20_A_star_x) - 1j*self.Q*(2/self.beta)*(self.v11_B_star*self.v20_A_x) - 1j*2*self.Q*(2/self.beta)*(self.v11_B_star_x*self.v22_A)
        
        N31_u = N31_u_my1 + N31_u_my2 + N31_u_my3 + N31_u_my4
        
        self.N31_u = N31_u.evaluate()
        
        N31_A_my1 = -1j*self.Q*(self.v11_A*self.v20_psi_x) - 1j*self.Q*(self.v11_A*self.v20_psi_star_x) + 1j*self.Q*(self.v11_A_star*self.v22_psi_x) + 1j*2*self.Q*(self.v11_A_star_x*self.v22_psi)
        N31_A_my2 = 1j*self.Q*(self.v11_psi*self.v20_A_x) + 1j*self.Q*(self.v11_psi*self.v20_A_star_x) - 1j*self.Q*(self.v11_psi_star*self.v22_A_x) - 1j*2*self.Q*(self.v11_psi_star_x*self.v22_A)
        
        N31_A = N31_A_my1 + N31_A_my2
        
        self.N31_A = N31_A.evaluate()
        
        N31_B_my1 = 1j*self.Q*(self.v11_psi*self.v20_B_x) + 1j*self.Q*(self.v11_psi*self.v20_B_star_x) - 1j*self.Q*(self.v11_psi_star*self.v22_B_x) - 1j*2*self.Q*(self.v11_psi_star_x*self.v22_B)
        N31_B_my2 = -1j*self.Q*(self.v11_B*self.v20_psi_x) - 1j*self.Q*(self.v11_B*self.v20_psi_star_x) + 1j*self.Q*(self.v11_B_star*self.v22_psi_x) + 1j*2*self.Q*(self.v11_B_star_x*self.v22_psi)
        N31_B_my3 = -1j*self.Q*(self.v11_A*self.v20_u_x) - 1j*self.Q*(self.v11_A*self.v20_u_star_x) + 1j*self.Q*(self.v11_A_star*self.v22_u_x) + 1j*2*self.Q*(self.v11_A_star_x*self.v22_u)
        N31_B_my4 = 1j*self.Q*(self.v11_u*self.v20_A_x) + 1j*self.Q*(self.v11_u*self.v20_A_star_x) - 1j*self.Q*(self.v11_u_star*self.v22_A_x) - 1j*2*self.Q*(self.v11_u_star_x*self.v22_A)
        
        N31_B = N31_B_my1 + N31_B_my2 + N31_B_my3 + N31_B_my4
        
        self.N31_B = N31_B.evaluate()
        
         
    def plot(self):
    
        fig = plt.figure(figsize=(16,4))
        
        # plot N31
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.plot(self.x, self.N31_psi['g'].imag, color="black")
        ax1.plot(self.x, self.N31_psi['g'].real, color="red")
        ax1.set_title(r"$Im(N_{31}^{(\psi)})$")

        ax2 = fig.add_subplot(1, 4, 2)
        ax2.plot(self.x, self.N31_u['g'].real, color="black")
        ax2.plot(self.x, self.N31_u['g'].imag, color="red")
        ax2.set_title(r"$Re(N_{31}^{(u)})$")

        ax3 = fig.add_subplot(1, 4, 3)
        ax3.plot(self.x, self.N31_A['g'].real, color="black")
        ax3.plot(self.x, self.N31_A['g'].imag, color="red")
        ax3.set_title(r"$Re(N_{31}^{(A)})$")

        ax4 = fig.add_subplot(1, 4, 4)
        ax4.plot(self.x, self.N31_B['g'].imag, color="black")
        ax4.plot(self.x, self.N31_B['g'].real, color="red")
        ax4.set_title(r"$Im(N_{31}^{(B)})$")
        
        fig.savefig("n3.png")
                
class AmplitudeAlpha():

    """
    Solves the coefficients of the first amplitude equation for alpha -- e^(iQz) terms.
    
    """
    
    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, run = True, norm = True):
        
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.q = q
        self.beta = beta
        
        if run == True:
        
            self.va = AdjointHomogenous()
            self.va.solve(save = False, norm = True)
        
            v1 = OrderE()
            v1.solve(save=False)
            
            if norm == True:
            
                psi_1 = v1.LEV.state['psi']*v1.scale
                u_1 = v1.LEV.state['u']*v1.scale
                A_1 = v1.LEV.state['A']*v1.scale
                B_1 = v1.LEV.state['B']*v1.scale
            
                self.v11_psi = psi_1.evaluate()
                self.v11_u = u_1.evaluate()
                self.v11_A = A_1.evaluate()
                self.v11_B = B_1.evaluate()
                
                
            else:
            
                self.v11_psi = v1.LEV.state['psi']
                self.v11_u = v1.LEV.state['u']
                self.v11_A = v1.LEV.state['A']
                self.v11_B = v1.LEV.state['B']
            
            o2 = OrderE2()
            o2.solve20()
            o2.solve21(norm = norm)
            o2.solve22()
            
            self.v20_psi = o2.LEV20psi.state['psi20']
            self.v20_u = o2.LEV20u.state['u20']
            self.v20_A = o2.LEV20A.state['A20']
            self.v20_B = o2.LEV20B.state['B20']  
            
            if norm == True:
                v21_psi = o2.LEV21.state['psi21']*o2.scale   
                v21_u = o2.LEV21.state['u21']*o2.scale 
                v21_A = o2.LEV21.state['A21']*o2.scale 
                v21_B = o2.LEV21.state['B21']*o2.scale 
                
                self.v21_psi = v21_psi.evaluate()     
                self.v21_u = v21_u.evaluate() 
                self.v21_A = v21_A.evaluate() 
                self.v21_B = v21_B.evaluate() 
            
            else: 
                
                self.v21_psi = o2.LEV21.state['psi21']
                self.v21_u = o2.LEV21.state['u21']
                self.v21_A = o2.LEV21.state['A21']
                self.v21_B = o2.LEV21.state['B21']
                
            self.v22_psi = o2.LEV22.state['psi22']
            self.v22_u = o2.LEV22.state['u22']
            self.v22_A = o2.LEV22.state['A22']
            self.v22_B = o2.LEV22.state['B22']
            
            self.n3 = N3()
            self.n3.solve31()
            
            
    def solve(self, gridnum = gridnum, save = False):
    
        self.gridnum = gridnum
        self.x = domain.grid(0)
        
        # complex conjugate of N31
        self.N31_psi_star = domain.new_field()
        self.N31_psi_star.name = 'N31_psi_star'
        self.N31_psi_star['g'] = self.n3.N31_psi['g'].conj()
        
        self.N31_u_star = domain.new_field()
        self.N31_u_star.name = 'N31_u_star'
        self.N31_u_star['g'] = self.n3.N31_u['g'].conj()
        
        self.N31_A_star = domain.new_field()
        self.N31_A_star.name = 'N31_A_star'
        self.N31_A_star['g'] = self.n3.N31_A['g'].conj()
        
        self.N31_B_star = domain.new_field()
        self.N31_B_star.name = 'N31_B_star'
        self.N31_B_star['g'] = self.n3.N31_B['g'].conj()
        
        c_psi = self.va.psi*self.N31_psi_star
        self.c_psi = c_psi.evaluate()
        
        self.c_psi = np.sum(self.c_psi['g'])
        
        print('c : ', self.c_amp)
        
        
            
                
                
                

