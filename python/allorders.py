import numpy as np
import matplotlib.pyplot as plt
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue
from scipy.linalg import eig, norm
import pylab
import copy
import pickle

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
        
    
    def solve(self, gridnum = 64, save = True):

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
        
        self.psi = LEV.state['psi']['g']
        self.u = LEV.state['u']['g']
        self.A = LEV.state['A']['g']
        self.B = LEV.state['B']['g']
        
        psi = copy.copy(LEV.state['psi'])

        if save == True:
            pickle.dump(psi, open("V_adj_psi.p", "wb"), protocol=-1)
            pickle.dump(LEV.state['u'], open("V_adj_u.p", "wb"))
            pickle.dump(LEV.state['A'], open("V_adj_A.p", "wb"))
            pickle.dump(LEV.state['B'], open("V_adj_B.p", "wb"))
            


    def plot(self):
        
        #Normalized to resemble Umurhan+
        norm1 = -0.9/np.min(self.u.imag) #norm(LEV.eigenvectors)
    
        fig = plt.figure()
    
        ax1 = fig.add_subplot(221)
        ax1.plot(self.x, self.psi.imag*norm1, color="black")
        ax1.plot(self.x, self.psi.real*norm1, color="red")
        ax1.set_title(r"Im($\psi^\dagger$)")

        ax2 = fig.add_subplot(222)
        ax2.plot(self.x, self.u.imag*norm1, color="red")
        ax2.plot(self.x, self.u.real*norm1, color="black")
        ax2.set_title("Re($u^\dagger$)")

        ax3 = fig.add_subplot(223)
        ax3.plot(self.x, self.A.imag*norm1, color="red")
        ax3.plot(self.x, self.A.real*norm1, color="black")
        ax3.set_title("Re($A^\dagger$)")

        ax4 = fig.add_subplot(224)
        ax4.plot(self.x, self.B.imag*norm1, color="black")
        ax4.plot(self.x, self.B.real*norm1, color="red")
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
        
    
    def solve(self, gridnum = 64, save = True):

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

        lv1.expand(domain)
        LEV = LinearEigenvalue(lv1, domain)
        LEV.solve(LEV.pencils[0])
        
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
    
    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, run = True):
        
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.q = q
        self.beta = beta
        
        if run == True:
            v1 = OrderE()
            v1.solve(save=False)
            self.psi_1 = v1.LEV.state['psi']
            self.u_1 = v1.LEV.state['u']
            self.A_1 = v1.LEV.state['A']
            self.B_1 = v1.LEV.state['B']
        
        
    def run_V1(self):
        v1 = OrderE()
        v1.solve(save=False)
        self.psi_1 = v1.LEV.state['psi']
        self.u_1 = v1.LEV.state['u']
        self.A_1 = v1.LEV.state['A']
        self.B_1 = v1.LEV.state['B']
        
    
    def solve(self, gridnum = 64, save = True):

        self.gridnum = gridnum
        
        # inverse magnetic reynolds number
        iRm = 1./self.Rm

        # rayleigh number defined from prandtl number
        R = self.Rm/self.Pm
        iR = 1./R
        
        beta = self.beta
        Q = self.Q
        q = self.q
        
        #psi_1 = pickle.load(open("V_1_psi.p", "rb"))
        #u_1 = pickle.load(open("V_1_u.p", "rb"))
        #A_1 = pickle.load(open("V_1_A.p", "rb"))
        #B_1 = pickle.load(open("V_1_B.p", "rb"))
        
        # take derivatives
        psi_1 = self.psi_1
        A_1 = self.A_1
        u_1 = self.u_1
        B_1 = self.B_1
        
        print(A_1['g'])
        
        psi_1_x = psi_1.differentiate(0)
        psi_1_xx = psi_1_x.differentiate(0)
        psi_1_xxx = psi_1_xx.differentiate(0)
        
        u_1_x = u_1.differentiate(0)
        
        A_1_x = A_1.differentiate(0)
        A_1_xx = A_1_x.differentiate(0)
        A_1_xxx = A_1_xx.differentiate(0)
        
        B_1_x = B_1.differentiate(0)
        
        
        N2psi = 1j*Q*psi_1*psi_1_xxx - 1j*Q*psi_1*Q**2*psi_1_xx - psi_1_x*1j*Q*psi_1_xx + psi_1_x*Q**2*psi_1 - (2/self.beta)*1j*Q*A_1*A_1_xxx + (2/self.beta)*1j*Q*A_1*Q**2*A_1_x + (2/self.beta)*A_1_x*1j*Q*A_1_xx - (2/self.beta)*A_1_x*Q**2*A_1
        self.N2_psi = N2psi.evaluate()
        
        N2u = 1j*Q*psi_1*u_1_x - psi_1_x*1j*Q*u_1 - (2/self.beta)*1j*Q*A_1*B_1_x + (2/self.beta)*A_1_x*1j*Q*B_1
        self.N2_u = N2u.evaluate()
        
        N2A = -1j*Q*A_1*psi_1_x + A_1_x*1j*Q*psi_1
        self.N2_A = N2A.evaluate()
        
        N2B = 1j*Q*psi_1*B_1_x - psi_1_x*1j*Q*B_1 - 1j*Q*A_1*u_1_x + A_1_x*1j*Q*u_1
        self.N2_B = N2B.evaluate()
        
        
class OrderE2():

    """
    Solves the second order equation L V2 = N2 - Ltwiddle V1
    Returns V2
    
    """
    
    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, run = True):
        
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.q = q
        self.beta = beta
        
        if run == True:
            v1 = OrderE()
            v1.solve(save=False)
            self.psi_1 = v1.LEV.state['psi']
            self.u_1 = v1.LEV.state['u']
            self.A_1 = v1.LEV.state['A']
            self.B_1 = v1.LEV.state['B']
            
            n2 = N2()
            n2.solve()
            self.n2_psi = n2.N2_psi
            self.n2_u = n2.N2_u
            self.n2_A = n2.N2_A
            self.n2_B = n2.N2_B
            
    def solve(self, gridnum = 64, save = True):
    
        # inverse magnetic reynolds number
        iRm = 1./self.Rm

        # rayleigh number defined from prandtl number
        R = self.Rm/self.Pm
        iR = 1./R
        
        beta = self.beta
        Q = self.Q
        q = self.q
        
        # derivatives
        psi_1_x = self.psi_1.differentiate(0)
        psi_1_xx = psi_1_x.differentiate(0)
        
        A_1_x = self.A_1.differentiate(0)
        A_1_xx = A_1_x.differentiate(0)
        
        # second term: L1twiddle V1
        term2_psi = -3*(2/self.beta)*Q**2*self.A_1 + (2/self.beta)*A_1_xx - 4*iR*1j*Q**3*self.psi_1 + 4*iR*1j*Q*psi_1_xx + 2*self.u_1
        self.term2_psi = term2_psi.evaluate()
        
        term2_u = (2/self.beta)*self.B_1 + 2*iR*Q*self.u_1 + (q - 2)*self.psi_1
        self.term2_u = term2_u.evaluate()
        
        term2_A = 2*iRm*1j*Q*self.A_1 + self.psi_1
        self.term2_A = term2_A.evaluate()
        
        term2_B = -q*self.A_1 + 2*iRm*1j*Q*self.B_1 + self.u_1
        self.term2_B = term2_B.evaluate()
        
        # righthand side
        n2_psi = self.n2_psi
        term2_psi = self.term2_psi
        rhs_psi = n2_psi + term2_psi
        self.rhs_psi = rhs_psi.evaluate()
        
        rhs_u = self.n2_u + self.term2_u
        self.rhs_u = rhs_u.evaluate()
        
        rhs_A = self.n2_A + self.term2_A
        self.rhs_A = rhs_A.evaluate()
        
        rhs_B = self.n2_B + self.term2_B
        self.rhs_B = rhs_B.evaluate()
        
        # nonconstant coefficients
        #rhs_psi = x_basis.grid**2
        #rhs_psi = x_basis.grid
        rhs_psi = self.rhs_psi['g']
        #rhs_psi = 2
        rhs_u = self.rhs_u['g']
        rhs_A = self.rhs_A['g']
        rhs_B = self.rhs_B['g']
        
        # define problem using righthand side as nonconstant coefficients
        
        lv2 = ParsedProblem(['x'],
                              field_names=['psi20', 'psi20x', 'psi20xx', 'psi20xxx', 'u20', 'u20x', 'A20', 'A20x', 'B20', 'B20x', 'psi21', 'psi21x', 'psi21xx', 'psi21xxx', 'u21', 'u21x', 'A21', 'A21x', 'B21', 'B21x', 'psi22', 'psi22x', 'psi22xx', 'psi22xxx', 'u22', 'u22x', 'A22', 'A22x', 'B22', 'B22x'],
                              param_names=['Q', 'iR', 'iRm', 'q', 'beta', 'rhs_psi', 'rhs_u', 'rhs_A', 'rhs_B'])
                  
        #x_basis = Chebyshev(gridnum)
        #domain = Domain([x_basis], grid_dtype=np.complex128)

        # second order equations
        lv2.add_equation("1j*dt(dx(psi20x)) + 1j*dt(psi20) + 1j*Q**2*dt(psi21) - 1j*dt(dx(psi21x)) - 1j*4*Q**2*dt(psi22) + 1j*dt(dx(psi22x)) + (2/beta)*dx(A20x) + (2/beta)*A20 + 1j*(2/beta)*Q**3*A21 - 1j*(2/beta)*Q*dx(A21x) - 8*1j*(2/beta)*Q**3*A22 + 2*1j*(2/beta)*Q*dx(A22x) - 2*1j*Q*u21 + 4*1j*Q*u22 + iR*dx(psi20xxx) + iR*2*dx(psi20x) + iR*psi20 - iR*Q**4*psi21 + 2*iR*Q**2*dx(psi21x) - iR*dx(psi21xxx) + 16*iR*Q**4*psi22 - 8*iR*Q**2*dx(psi22x) + iR*dx(psi22xxx) + 2*u20 = rhspsi")
        lv2.add_equation("1j*dt(u20) + 1j*dt(u21) + 1j*dt(u22) + (2/beta)*B20 - 1j*(2/beta)*Q*B21 + 2*1j*(2/beta)*Q*B22 - 1j*Q*(q - 2)*psi21 + 2*1j*Q*(q - 2)*psi22 + (q - 2)*psi20 + iR*dx(u20x) + iR*u20 + iR*Q**2*u21 - iR*dx(u21x) - 4*iR*Q**2*u22 + iR*dx(u22x) = rhs_u")
        lv2.add_equation("1j*dt(A20) + 1j*dt(A21) + 1j*dt(A22) + iRm*dx(A20x) + iRm*A20 + iRm*Q**2*A21 - iRm*dx(A21x) - 4*iRm*Q**2*A22 + iRm*dx(A22x) -1j*Q*psi21 + 2*1j*Q*psi22 + psi20 = rhs_A")
        lv2.add_equation("1j*dt(B20) + 1j*dt(B21) + 1j*dt(B22) + -q*A20 + 1j*Q*q*A21 - 2*1j*Q*q*A22 + iRm*dx(B20x) + iRm*B20 + iRm*Q**2*B21 - iRm*dx(B21x) - iRm*4*Q**2*B22 + iRm*dx(B22x) - 1j*Q*u21 + 2*1j*Q*u22 + u20 = rhs_B")
        
        # need to 'stack' equations for each component of V2 in order to have 30 equations
        
        
        lv2.add_equation("dx(psi20) - psi20x = 0")
        lv2.add_equation("dx(psi20x) - psi20xx = 0")
        lv2.add_equation("dx(psi20xx) - psi20xxx = 0")
        
        lv2.add_equation("dx(psi21) - psi21x = 0")
        lv2.add_equation("dx(psi21x) - psi21xx = 0")
        lv2.add_equation("dx(psi21xx) - psi21xxx = 0")
        
        lv2.add_equation("dx(psi22) - psi22x = 0")
        lv2.add_equation("dx(psi22x) - psi22xx = 0")
        lv2.add_equation("dx(psi22xx) - psi22xxx = 0")
        
        lv2.add_equation("dx(u20) - u20x = 0")
        lv2.add_equation("dx(u21) - u21x = 0")
        lv2.add_equation("dx(u22) - u22x = 0")
        
        lv2.add_equation("dx(A20) - A20x = 0")
        lv2.add_equation("dx(A21) - A21x = 0")
        lv2.add_equation("dx(A22) - A22x = 0")
        
        lv2.add_equation("dx(B20) - B20x = 0")
        lv2.add_equation("dx(B21) - B21x = 0")
        lv2.add_equation("dx(B22) - B22x = 0")

        # boundary conditions
        lv2.add_left_bc("psi20 = 0")
        lv2.add_right_bc("psi20 = 0")
        lv2.add_left_bc("psi21 = 0")
        lv2.add_right_bc("psi21 = 0")
        lv2.add_left_bc("psi22 = 0")
        lv2.add_right_bc("psi22 = 0")
        
        lv2.add_left_bc("u20 = 0")
        lv2.add_right_bc("u20 = 0")
        lv2.add_left_bc("u21 = 0")
        lv2.add_right_bc("u21 = 0")
        lv2.add_left_bc("u22 = 0")
        lv2.add_right_bc("u22 = 0")
        
        lv2.add_left_bc("A20 = 0")
        lv2.add_right_bc("A20 = 0")
        lv2.add_left_bc("A21 = 0")
        lv2.add_right_bc("A21 = 0")
        lv2.add_left_bc("A22 = 0")
        lv2.add_right_bc("A22 = 0")
        
        lv2.add_left_bc("psi20x = 0")
        lv2.add_right_bc("psi20x = 0")
        lv2.add_left_bc("psi21x = 0")
        lv2.add_right_bc("psi21x = 0")
        lv2.add_left_bc("psi22x = 0")
        lv2.add_right_bc("psi22x = 0")
        
        lv2.add_left_bc("B20x = 0")
        lv2.add_right_bc("B20x = 0")
        lv2.add_left_bc("B21x = 0")
        lv2.add_right_bc("B21x = 0")
        lv2.add_left_bc("B22x = 0")
        lv2.add_right_bc("B22x = 0")

        # parameters
        lv2.parameters['Q'] = Q
        lv2.parameters['iR'] = iR
        lv2.parameters['iRm'] = iRm
        lv2.parameters['q'] = q
        lv2.parameters['beta'] = beta
        lv2.parameters['rhs_psi'] = rhs_psi
        lv2.parameters['rhs_u'] = rhs_u
        lv2.parameters['rhs_A'] = rhs_A
        lv2.parameters['rhs_B'] = rhs_B

        # expand domain to gridnum points
        lv2.expand(domain, order = gridnum)
        LEV = LinearEigenvalue(lv2, domain)
        LEV.solve(LEV.pencils[0])
        
        self.LEV = LEV

        #Find the eigenvalue that is closest to zero.
        evals = LEV.eigenvalues
        indx = np.arange(len(evals))
        e0 = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
        print('eigenvalue', evals[e0])
       
        # set state
        self.x = domain.grid(0)
        LEV.set_state(e0[0])
        
        self.psi20 = LEV.state['psi20']['g']
        self.u20 = LEV.state['u20']['g']
        self.A20 = LEV.state['A20']['g']
        self.B20 = LEV.state['B20']['g']
        
        self.psi21 = LEV.state['psi21']['g']
        self.u21 = LEV.state['u21']['g']
        self.A21 = LEV.state['A21']['g']
        self.B21 = LEV.state['B21']['g']
        
        self.psi22 = LEV.state['psi22']['g']
        self.u22 = LEV.state['u22']['g']
        self.A22 = LEV.state['A22']['g']
        self.B22 = LEV.state['B22']['g']
        
        
    
        
            
            

