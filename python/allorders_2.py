import numpy as np
import matplotlib.pyplot as plt
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue, LinearBVP
from scipy.linalg import eig, norm
import pylab
import copy
import pickle
import plot_tools
import streamplot_uneven as su
import random

import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

gridnum = 64
x_basis = Chebyshev(gridnum)
domain = Domain([x_basis], grid_dtype=np.complex128)

class MRI():

    """
    Base class for MRI equations.
    """

    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
    
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.q = q
        self.beta = beta
        self.norm = norm
        
        # Inverse magnetic reynolds number
        self.iRm = 1.0/self.Rm
        
        # Reynolds number
        self.R = self.Rm/self.Pm
        self.iR = 1.0/self.R
        
        self.gridnum = gridnum
        #self.x_basis = Chebyshev(gridnum)
        #self.domain = Domain([self.x_basis], grid_dtype=np.complex128)
    
        print("MRI parameters: ", self.Q, self.Rm, self.Pm, self.q, self.beta, 'norm = ', norm, "Reynolds number", self.R)
        
    def set_boundary_conditions(self, problem):
        
        """
        Adds MRI problem boundary conditions to a ParsedProblem object.
        """
        
        problem.add_left_bc("u = 0")
        problem.add_right_bc("u = 0")
        problem.add_left_bc("psi = 0")
        problem.add_right_bc("psi = 0")
        problem.add_left_bc("A = 0")
        problem.add_right_bc("A = 0")
        problem.add_left_bc("psix = 0")
        problem.add_right_bc("psix = 0")
        problem.add_left_bc("Bx = 0")
        problem.add_right_bc("Bx = 0")
        
        return problem
    
    def solve_LEV(self, problem):
    
        """
        Solves the linear eigenvalue problem for a ParsedProblem object.
        """
        
        problem.expand(domain)
        LEV = LinearEigenvalue(problem, domain)
        LEV.solve(LEV.pencils[0])
        
        return LEV
        
    def get_smallest_eigenvalue_index(self, LEV):
        
        """
        Return index of smallest eigenvalue. Can be positive or negative.
        """
    
        evals = LEV.eigenvalues
        indx = np.arange(len(evals))
        smallest_eval_indx = indx[np.abs(evals) == np.nanmin(np.abs(evals))]
        
        return smallest_eval_indx
    
    def get_smallest_eigenvalue_index_from_above(self, LEV):
    
        """
        Return index of smallest positive eigenvalue.
        """
    
        evals = LEV.eigenvalues
      
        # Mask all nans and infs 
        evals_masked = np.ma.masked_invalid(evals)

        # Anything greater than or equal to zero
        gt_zero = np.ma.less_equal(evals_masked, 0 + 0j)

        # If all eigenvalues are negative, return None
        if np.all(gt_zero):
            return None 
    
        final_arr = np.ma.masked_array(np.absolute(evals_masked), gt_zero)
        smallest_eval_indx = final_arr.argmin()
        
        return smallest_eval_indx
        
    def normalize_all_real_or_imag(self, LEV):
        
        """
        Normalize state vectors such that they are purely real or purely imaginary.
        """
        
        n = np.abs(self.LEV.state['psi']['g'])[13]
        a = self.LEV.state['psi']['g'].real[13]/n
        b = self.LEV.state['psi']['g'].imag[13]/n
        scale = 1j*a/(b*(a**2/b+b)) + 1./(a**2/b +b)
        
        return scale
        
    def get_derivative(self, field):
    
        """
        Take derivative of a single field.
        """
        
        field_x = field.differentiate(0)
        
        if field.name.endswith("x"):
            field_x.name = field.name + "x"
        else:
            field_x.name = field.name + "_x"
            
        return field_x
        
    def get_complex_conjugate(self, field):
        
        """
        Take complex conjugate of a single field.
        """
        
        field_star = domain.new_field()
        field_star.name = field.name + "_star"
        field_star['g'] = field['g'].conj()
        
        return field_star
        
    
class AdjointHomogenous(MRI):

    """
    Solves the adjoint homogenous equation L^dagger V^dagger = 0
    Returns V^dagger
    """

    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
      
        # Set up problem object
        lv1 = ParsedProblem(['x'],
                field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                param_names=['Q', 'iR', 'iRm', 'q', 'beta'])

        lv1.add_equation("1j*Q**2*dt(psi) - 1j*dt(psixx) + 1j*Q*A + 1j*(q - 2)*Q*u - iR*Q**4*psi + 2*iR*Q**2*psixx - iR*dx(psixxx) = 0")
        lv1.add_equation("-1j*dt(u) + 1j*Q*B + 2*1j*Q*psi + iR*Q**2*u - iR*dx(ux) = 0")
        lv1.add_equation("-1j*dt(A) + iRm*Q**2*A - iRm*dx(Ax) - 1j*Q*q*B - 1j*(2/beta)*Q**3*psi + 1j*(2/beta)*Q*psixx = 0")
        lv1.add_equation("-1j*dt(B) + iRm*Q**2*B - iRm*dx(Bx) + 1j*(2/beta)*Q*u = 0")

        lv1.add_equation("dx(psi) - psix = 0")
        lv1.add_equation("dx(psix) - psixx = 0")
        lv1.add_equation("dx(psixx) - psixxx = 0")
        lv1.add_equation("dx(u) - ux = 0")
        lv1.add_equation("dx(A) - Ax = 0")
        lv1.add_equation("dx(B) - Bx = 0")

        lv1.parameters['Q'] = self.Q
        lv1.parameters['iR'] = self.iR
        lv1.parameters['iRm'] = self.iRm
        lv1.parameters['q'] = self.q
        lv1.parameters['beta'] = self.beta
    
        # Set boundary conditions for MRI problem
        lv1 = self.set_boundary_conditions(lv1)
        
        # Solve linear eigenvalue problem
        self.LEV = self.solve_LEV(lv1)
        smallest_eval_indx = self.get_smallest_eigenvalue_index_from_above(self.LEV)
        self.LEV.set_state(smallest_eval_indx)
        
        if self.norm == True:
            scale = self.normalize_all_real_or_imag(self.LEV)
            
            self.psi = (self.LEV.state['psi']*scale).evaluate()
            self.u = (self.LEV.state['u']*scale).evaluate()
            self.A = (self.LEV.state['A']*scale).evaluate()
            self.B = (self.LEV.state['B']*scale).evaluate()
        else:
            self.psi = self.LEV.state['psi']
            self.u = self.LEV.state['u']
            self.A = self.LEV.state['A']
            self.B = self.LEV.state['B']
        
        self.psi.name = "psi"
        self.u.name = "u"
        self.A.name = "A"
        self.B.name = "B"
            
        # Take all relevant derivates for use with higher order terms
        self.psi_x = self.get_derivative(self.psi)
        self.psi_xx = self.get_derivative(self.psi_x)
        self.psi_xxx = self.get_derivative(self.psi_xx)
      
        self.u_x = self.get_derivative(self.u)
        
        self.A_x = self.get_derivative(self.A)
        self.A_xx = self.get_derivative(self.A_x)
        self.A_xxx = self.get_derivative(self.A_xx)
        
        self.B_x = self.get_derivative(self.B)
        
            
class OrderE(MRI):

    """
    Solves the order(epsilon) equation L V_1 = 0
    This is simply the linearized MRI.
    Returns V_1
    """

    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
      
        lv1 = ParsedProblem(['x'],
                field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                param_names=['Q', 'iR', 'iRm', 'q', 'beta'])

        lv1.add_equation("-1j*dt(psixx) + 1j*Q**2*dt(psi) - iR*dx(psixxx) + 2*iR*Q**2*psixx - iR*Q**4*psi - 2*1j*Q*u - (2/beta)*1j*Q*dx(Ax) + (2/beta)*Q**3*1j*A = 0")
        lv1.add_equation("-1j*dt(u) - iR*dx(ux) + iR*Q**2*u + (2-q)*1j*Q*psi - (2/beta)*1j*Q*B = 0") 
        lv1.add_equation("-1j*dt(A) - iRm*dx(Ax) + iRm*Q**2*A - 1j*Q*psi = 0") 
        lv1.add_equation("-1j*dt(B) - iRm*dx(Bx) + iRm*Q**2*B - 1j*Q*u + q*1j*Q*A = 0")
        
        lv1.add_equation("dx(psi) - psix = 0")
        lv1.add_equation("dx(psix) - psixx = 0")
        lv1.add_equation("dx(psixx) - psixxx = 0")
        lv1.add_equation("dx(u) - ux = 0")
        lv1.add_equation("dx(A) - Ax = 0")
        lv1.add_equation("dx(B) - Bx = 0")

        lv1.parameters['Q'] = self.Q
        lv1.parameters['iR'] = self.iR
        lv1.parameters['iRm'] = self.iRm
        lv1.parameters['q'] = self.q
        lv1.parameters['beta'] = self.beta
    
        lv1 = self.set_boundary_conditions(lv1)
        self.LEV = self.solve_LEV(lv1)
        smallest_eval_indx = self.get_smallest_eigenvalue_index_from_above(self.LEV)
        
        self.LEV.set_state(smallest_eval_indx)
        
        if self.norm == True:
            scale = self.normalize_all_real_or_imag(self.LEV)
            
            self.psi = (self.LEV.state['psi']*scale).evaluate()
            self.u = (self.LEV.state['u']*scale).evaluate()
            self.A = (self.LEV.state['A']*scale).evaluate()
            self.B = (self.LEV.state['B']*scale).evaluate()
        else:
            self.psi = self.LEV.state['psi']
            self.u = self.LEV.state['u']
            self.A = self.LEV.state['A']
            self.B = self.LEV.state['B']
            
        self.psi.name = "psi"
        self.u.name = "u"
        self.A.name = "A"
        self.B.name = "B"
            
        # Take all relevant derivates for use with higher order terms
        self.psi_x = self.get_derivative(self.psi)
        self.psi_xx = self.get_derivative(self.psi_x)
        self.psi_xxx = self.get_derivative(self.psi_xx)
      
        self.u_x = self.get_derivative(self.u)
        
        self.A_x = self.get_derivative(self.A)
        self.A_xx = self.get_derivative(self.A_x)
        self.A_xxx = self.get_derivative(self.A_xx)
        
        self.B_x = self.get_derivative(self.B)
        
        # Also take relevant complex conjugates
        self.psi_star = self.get_complex_conjugate(self.psi)
        self.psi_star_x = self.get_derivative(self.psi_star)
        self.psi_star_xx = self.get_derivative(self.psi_star_x)
        self.psi_star_xxx = self.get_derivative(self.psi_star_xx)
        
        self.u_star = self.get_complex_conjugate(self.u)
        self.u_star_x = self.get_derivative(self.u_star)
        
        self.A_star = self.get_complex_conjugate(self.A)
        self.A_star_x = self.get_derivative(self.A_star)
        self.A_star_xx = self.get_derivative(self.A_star_x)
        self.A_star_xxx = self.get_derivative(self.A_star_xx)
        
        self.B_star = self.get_complex_conjugate(self.B)
        self.B_star_x = self.get_derivative(self.B_star)
        
class N2(MRI):

    """
    Solves the nonlinear term N2
    Returns N2
    
    """
    
    def __init__(self, o1 = None, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
    
        if o1 == None:
            o1 = OrderE(Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
        else:
            MRI.__init__(self, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
    
        N22psi = 1j*self.Q*o1.psi*(o1.psi_xxx - self.Q**2*o1.psi_x) - o1.psi_x*(1j*self.Q*o1.psi_xx - 1j*self.Q**3*o1.psi) + (2/self.beta)*o1.A_x*(1j*self.Q*o1.A_xx - 1j*self.Q**3*o1.A) - (2/self.beta)*1j*self.Q*o1.A*(o1.A_xxx - self.Q**2*o1.A_x)
        self.N22_psi = N22psi.evaluate()
        self.N22_psi.name = "N22_psi"
    
        N20psi = 1j*self.Q*o1.psi*(o1.psi_star_xxx - self.Q**2*o1.psi_star_x) - o1.psi_x*(-1j*self.Q*o1.psi_star_xx + 1j*self.Q**3*o1.psi_star) + (2/self.beta)*o1.A_x*(-1j*self.Q*o1.A_star_xx + 1j*self.Q**3*o1.A_star) - (2/self.beta)*1j*self.Q*o1.A*(o1.A_star_xxx - self.Q**2*o1.A_star_x)
        self.N20_psi = N20psi.evaluate()
        self.N20_psi.name = "N20_psi"
        
        N22u = 1j*self.Q*o1.psi*o1.u_x - o1.psi_x*1j*self.Q*o1.u - (2/self.beta)*1j*self.Q*o1.A*o1.B_x + (2/self.beta)*o1.A_x*1j*self.Q*o1.B
        self.N22_u = N22u.evaluate()
        self.N22_u.name = "N22_u"
        
        N20u = 1j*self.Q*o1.psi*o1.u_star_x + o1.psi_x*1j*self.Q*o1.u_star - (2/self.beta)*1j*self.Q*o1.A*o1.B_star_x - (2/self.beta)*o1.A_x*1j*self.Q*o1.B_star
        self.N20_u = N20u.evaluate()
        self.N20_u.name = "N20_u"
        
        N22A = -1j*self.Q*o1.A*o1.psi_x + o1.A_x*1j*self.Q*o1.psi
        self.N22_A = N22A.evaluate()
        self.N22_A.name = "N22_A"
        
        N20A = -1j*self.Q*o1.A*o1.psi_star_x - o1.A_x*1j*self.Q*o1.psi_star
        self.N20_A = N20A.evaluate()
        self.N20_A.name = "N20_A"

        N22B = 1j*self.Q*o1.psi*o1.B_x - o1.psi_x*1j*self.Q*o1.B - 1j*self.Q*o1.A*o1.u_x + o1.A_x*1j*self.Q*o1.u
        self.N22_B = N22B.evaluate()
        self.N22_B.name = "N22_B"
        
        N20B = 1j*self.Q*o1.psi*o1.B_star_x + o1.psi_x*1j*self.Q*o1.B_star - 1j*self.Q*o1.A*o1.u_star_x - o1.A_x*1j*self.Q*o1.u_star
        self.N20_B = N20B.evaluate()
        self.N20_B.name = "N20_B"

        
        
        
        
        
        
        
      
      
      
      
    