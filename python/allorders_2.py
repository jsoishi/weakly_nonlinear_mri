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
    
    def take_derivatives(self, LEV):
    
        """
        Take first, second, and third derivatives of all terms.
        """
    
        self.psi_x = LEV.state['psi'].differentiate(0)
        self.psi_xx = self.psi_x.differentiate(0)
        self.psi_xxx = self.psi_xx.differentiate(0)
        
        self.u_x = LEV.state['u'].differentiate(0)
        self.u_xx = self.u_x.differentiate(0)
        self.u_xxx = self.u_xx.differentiate(0)
        
        self.A_x = LEV.state['A'].differentiate(0)
        self.A_xx = self.A_x.differentiate(0)
        self.A_xxx = self.A_xx.differentiate(0)
        
        self.B_x = LEV.state['B'].differentiate(0)
        self.B_xx = self.B_x.differentiate(0)
        self.B_xxx = self.B_xx.differentiate(0)
        
    def take_complex_conjugates(self, LEV):
    
        """
        Take complex conjugates of all terms, including derivatives.
        """
        
        self.psi_star = domain.new_field()
        self.psi_star.name = "psi_star"
        self.psi_star['g'] = self.psi['g'].conj()
        
        self.psi_x_star = domain.new_field()
        self.psi_x_star.name = "psi_x_star"
        self.psi_x_star['g'] = self.psi_x['g'].conj()
        
        self.psi_xx_star = domain.new_field()
        self.psi_xx_star.name = "psi_xx_star"
        self.psi_xx_star['g'] = self.psi_xx['g'].conj()
        
        self.psi_xxx_star = domain.new_field()
        self.psi_xxx_star.name = "psi_xx_star"
        self.psi_xxx_star['g'] = self.psi_xxx['g'].conj()
        
        self.u_star = domain.new_field()
        self.u_star.name = "u_star"
        self.u_star['g'] = self.u['g'].conj()
        
        self.u_x_star = domain.new_field()
        self.u_x_star.name = "u_x_star"
        self.u_x_star['g'] = self.u_x['g'].conj()
        
        self.u_xx_star = domain.new_field()
        self.u_xx_star.name = "u_xx_star"
        self.u_xx_star['g'] = self.u_xx['g'].conj()
        
        self.u_xxx_star = domain.new_field()
        self.u_xxx_star.name = "u_xx_star"
        self.u_xxx_star['g'] = self.u_xxx['g'].conj()
        
        self.A_star = domain.new_field()
        self.A_star.name = "A_star"
        self.A_star['g'] = self.A['g'].conj()
        
        self.A_x_star = domain.new_field()
        self.A_x_star.name = "A_x_star"
        self.A_x_star['g'] = self.A_x['g'].conj()
        
        self.A_xx_star = domain.new_field()
        self.A_xx_star.name = "A_xx_star"
        self.A_xx_star['g'] = self.A_xx['g'].conj()
        
        self.A_xxx_star = domain.new_field()
        self.A_xxx_star.name = "A_xx_star"
        self.A_xxx_star['g'] = self.A_xxx['g'].conj()
        
        self.B_star = domain.new_field()
        self.B_star.name = "B_star"
        self.B_star['g'] = self.B['g'].conj()
        
        self.B_x_star = domain.new_field()
        self.B_x_star.name = "B_x_star"
        self.B_x_star['g'] = self.B_x['g'].conj()
        
        self.B_xx_star = domain.new_field()
        self.B_xx_star.name = "B_xx_star"
        self.B_xx_star['g'] = self.B_xx['g'].conj()
        
        self.B_xxx_star = domain.new_field()
        self.B_xxx_star.name = "B_xx_star"
        self.B_xxx_star['g'] = self.B_xxx['g'].conj()
        
    
class AdjointHomogenous(MRI):

    """
    Solves the adjoint homogenous equation L^dagger V^dagger = 0
    Returns V^dagger
    """

    def __init__(self):
        MRI.__init__(self)
      
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
            
        # Take all relevant derivates and complex conjugates for use with higher order terms
        self.take_derivatives(self.LEV)
        self.take_complex_conjugates(self.LEV)
            
class OrderE(MRI):

    """
    Solves the order(epsilon) equation L V_1 = 0
    This is simply the linearized MRI.
    Returns V_1
    """

    def __init__(self):
        MRI.__init__(self)
      
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
            
        # Take all relevant derivates and complex conjugates for use with higher order terms
        self.take_derivatives(self.LEV)
        self.take_complex_conjugates(self.LEV)
        
class N2():

    """
    Solves the nonlinear term N2
    Returns N2
    
    """
    
    
            

        
        
        
        
        
        
        
      
      
      
      
    