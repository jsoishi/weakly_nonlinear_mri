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
        
    def solve_BVP(self, problem):
    
        """
        Solves the boundary value problem for a ParsedProblem object.
        """
    
        problem.expand(domain, order = gridnum)
        BVP = LinearBVP(problem, domain)
        BVP.solve()
        
        return BVP
        
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
        
    def take_inner_product(self, vector1, vector2):
        
        """
        Take inner product < vector1 | vector2 >
        """

        inner_product = vector1[0]['g']*vector2[0]['g'].conj() + vector1[1]['g']*vector2[1]['g'].conj() + vector1[2]['g']*vector2[2]['g'].conj() + vector1[3]['g']*vector2[3]['g'].conj()
        
        ip = domain.new_field()
        ip.name = "inner product"
        ip['g'] = inner_product
        ip = ip.integrate(x_basis)
        ip = ip['g'][0]/2.0
        
        return ip
    
class AdjointHomogenous(MRI):

    """
    Solves the adjoint homogenous equation L^dagger V^dagger = 0
    Returns V^dagger
    """

    def __init__(self, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        
        print("initializing Adjoint Homogenous")
        
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
        
        print("initializing Order E")
        
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
    
        print("initializing N2")
    
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

        
class OrderE2(MRI):

    """
    Solves the second order equation L V2 = N2 - Ltwiddle V1
    Returns V2
    
    """
    
    def __init__(self, o1 = None, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
    
        print("initializing Order E2")
        
        if o1 == None:
            o1 = OrderE(Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            n2 = N2(Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
        else:
            MRI.__init__(self, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
            n2 = N2(Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
    
        # righthand side for the 20 terms (e^0)
        rhs20_psi = n2.N20_psi['g']
        rhs20_u = n2.N20_u['g'] 
        rhs20_A = n2.N20_A['g'] 
        rhs20_B = n2.N20_B['g'] 
        
        # V20 equations are separable because dz terms -> 0
        bv20psi = ParsedProblem(['x'],
                      field_names=['psi20', 'psi20x', 'psi20xx', 'psi20xxx'],
                      param_names=['iR', 'rhs20_psi'])
        bv20psi.add_equation("iR*dx(psi20xxx) = rhs20_psi")
        bv20psi.add_equation("dx(psi20) - psi20x = 0")
        bv20psi.add_equation("dx(psi20x) - psi20xx = 0")
        bv20psi.add_equation("dx(psi20xx) - psi20xxx = 0")
        bv20psi.parameters['iR'] = self.iR
        bv20psi.parameters['rhs20_psi'] = rhs20_psi
        bv20psi.add_left_bc("psi20 = 0")
        bv20psi.add_right_bc("psi20 = 0")
        bv20psi.add_left_bc("psi20x = 0")
        bv20psi.add_right_bc("psi20x = 0")
        
        bv20u = ParsedProblem(['x'],
                      field_names=['u20', 'u20x'],
                      param_names=['iR', 'rhs20_u'])
        bv20u.add_equation("iR*dx(u20x) = rhs20_u")
        bv20u.add_equation("dx(u20) - u20x = 0")
        bv20u.parameters['iR'] = self.iR
        bv20u.parameters['rhs20_u'] = rhs20_u
        bv20u.add_left_bc("u20 = 0")
        bv20u.add_right_bc("u20 = 0")
        
        bv20A = ParsedProblem(['x'],
                              field_names=['A20', 'A20x'],
                              param_names=['iRm', 'rhs20_A'])
        bv20A.add_equation("iRm*dx(A20x) = rhs20_A")
        bv20A.add_equation("dx(A20) - A20x = 0")
        bv20A.parameters['iRm'] = self.iRm
        bv20A.parameters['rhs20_A'] = rhs20_A
        bv20A.add_left_bc("A20 = 0")
        bv20A.add_right_bc("A20 = 0")
        
        bv20B = ParsedProblem(['x'],
                              field_names=['B20', 'B20x'],
                              param_names=['iRm', 'rhs20_B'])
        bv20B.add_equation("iRm*dx(B20x) = rhs20_B")
        bv20B.add_equation("dx(B20) - B20x = 0")
        bv20B.parameters['iRm'] = self.iRm
        bv20B.parameters['rhs20_B'] = rhs20_B
        bv20B.add_left_bc("B20x = 0")
        bv20B.add_right_bc("B20x = 0")
      
        self.BVPpsi = self.solve_BVP(bv20psi)
        self.psi20 = self.BVPpsi.state['psi20']
        
        self.BVPu = self.solve_BVP(bv20u)
        self.u20 = self.BVPu.state['u20']
        
        self.BVPA = self.solve_BVP(bv20A)
        self.A20 = self.BVPA.state['A20']
        
        self.BVPB = self.solve_BVP(bv20B)
        self.B20 = self.BVPB.state['B20']
        
        # V21 equations are coupled
        # second term: L1twiddle V1
        term2_psi = -3*(2/self.beta)*self.Q**2*o1.A + (2/self.beta)*o1.A_xx - 4*self.iR*1j*self.Q**3*o1.psi + 4*self.iR*1j*self.Q*o1.psi_xx + 2*o1.u
        self.term2_psi = term2_psi.evaluate()
        
        term2_u = (2/self.beta)*o1.B + 2*self.iR*self.Q*o1.u + (self.q - 2)*o1.psi
        self.term2_u = term2_u.evaluate()
        
        term2_A = 2*self.iRm*1j*self.Q*o1.A + o1.psi
        self.term2_A = term2_A.evaluate()
        
        term2_B = -self.q*o1.A + 2*self.iRm*1j*self.Q*o1.B + o1.u
        self.term2_B = term2_B.evaluate()
        
        # righthand side for the 21 terms (e^iQz)
        rhs21_psi = -self.term2_psi['g']
        rhs21_u = -self.term2_u['g']
        rhs21_A = -self.term2_A['g']
        rhs21_B = -self.term2_B['g']
                
        # define problem using righthand side as nonconstant coefficients
        
        bv21 = ParsedProblem(['x'],
              field_names=['psi21', 'psi21x', 'psi21xx', 'psi21xxx', 'u21', 'u21x', 'A21', 'A21x', 'B21', 'B21x'],
              param_names=['Q', 'iR', 'iRm', 'q', 'beta', 'rhs20_psi', 'rhs20_u', 'rhs20_A', 'rhs20_B', 'rhs21_psi', 'rhs21_u', 'rhs21_A', 'rhs21_B', 'rhs22_psi', 'rhs22_u', 'rhs22_A', 'rhs22_B'])
          
        bv21.add_equation("1j*(2/beta)*Q**3*A21 - 1j*(2/beta)*Q*dx(A21x) - 2*1j*Q*u21 - iR*Q**4*psi21 + 2*iR*Q**2*psi21xx - iR*dx(psi21xxx) = rhs21_psi")
        bv21.add_equation("-1j*(2/beta)*Q*B21 - 1j*Q*(q - 2)*psi21 + iR*Q**2*u21 - iR*dx(u21x) = rhs21_u")
        bv21.add_equation("iRm*Q**2*A21 - iRm*dx(A21x) - 1j*Q*psi21 = rhs21_A")
        bv21.add_equation("1j*Q*q*A21 + iRm*Q**2*B21 - iRm*dx(B21x) - 1j*Q*u21 = rhs21_B")    

        bv21.add_equation("dx(psi21) - psi21x = 0")
        bv21.add_equation("dx(psi21x) - psi21xx = 0")
        bv21.add_equation("dx(psi21xx) - psi21xxx = 0")
        bv21.add_equation("dx(u21) - u21x = 0")
        bv21.add_equation("dx(A21) - A21x = 0")
        bv21.add_equation("dx(B21) - B21x = 0")

        # boundary conditions
        bv21.add_left_bc("psi21 = 0")
        bv21.add_right_bc("psi21 = 0")
        bv21.add_left_bc("u21 = 0")
        bv21.add_right_bc("u21 = 0")
        bv21.add_left_bc("A21 = 0")
        bv21.add_right_bc("A21 = 0")
        bv21.add_left_bc("psi21x = 0")
        bv21.add_right_bc("psi21x = 0")
        bv21.add_left_bc("B21x = 0")
        bv21.add_right_bc("B21x = 0")

        # parameters
        bv21.parameters['Q'] = self.Q
        bv21.parameters['iR'] = self.iR
        bv21.parameters['iRm'] = self.iRm
        bv21.parameters['q'] = self.q
        bv21.parameters['beta'] = self.beta
        bv21.parameters['rhs21_psi'] = rhs21_psi
        bv21.parameters['rhs21_u'] = rhs21_u
        bv21.parameters['rhs21_A'] = rhs21_A
        bv21.parameters['rhs21_B'] = rhs21_B
        
        self.BVP21 = self.solve_BVP(bv21)
        self.psi21 = self.BVP21.state['psi21']
        self.u21 = self.BVP21.state['u21']
        self.A21 = self.BVP21.state['A21']
        self.B21 = self.BVP21.state['B21']
        
        #V22 equations are coupled
        rhs22_psi = n2.N22_psi['g'] 
        rhs22_u = n2.N22_u['g'] 
        rhs22_A = n2.N22_A['g'] 
        rhs22_B = n2.N22_B['g'] 
        
        self.rhs22_psi = rhs22_psi
        self.rhs22_u = rhs22_u
        self.rhs22_A = rhs22_A
        self.rhs22_B = rhs22_B
                
        # define problem using righthand side as nonconstant coefficients
        bv22 = ParsedProblem(['x'],
              field_names=['psi22', 'psi22x', 'psi22xx', 'psi22xxx', 'u22', 'u22x', 'A22', 'A22x', 'B22', 'B22x'],
              param_names=['Q', 'iR', 'iRm', 'q', 'beta', 'rhs22_psi', 'rhs22_u', 'rhs22_A', 'rhs22_B'])
        
        bv22.add_equation("-8*1j*(2/beta)*Q**3*A22 + 2*1j*(2/beta)*Q*dx(A22x) + 4*1j*Q*u22 + 16*iR*Q**4*psi22 - 8*iR*Q**2*psi22xx + iR*dx(psi22xxx) = rhs22_psi")
        bv22.add_equation("2*1j*(2/beta)*Q*B22 + 2*1j*Q*(q-2)*psi22 - 4*iR*Q**2*u22 + iR*dx(u22x) = rhs22_u")
        bv22.add_equation("-iRm*4*Q**2*A22 + iRm*dx(A22x) + 2*1j*Q*psi22 = rhs22_A")
        bv22.add_equation("-2*1j*Q*q*A22 - iRm*4*Q**2*B22 + iRm*dx(B22x) + 2*1j*Q*u22 = rhs22_B")
        
        bv22.add_equation("dx(psi22) - psi22x = 0")
        bv22.add_equation("dx(psi22x) - psi22xx = 0")
        bv22.add_equation("dx(psi22xx) - psi22xxx = 0")      
        bv22.add_equation("dx(u22) - u22x = 0")
        bv22.add_equation("dx(A22) - A22x = 0")
        bv22.add_equation("dx(B22) - B22x = 0")

        # boundary conditions
        bv22.add_left_bc("psi22 = 0")
        bv22.add_right_bc("psi22 = 0")
        bv22.add_left_bc("u22 = 0")
        bv22.add_right_bc("u22 = 0")
        bv22.add_left_bc("A22 = 0")
        bv22.add_right_bc("A22 = 0")
        bv22.add_left_bc("psi22x = 0")
        bv22.add_right_bc("psi22x = 0")
        bv22.add_left_bc("B22x = 0")
        bv22.add_right_bc("B22x = 0")

        # parameters
        bv22.parameters['Q'] = self.Q
        bv22.parameters['iR'] = self.iR
        bv22.parameters['iRm'] = self.iRm
        bv22.parameters['q'] = self.q
        bv22.parameters['beta'] = self.beta
        bv22.parameters['rhs22_psi'] = rhs22_psi
        bv22.parameters['rhs22_u'] = rhs22_u
        bv22.parameters['rhs22_A'] = rhs22_A
        bv22.parameters['rhs22_B'] = rhs22_B
        
        self.BVP22 = self.solve_BVP(bv22)
        self.psi22 = self.BVP22.state['psi22']
        self.u22 = self.BVP22.state['u22']
        self.A22 = self.BVP22.state['A22']
        self.B22 = self.BVP22.state['B22']
        
        # These should be zero... 
        self.psi20['g'] = np.zeros(gridnum, np.complex_)
        self.B20['g'] = np.zeros(gridnum, np.complex_)
        
        # Take relevant derivatives and complex conjugates
        self.psi20_x = self.get_derivative(self.psi20)
        self.psi20_xx = self.get_derivative(self.psi20_x)
        self.psi20_xxx = self.get_derivative(self.psi20_xx)
        
        self.psi20_star = self.get_complex_conjugate(self.psi20)
        
        self.psi20_star_x = self.get_derivative(self.psi20_star)
        self.psi20_star_xx = self.get_derivative(self.psi20_star_x)
        self.psi20_star_xxx = self.get_derivative(self.psi20_star_xx)
        
        self.psi22_x = self.get_derivative(self.psi22)
        self.psi22_xx = self.get_derivative(self.psi22_x)
        self.psi22_xxx = self.get_derivative(self.psi22_xx)
        
        # u
        self.u20_x = self.get_derivative(self.u20)
        
        self.u20_star = self.get_complex_conjugate(self.u20)
        
        self.u20_star_x = self.get_derivative(self.u20_star)
        
        self.u22_x = self.get_derivative(self.u22)
        
        self.u22_star = self.get_complex_conjugate(self.u22)
        
        self.u22_star_x = self.get_derivative(self.u22_star)
        
        # B 
        self.B20_x = self.get_derivative(self.B20)
        
        self.B20_star = self.get_complex_conjugate(self.B20)
        
        self.B20_star_x = self.get_derivative(self.B20_star)
        
        self.B22_x = self.get_derivative(self.B22)
        
        self.B22_star = self.get_complex_conjugate(self.B22)
        
        self.B22_star_x = self.get_derivative(self.B22_star)
        
        # A 
        self.A20_x = self.get_derivative(self.A20)
        self.A20_xx = self.get_derivative(self.A20_x)
        self.A20_xxx = self.get_derivative(self.A20_xx)
        
        self.A20_star = self.get_complex_conjugate(self.A20)
        
        self.A20_star_x = self.get_derivative(self.A20_star)
        self.A20_star_xx = self.get_derivative(self.A20_star_x)
        self.A20_star_xxx = self.get_derivative(self.A20_star_xx)
        
        self.A22_x = self.get_derivative(self.A22)
        self.A22_xx = self.get_derivative(self.A22_x)
        self.A22_xxx = self.get_derivative(self.A22_xx)
        
        
        
class N3(MRI):

    """
    Solves the nonlinear vector N3
    Returns N3
    
    """
    
    def __init__(self, o1 = None, o2 = None, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        
        print("initializing N3")
        
        if o1 == None:
            o1 = OrderE(Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            n2 = N2(Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
        else:
            MRI.__init__(self, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
            n2 = N2(Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
    
        if o2 == None:
            o2 = OrderE2(o1 = o1, Q = self.Q, Rm = self.Rm, Pm = self.Pm, q = self.q, beta = self.beta, norm = self.norm)
        
        # Components of N31
        # psi component
        N31_psi_my1 = 1j*self.Q*(o1.psi*o2.psi20_xxx) + 1j*self.Q*(o1.psi*o2.psi20_star_xxx) - 1j*self.Q*(o1.psi_star*o2.psi22_xxx) - 1j*2*self.Q*(o1.psi_star_x*o2.psi22_xx) + 1j*8*self.Q**3*(o1.psi_star_x*o2.psi22) + 1j*4*self.Q**3*(o1.psi_star*o2.psi22_x)
        N31_psi_my2 = -1j*self.Q*(2/self.beta)*(o1.A*o2.A20_xxx) - 1j*self.Q*(2/self.beta)*(o1.A*o2.A20_star_xxx) + 1j*self.Q*(2/self.beta)*(o1.A_star*o2.A22_xxx) + 1j*2*self.Q*(2/self.beta)*(o1.A_star_x*o2.A22_xx) - 1j*8*self.Q**3*(2/self.beta)*(o1.A_star_x*o2.A22) - 1j*4*self.Q**3*(2/self.beta)*(o1.A_star*o2.A22_x)
        N31_psi_my3 = 1j*2*self.Q*(o2.psi22*o1.psi_star_xxx) - 1j*2*self.Q**3*(o2.psi22*o1.psi_star_x) - 1j*self.Q*(o2.psi20_x*o1.psi_xx) + 1j*self.Q*(o2.psi22_x*o1.psi_star_xx) - 1j*self.Q*(o2.psi20_star_x*o1.psi_xx) + 1j*self.Q**3*(o2.psi20_x*o1.psi) + 1j*self.Q**3*(o2.psi20_star_x*o1.psi) - 1j*self.Q**3*(o2.psi22_x*o1.psi_star)
        N31_psi_my4 = -1j*2*self.Q*(2/self.beta)*(o2.A22*o1.A_star_xxx) + 1j*2*self.Q**3*(2/self.beta)*(o2.A22*o1.A_star_x) + 1j*self.Q*(2/self.beta)*(o2.A20_x*o1.A_xx) - 1j*self.Q*(2/self.beta)*(o2.A22_x*o1.A_star_xx) + 1j*self.Q*(2/self.beta)*(o2.A20_star_x*o1.A_xx) - 1j*self.Q**3*(2/self.beta)*(o2.A20_x*o1.A) - 1j*self.Q**3*(2/self.beta)*(o2.A20_star_x*o1.A) + 1j*self.Q**3*(2/self.beta)*(o2.A22_x*o1.A_star)
        
        N31_psi = N31_psi_my1 + N31_psi_my2 + N31_psi_my3 +  N31_psi_my4
        
        self.N31_psi = N31_psi.evaluate()
        
        # u component
        N31_u_my1 = 1j*self.Q*(o1.psi*o2.u20_x) + 1j*self.Q*(o1.psi*o2.u20_star_x) - 1j*self.Q*(o1.psi_star*o2.u22_x) - 1j*2*self.Q*(o1.psi_star_x*o2.u22)
        N31_u_my2 = -1j*self.Q*(o1.u*o2.psi20_x) - 1j*self.Q*(o1.u*o2.psi20_star_x) + 1j*self.Q*(o1.u_star*o2.psi22_x) + 1j*2*self.Q*(o1.u_star_x*o2.psi22)
        N31_u_my3 = -1j*self.Q*(2/self.beta)*(o1.A*o2.B20_x) - 1j*self.Q*(2/self.beta)*(o1.A*o2.B20_star_x) + 1j*self.Q*(2/self.beta)*(o1.A_star*o2.B22_x) + 1j*2*self.Q*(2/self.beta)*(o1.A_star_x*o2.B22)
        N31_u_my4 = 1j*self.Q*(2/self.beta)*(o1.B*o2.A20_x) + 1j*self.Q*(2/self.beta)*(o1.B*o2.A20_star_x) - 1j*self.Q*(2/self.beta)*(o1.B_star*o2.A20_x) - 1j*2*self.Q*(2/self.beta)*(o1.B_star_x*o2.A22)
        
        N31_u = N31_u_my1 + N31_u_my2 + N31_u_my3 + N31_u_my4
        
        self.N31_u = N31_u.evaluate()
        
        # A component
        N31_A_my1 = -1j*self.Q*(o1.A*o2.psi20_x) - 1j*self.Q*(o1.A*o2.psi20_star_x) + 1j*self.Q*(o1.A_star*o2.psi22_x) + 1j*2*self.Q*(o1.A_star_x*o2.psi22)
        N31_A_my2 = 1j*self.Q*(o1.psi*o2.A20_x) + 1j*self.Q*(o1.psi*o2.A20_star_x) - 1j*self.Q*(o1.psi_star*o2.A22_x) - 1j*2*self.Q*(o1.psi_star_x*o2.A22)
        
        N31_A = N31_A_my1 + N31_A_my2
        
        self.N31_A = N31_A.evaluate()
        
        # B component
        N31_B_my1 = 1j*self.Q*(o1.psi*o2.B20_x) + 1j*self.Q*(o1.psi*o2.B20_star_x) - 1j*self.Q*(o1.psi_star*o2.B22_x) - 1j*2*self.Q*(o1.psi_star_x*o2.B22)
        N31_B_my2 = -1j*self.Q*(o1.B*o2.psi20_x) - 1j*self.Q*(o1.B*o2.psi20_star_x) + 1j*self.Q*(o1.B_star*o2.psi22_x) + 1j*2*self.Q*(o1.B_star_x*o2.psi22)
        N31_B_my3 = -1j*self.Q*(o1.A*o2.u20_x) - 1j*self.Q*(o1.A*o2.u20_star_x) + 1j*self.Q*(o1.A_star*o2.u22_x) + 1j*2*self.Q*(o1.A_star_x*o2.u22)
        N31_B_my4 = 1j*self.Q*(o1.u*o2.A20_x) + 1j*self.Q*(o1.u*o2.A20_star_x) - 1j*self.Q*(o1.u_star*o2.A22_x) - 1j*2*self.Q*(o1.u_star_x*o2.A22)
        
        N31_B = N31_B_my1 + N31_B_my2 + N31_B_my3 + N31_B_my4
        
        self.N31_B = N31_B.evaluate()
        
        
class AmplitudeAlpha(MRI):

    """
    Solves the coefficients of the first amplitude equation for alpha -- e^(iQz) terms.
    
    """
    
    def __init__(self, o1 = None, o2 = None, Q = 0.75, Rm = 4.8775, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        
        print("initializing Amplitude Alpha")
      
        if o1 == None:
            o1 = OrderE(Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            n2 = N2(Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
        else:
            MRI.__init__(self, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
            n2 = N2(Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
    
        if o2 == None:
            o2 = OrderE2(o1 = o1, Q = self.Q, Rm = self.Rm, Pm = self.Pm, q = self.q, beta = self.beta, norm = self.norm)
        
        n3 = N3(o1 = o1, o2 = o2, Q = self.Q, Rm = self.Rm, Pm = self.Pm, q = self.q, beta = self.beta, norm = self.norm)
        ah = AdjointHomogenous(Q = self.Q, Rm = self.Rm, Pm = self.Pm, q = self.q, beta = self.beta, norm = self.norm)
        
        self.x = domain.grid(0)
        
        a_psi_rhs = o1.psi_xx - self.Q**2*o1.psi
        a_psi_rhs = a_psi_rhs.evaluate()
        
        u20_twiddle = domain.new_field()
        u20_twiddle.name = 'self.v20_utwiddle'
        u20_twiddle['g'] = 0.5*(2/self.beta)*self.R*(self.x**2 - 1)
        
        allzeros = domain.new_field()
        allzeros['g'] = np.zeros(len(self.x), np.complex_)
        
        u20_twiddle_x = self.get_derivative(u20_twiddle)
        
        c_twiddle_u_rhs = (1j*self.Q*o1.psi)*u20_twiddle_x
        c_twiddle_u_rhs = c_twiddle_u_rhs.evaluate()
        
        c_twiddle_B_rhs = (-1j*self.Q*o1.psi)*(u20_twiddle_x)
        c_twiddle_B_rhs = c_twiddle_B_rhs.evaluate()
        
        b_psi_rhs = (2/self.beta)*o1.A_xx
        b_psi_rhs = b_psi_rhs.evaluate()
        
        # a = <va . D V11*>
        self.a = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [a_psi_rhs, o1.u, o1.A, o1.B])
        
        # c = <va . N31*>
        self.c = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [n3.N31_psi, n3.N31_u, n3.N31_A, n3.N31_B])
        
        # ctwiddle = < va . N31_twiddle_star >. Should be zero.
        self.ctwiddle = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [allzeros, c_twiddle_u_rhs, allzeros, c_twiddle_B_rhs])
        
        # b = < va . (X v11)* >
        self.b = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [b_psi_rhs, o1.B, o1.psi, o1.u])
  
        # h = < va . (L2twiddle v11 - L1twiddle v21)* >
    
    
    
    