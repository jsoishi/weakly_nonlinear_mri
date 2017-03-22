import numpy as np
from mpi4py import MPI
import dedalus.public as de
from eigentools import Eigenproblem
import random

import logging
logger = logging.getLogger(__name__)


class MRI():

    """
    Base class for MRI equations.
    
    Defaults: For Pm of 0.001 critical Rm is 4.879  critical Q is 0.748
    
    CriticalFinder gives:
    critical wavenumber alpha =    0.74514
    critical Re =    4.87903
    """

    def __init__(self, domain, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, B0 = 1.0, norm = True):

        self.domain = domain
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.q = q
        self.beta = beta
        self.norm = norm
        self.B0 = B0
        
        # Inverse magnetic reynolds number
        self.iRm = 1.0/self.Rm
        
        # Reynolds number
        self.R = self.Rm/self.Pm
        self.iR = 1.0/self.R
        
        self.gridnum = self.domain.bases[0].coeff_size
        self.x = self.domain.grid(0)
        
        if self.domain.bases[0].__class__.__name__ == "Chebyshev":
            self.setbcs = True
        else:
            self.setbcs = False
            logger.info("Not running with Chebyshev basis; will not set boundary conditions.")
        
        logger.info("MRI parameters: Q = {}; Rm = {}; Pm = {}; q = {}; beta = {}; norm = {}, Re = {}, B0 = {}".format(self.Q, self.Rm, self.Pm, self.q, self.beta, norm, self.R, self.B0))
        
        
    def set_boundary_conditions(self, problem, noslip=True):
        
        """
        Adds MRI problem boundary conditions to a ParsedProblem object.
        """
        
        if noslip:
            problem.add_bc("left(u) = 0")
            problem.add_bc("right(u) = 0")
            problem.add_bc("left(psi) = 0")
            problem.add_bc("right(psi) = 0")
            problem.add_bc("left(psix) = 0")
            problem.add_bc("right(psix) = 0")
            
            # magnetic b.c.'s
            problem.add_bc("left(A) = 0")
            problem.add_bc("right(A) = 0")
            problem.add_bc("left(Bx) = 0")
            problem.add_bc("right(Bx) = 0")
        
        # free-slip b.c.'s
        else:
            problem.add_bc("left(u) = 0")
            problem.add_bc("right(u) = 0")
            problem.add_bc("left(psi) = 0")
            problem.add_bc("right(psi) = 0")
            problem.add_bc("left(psixx) = 0")
            problem.add_bc("right(psixx) = 0")
        
            # magnetic b.c.'s
            problem.add_bc("left(A) = 0")
            problem.add_bc("right(A) = 0")
            problem.add_bc("left(B) = 0")
            problem.add_bc("right(B) = 0")
        
        return problem

    def fastest_growing(self):
        gr, largest_eval_indx,freq  = self.EP.growth_rate({})
        self.growth_rate = gr
        self.largest_eval_indx = largest_eval_indx
        self.EP.solver.set_state(largest_eval_indx)
    
    def solve_BVP(self, BVP):
    
        """
        Solves the boundary value problem for a BVP object.
        """
    
        solver = BVP.build_solver()
        solver.solve()
        
        return solver
        
    def normalize_all_real_or_imag(self, LEV):
        
        """
        Normalize state vectors such that they are purely real or purely imaginary.
        """
        
        n = np.abs(LEV.state['psi']['g'])[13]
        a = LEV.state['psi']['g'].real[13]/n
        b = LEV.state['psi']['g'].imag[13]/n
        scale = 1j*a/(b*(a**2/b+b)) + 1./(a**2/b +b)
        
        
        #HACK
        #scale = -1.0*scale
        
        
        return scale
        
    def normalize_all_real_or_imag_bystate(self, state):
        
        """
        Normalize state vectors such that they are purely real or purely imaginary.
        """
        
        n = np.abs(state['g'])[13]
        a = state['g'].real[13]/n
        b = state['g'].imag[13]/n
        scale = 1j*a/(b*(a**2/b+b)) + 1./(a**2/b +b)
        
        return scale
        
    def normalize_inner_product_eq_1(self, psi1, u1, A1, B1, psi2, u2, A2, B2):
    
        c_ip = self.take_inner_product([psi1, u1, A1, B1], [psi2, u2, A2, B2])
        normc = 1.0/(c_ip)
        psi1 = (psi1*normc).evaluate()
        u1 = (u1*normc).evaluate()
        A1 = (A1*normc).evaluate()
        B1 = (B1*normc).evaluate()
        
        return psi1, u1, A1, B1
        
    def normalize_vector(self, evector):
        
        """
        Normalize state vectors.
        """
        
        evector = evector/np.linalg.norm(evector)
        #evector = evector/np.nanmax(np.abs(evector))
        
        return evector
        
    def normalize_state_vector(self, psi, u, A, B):
        
        """
        Normalize total state vector.
        """
        logger.warn("norm hack: Using max(A) from URM07")

        # this value read from A(x = 0) figure 2c of Umurhan, Regev, &
        # Menou (2007) using WebPlotDigitizer. I estimate the error to
        # be +0.03/-0.04.
        Amax = 0.535#1.0#0.535#*2
        norm = A.interpolate(x = 0)['g'][0]/Amax
        
        #logger.warn('normalizing by int u instead')
        #intu = u.integrate('x')
        #norm = intu['g'][0]
        #logger.warn("Normalizing by {}".format(norm))
        
        psi['g'] = psi['g']/norm
        u['g'] = u['g']/norm
        A['g'] = A['g']/norm
        B['g'] = B['g']/norm
        
        self.normconst = norm
        
        return psi, u, A, B

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
        
        field_star = self.domain.new_field()
        field_star.name = field.name + "_star"
        field_star['g'] = field['g'].conj()
        
        return field_star
        
    def take_inner_product(self, vector1, vector2):
        
        """
        Take inner product < vector2 | vector1 > 
        Defined as integral of (vector2.conj * vector1)
        """
        
        inner_product = vector1[0]['g']*vector2[0]['g'].conj() + vector1[1]['g']*vector2[1]['g'].conj() + vector1[2]['g']*vector2[2]['g'].conj() + vector1[3]['g']*vector2[3]['g'].conj()
        
        ip = self.domain.new_field()
        ip.name = "inner product"
        ip['g'] = inner_product
        ip = ip.integrate('x')
        ip = ip['g'][0] 
        
        return ip
        
    def take_inner_product_real(self, vector1, vector2):
        
        """
        Take the real inner product < vector2 | vector1 > 
        Defined as integral of (vector2.conj * vector1) + c.c.
        """
        ip = self.take_inner_product(vector1, vector2)
        return ip + ip.conj()
    
class AdjointHomogenous(MRI):

    """
    Solves the adjoint homogenous equation L^dagger V^dagger = 0
    Returns V^dagger
    """

    def __init__(self, domain, o1 = None, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True, finalize=True, noslip=True):
        
        logger.info("initializing Adjoint Homogenous")
        
        if o1 == None:
            self.o1 = OrderE(domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
        else:
            MRI.__init__(self, domain, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
      
        # Set up problem object
        lv1 = de.EVP(self.domain,
                     ['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'], 'sigma')

        lv1.parameters['Q'] = self.Q
        lv1.parameters['iR'] = self.iR
        lv1.parameters['iRm'] = self.iRm
        lv1.parameters['q'] = self.q
        lv1.parameters['beta'] = self.beta

        # JSO: these are correct with my rederivation.
        #wait - adjoint should be taken as (L.subs(dz, 1j*Q)).adjoint() -- the following are correct.
        lv1.add_equation("-sigma*Q**2*psi + sigma*psixx + 1j*Q*A + 1j*(q - 2)*Q*u - iR*Q**4*psi + 2*iR*Q**2*psixx - iR*dx(psixxx) = 0")
        lv1.add_equation("sigma*u + 1j*Q*B + 2*1j*Q*psi + iR*Q**2*u - iR*dx(ux) = 0")
        lv1.add_equation("sigma*A + iRm*Q**2*A - iRm*dx(Ax) - 1j*Q*q*B - 1j*(2/beta)*Q**3*psi + 1j*(2/beta)*Q*psixx = 0")
        lv1.add_equation("sigma*B + iRm*Q**2*B - iRm*dx(Bx) + 1j*(2/beta)*Q*u = 0")
        
        lv1.add_equation("dx(psi) - psix = 0")
        lv1.add_equation("dx(psix) - psixx = 0")
        lv1.add_equation("dx(psixx) - psixxx = 0")
        lv1.add_equation("dx(u) - ux = 0")
        lv1.add_equation("dx(A) - Ax = 0")
        lv1.add_equation("dx(B) - Bx = 0")
        
        self.lv1 = lv1

        # Set boundary conditions for MRI problem
        if self.setbcs:
            self.lv1 = self.set_boundary_conditions(lv1, noslip=noslip)
        self.EP = Eigenproblem(self.lv1)

        if finalize:
            self.finalize()
            
    def finalize(self):
        self.fastest_growing()
                
        self.psi = self.EP.solver.state['psi']
        self.u = self.EP.solver.state['u']
        self.A = self.EP.solver.state['A']
        self.B = self.EP.solver.state['B']
               
        if self.norm == True:
            scale = self.normalize_all_real_or_imag(self.EP.solver)
            
            self.psi = (self.psi*scale).evaluate()
            self.u = (self.u*scale).evaluate()
            self.A = (self.A*scale).evaluate()
            self.B = (self.B*scale).evaluate()
            
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

    def __init__(self, domain, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, B0 = 1.0, norm = True, finalize=True, noslip=True):
        
        logger.info("initializing Order E")
        
        MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm, B0 = B0)
        
        lv1 = de.EVP(self.domain,
                     ['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],'sigma')
        
        lv1.parameters['Q'] = self.Q
        lv1.parameters['iR'] = self.iR
        lv1.parameters['iRm'] = self.iRm
        lv1.parameters['q'] = self.q
        lv1.parameters['beta'] = self.beta
        lv1.parameters['B0'] = self.B0
        
        lv1.add_equation("sigma*psixx - sigma*Q**2*psi - iR*dx(psixxx) + 2*iR*Q**2*psixx - iR*Q**4*psi - 2*1j*Q*u - B0*(2/beta)*1j*Q*dx(Ax) + B0*(2/beta)*Q**3*1j*A = 0")
        lv1.add_equation("sigma*u - iR*dx(ux) + iR*Q**2*u - (q - 2)*1j*Q*psi - B0*(2/beta)*1j*Q*B = 0") 
        lv1.add_equation("sigma*A - iRm*dx(Ax) + iRm*Q**2*A - B0*1j*Q*psi = 0") 
        lv1.add_equation("sigma*B - iRm*dx(Bx) + iRm*Q**2*B - B0*1j*Q*u + q*1j*Q*A = 0")
        
        lv1.add_equation("dx(psi) - psix = 0")
        lv1.add_equation("dx(psix) - psixx = 0")
        lv1.add_equation("dx(psixx) - psixxx = 0")
        lv1.add_equation("dx(u) - ux = 0")
        lv1.add_equation("dx(A) - Ax = 0")
        lv1.add_equation("dx(B) - Bx = 0")
        
        self.lv1 = lv1

        if self.setbcs:
            self.lv1 = self.set_boundary_conditions(lv1, noslip=noslip)
        self.EP = Eigenproblem(self.lv1)
        if finalize:
            self.finalize()

    def finalize(self):
        self.fastest_growing()        
        
        # All eigenfunctions must be scaled s.t. their max is 1
        self.psi = self.EP.solver.state['psi']
        self.u = self.EP.solver.state['u']
        self.A = self.EP.solver.state['A']
        self.B = self.EP.solver.state['B']
        
        self.prenormpsi = self.psi
        
        if self.norm == True:
            scale = self.normalize_all_real_or_imag(self.EP.solver)
            
            self.psi = (self.psi*scale).evaluate()
            self.u = (self.u*scale).evaluate()
            self.A = (self.A*scale).evaluate()
            self.B = (self.B*scale).evaluate()
            
            self.psi, self.u, self.A, self.B = self.normalize_state_vector(self.psi, self.u, self.A, self.B)
        
            self.normpsi = self.psi
        else:
            self.normconst = None
            
        self.psi.name = "psi"
        self.u.name = "u"
        self.A.name = "A"
        self.B.name = "B"
            
        # Take all relevant derivates for use with higher order terms
        self.psi_x = self.get_derivative(self.psi)
        self.psi_xx = self.get_derivative(self.psi_x)
        self.psi_xxx = self.get_derivative(self.psi_xx)
      
        self.u_x = self.get_derivative(self.u)
        
        # relevant for alternate O2 calculation
        self.u_xx = self.get_derivative(self.u_x)
        
        self.A_x = self.get_derivative(self.A)
        self.A_xx = self.get_derivative(self.A_x)
        self.A_xxx = self.get_derivative(self.A_xx)
        
        self.B_x = self.get_derivative(self.B)
        
        # relevant for alternate O2 calculation
        self.B_xx = self.get_derivative(self.B_x)
        
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
    
    def __init__(self, domain, o1 = None, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
    
        logger.info("initializing N2")
    
        if o1 is None:
            o1 = OrderE(domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
        else:
            MRI.__init__(self, domain, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
    
        N22psi = 1j*self.Q*o1.psi*(o1.psi_xxx - self.Q**2*o1.psi_x) - o1.psi_x*(1j*self.Q*o1.psi_xx - 1j*self.Q**3*o1.psi) + (2/self.beta)*o1.A_x*(1j*self.Q*o1.A_xx - 1j*self.Q**3*o1.A) - (2/self.beta)*1j*self.Q*o1.A*(o1.A_xxx - self.Q**2*o1.A_x) # Confirmed 11/2/15
        self.N22_psi = N22psi.evaluate()
        self.N22_psi.name = "N22_psi"
    
        N20psi = 1j*self.Q*o1.psi*(o1.psi_star_xxx - self.Q**2*o1.psi_star_x) - o1.psi_x*(-1j*self.Q*o1.psi_star_xx + 1j*self.Q**3*o1.psi_star) + (2/self.beta)*o1.A_x*(-1j*self.Q*o1.A_star_xx + 1j*self.Q**3*o1.A_star) - (2/self.beta)*1j*self.Q*o1.A*(o1.A_star_xxx - self.Q**2*o1.A_star_x) # Confirmed 11/2/15

        self.N20_psi = N20psi.evaluate()
        self.N20_psi.name = "N20_psi"
        
        N22u = 1j*self.Q*o1.psi*o1.u_x - o1.psi_x*1j*self.Q*o1.u - (2/self.beta)*1j*self.Q*o1.A*o1.B_x + (2/self.beta)*o1.A_x*1j*self.Q*o1.B # confirmed 10/30/15
        self.N22_u = N22u.evaluate()
        self.N22_u.name = "N22_u"
        
        N20u = 1j*self.Q*o1.psi*o1.u_star_x + o1.psi_x*1j*self.Q*o1.u_star - (2/self.beta)*1j*self.Q*o1.A*o1.B_star_x - (2/self.beta)*o1.A_x*1j*self.Q*o1.B_star # confirmed 10/30/15

        self.N20_u = N20u.evaluate()
        self.N20_u.name = "N20_u"
        
        N22A = -1j*self.Q*o1.A*o1.psi_x + o1.A_x*1j*self.Q*o1.psi # confirmed 1/2/15
        self.N22_A = N22A.evaluate()
        self.N22_A.name = "N22_A"
        
        N20A = -1j*self.Q*o1.A*o1.psi_star_x - o1.A_x*1j*self.Q*o1.psi_star # confirmed 1/2/15
        self.N20_A = N20A.evaluate()
        self.N20_A.name = "N20_A"

        N22B = 1j*self.Q*o1.psi*o1.B_x - o1.psi_x*1j*self.Q*o1.B - 1j*self.Q*o1.A*o1.u_x + o1.A_x*1j*self.Q*o1.u # confirmed 1/2/15
        self.N22_B = N22B.evaluate()
        self.N22_B.name = "N22_B"
        
        N20B = 1j*self.Q*o1.psi*o1.B_star_x + o1.psi_x*1j*self.Q*o1.B_star - 1j*self.Q*o1.A*o1.u_star_x - o1.A_x*1j*self.Q*o1.u_star # confirmed 1/2/15

        self.N20_B = N20B.evaluate()
        self.N20_B.name = "N20_B"
        
class OrderE2(MRI):

    """
    Solves the second order equation L V2 = N2 - Ltwiddle V1
    Returns V2
    
    """
    
    def __init__(self, domain, o1 = None, ah = None, n2 = None, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
    
        logger.info("initializing Order E2")
        
        if o1 is None:
            o1 = OrderE(domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            n2 = N2(domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
        else:
            MRI.__init__(self, domain, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
            #n2 = N2(domain, o1 = o1, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
        
        self.o1 = o1
        self.n2 = n2
        # righthand side for the 20 terms (e^0)
        # need N20 + N20* on RHS
        N20_psi_cc = self.domain.new_field()
        N20_psi_cc['g'] = n2.N20_psi['g'].conj()
        N20_u_cc = self.domain.new_field()
        N20_u_cc['g'] = n2.N20_u['g'].conj()
        N20_A_cc = self.domain.new_field()
        N20_A_cc['g'] = n2.N20_A['g'].conj()
        N20_B_cc = self.domain.new_field()
        N20_B_cc['g'] = n2.N20_B['g'].conj()

        rhs20_psi = n2.N20_psi + N20_psi_cc
        rhs20_u = n2.N20_u + N20_u_cc
        rhs20_A = n2.N20_A + N20_A_cc
        rhs20_B = n2.N20_B + N20_B_cc

        # V20 equations are separable because dz terms -> 0
        bv20psi = de.LBVP(self.domain, ['psi20', 'psi20x', 'psi20xx', 'psi20xxx'])
        
        bv20psi.parameters['iR'] = self.iR
        bv20psi.parameters['rhs20_psi'] = rhs20_psi 
        bv20psi.add_equation("iR*dx(psi20xxx) = rhs20_psi")
        bv20psi.add_equation("dx(psi20) - psi20x = 0")
        bv20psi.add_equation("dx(psi20x) - psi20xx = 0")
        bv20psi.add_equation("dx(psi20xx) - psi20xxx = 0")
        
        if self.setbcs:
            bv20psi.add_bc("left(psi20) = 0")
            bv20psi.add_bc("right(psi20) = 0")
            bv20psi.add_bc("left(psi20x) = 0")
            bv20psi.add_bc("right(psi20x) = 0")
        
        bv20u = de.LBVP(self.domain, ['u20', 'u20x'])
        bv20u.parameters['iR'] = self.iR
        bv20u.parameters['rhs20_u'] = rhs20_u
        
        bv20u.add_equation("iR*dx(u20x) = rhs20_u")
        bv20u.add_equation("dx(u20) - u20x = 0")
        
        if self.setbcs:
            bv20u.add_bc("left(u20) = 0")
            bv20u.add_bc("right(u20) = 0")
        
        bv20A = de.LBVP(self.domain, ['A20', 'A20x'])
        bv20A.parameters['iRm'] = self.iRm
        bv20A.parameters['rhs20_A'] = rhs20_A
        
        bv20A.add_equation("iRm*dx(A20x) = rhs20_A")
        bv20A.add_equation("dx(A20) - A20x = 0")
        
        if self.setbcs:
            bv20A.add_bc("left(A20) = 0")
            bv20A.add_bc("right(A20) = 0")
        
        bv20B = de.LBVP(self.domain,['B20', 'B20x'])
        bv20B.parameters['iRm'] = self.iRm
        bv20B.parameters['rhs20_B'] = rhs20_B
        
        bv20B.add_equation("iRm*dx(B20x) = rhs20_B")
        bv20B.add_equation("dx(B20) - B20x = 0")
        
        if self.setbcs:
            bv20B.add_bc("left(B20x) = 0")
            bv20B.add_bc("right(B20x) = 0")
        
        self.BVPpsi = self.solve_BVP(bv20psi)
        self.psi20 = self.BVPpsi.state['psi20']

        self.BVPu = self.solve_BVP(bv20u)
        self.u20 = self.BVPu.state['u20']

        self.BVPA = self.solve_BVP(bv20A)
        self.A20 = self.BVPA.state['A20']

        self.BVPB = self.solve_BVP(bv20B)
        self.B20 = self.BVPB.state['B20']
        self.B20['g'] = 0.
        
        # V21 equations are coupled
        # second term: -L1twiddle V1
        term2_psi = 3*(2/self.beta)*self.Q**2*o1.A - (2/self.beta)*o1.A_xx + 4*self.iR*1j*self.Q**3*o1.psi - 4*self.iR*1j*self.Q*o1.psi_xx - 2*o1.u
        self.term2_psi = term2_psi.evaluate()
        term2_psi = term2_psi.evaluate()
        
        #term2_u = (2/self.beta)*o1.B + 2*self.iR*self.Q*o1.u + (self.q - 2)*o1.psi ## why does (q - 2) and (2 - q) make no diff here??
        term2_u = -(2/self.beta)*o1.B - 2*self.iR*1j*self.Q*o1.u - (self.q - 2)*o1.psi #added missing 1j in second term 10/14/15
        self.term2_u = term2_u.evaluate()
        term2_u = term2_u.evaluate()
        
        term2_A = -2*self.iRm*1j*self.Q*o1.A - o1.psi
        self.term2_A = term2_A.evaluate()
        term2_A = term2_A.evaluate()
        
        term2_B = self.q*o1.A - 2*self.iRm*1j*self.Q*o1.B - o1.u
        self.term2_B = term2_B.evaluate()
        term2_B = term2_B.evaluate()
        
        # righthand side for the 21 terms (e^iQz dependence)
        rhs21_psi = term2_psi
        rhs21_u = term2_u
        rhs21_A = term2_A
        rhs21_B = term2_B
        
        # These RHS terms must satisfy the solvability condition <V^dagger | RHS> = 0. Test that:
        if ah == False:
            self.ah = AdjointHomogenous(o1 = o1, Q = self.Q, Rm = self.Rm, Pm = self.Pm, q = self.q, beta = self.beta, norm = self.norm)
        else:
            self.ah = ah
        
        sctest = self.take_inner_product_real((term2_psi, term2_u, term2_A, term2_B),(self.ah.psi, self.ah.u, self.ah.A, self.ah.B))
        logger.info("solvability condition satisfied? sctest = {}".format(sctest))
        if np.abs(sctest) > 1E-10:
            logger.warn("CAUTION: solvability condition <V^dagger | RHS> = 0 failed for V21")
                
        # define problem using righthand side as nonconstant coefficients
        
        bv21 = de.LBVP(self.domain,
              ['psi21', 'psi21x', 'psi21xx', 'psi21xxx', 'u21', 'u21x', 'A21', 'A21x', 'B21', 'B21x'])
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
        
        bv21.add_equation("-1j*(2/beta)*Q**3*A21 + 1j*(2/beta)*Q*dx(A21x) + 2*1j*Q*u21 + iR*Q**4*psi21 - 2*iR*Q**2*psi21xx + iR*dx(psi21xxx) = rhs21_psi")
        bv21.add_equation("1j*(2/beta)*Q*B21 + 1j*Q*(q - 2)*psi21 - iR*Q**2*u21 + iR*dx(u21x) = rhs21_u") 
        bv21.add_equation("-iRm*Q**2*A21 + iRm*dx(A21x) + 1j*Q*psi21 = rhs21_A")
        bv21.add_equation("-1j*Q*q*A21 - iRm*Q**2*B21 + iRm*dx(B21x) + 1j*Q*u21 = rhs21_B")   
        
        bv21.add_equation("dx(psi21) - psi21x = 0")
        bv21.add_equation("dx(psi21x) - psi21xx = 0")
        bv21.add_equation("dx(psi21xx) - psi21xxx = 0")
        bv21.add_equation("dx(u21) - u21x = 0")
        bv21.add_equation("dx(A21) - A21x = 0")
        bv21.add_equation("dx(B21) - B21x = 0")

        # boundary conditions
        if self.setbcs:
            bv21.add_bc("left(psi21) = 0")
            bv21.add_bc("right(psi21) = 0")
            bv21.add_bc("left(u21) = 0")
            bv21.add_bc("right(u21) = 0")
            bv21.add_bc("left(A21) = 0")
            bv21.add_bc("right(A21) = 0")
            bv21.add_bc("left(psi21x) = 0")
            bv21.add_bc("right(psi21x) = 0")
            bv21.add_bc("left(B21x) = 0")
            bv21.add_bc("right(B21x) = 0")

        self.BVP21 = self.solve_BVP(bv21)
        self.psi21 = self.BVP21.state['psi21']
        self.u21 = self.BVP21.state['u21']
        self.A21 = self.BVP21.state['A21']
        self.B21 = self.BVP21.state['B21']
        
        #V22 equations are coupled
        rhs22_psi = n2.N22_psi
        rhs22_u = n2.N22_u
        rhs22_A = n2.N22_A
        rhs22_B = n2.N22_B
        
        self.rhs22_psi = rhs22_psi
        self.rhs22_u = rhs22_u
        self.rhs22_A = rhs22_A
        self.rhs22_B = rhs22_B
        
        # define problem using righthand side as nonconstant coefficients
        bv22 = de.LBVP(self.domain,
              ['psi22', 'psi22x', 'psi22xx', 'psi22xxx', 'u22', 'u22x', 'A22', 'A22x', 'B22', 'B22x'])
        
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
        if self.setbcs:
            bv22.add_bc("left(psi22) = 0")
            bv22.add_bc("right(psi22) = 0")
            bv22.add_bc("left(u22) = 0")
            bv22.add_bc("right(u22) = 0")
            bv22.add_bc("left(A22) = 0")
            bv22.add_bc("right(A22) = 0")
            bv22.add_bc("left(psi22x) = 0")
            bv22.add_bc("right(psi22x) = 0")
            bv22.add_bc("left(B22x) = 0")
            bv22.add_bc("right(B22x) = 0")
        
        self.BVP22 = self.solve_BVP(bv22)
        self.psi22 = self.BVP22.state['psi22']
        self.u22 = self.BVP22.state['u22']
        self.A22 = self.BVP22.state['A22']
        self.B22 = self.BVP22.state['B22']
        
        self.psi20.name = "psi20"
        self.u20.name = "u20"
        self.A20.name = "A20"
        self.B20.name = "B20"
            
        self.psi21.name = "psi21"
        self.u21.name = "u21"
        self.A21.name = "A21"
        self.B21.name = "B21"
                
        self.psi22.name = "psi22"
        self.u22.name = "u22"
        self.A22.name = "A22"
        self.B22.name = "B22"  
        
        # Take relevant derivatives and complex conjugates
        self.psi20_x = self.get_derivative(self.psi20)
        self.psi20_xx = self.get_derivative(self.psi20_x)
        self.psi20_xxx = self.get_derivative(self.psi20_xx)
        
        self.psi20_star = self.get_complex_conjugate(self.psi20)
        
        self.psi20_star_x = self.get_derivative(self.psi20_star)
        self.psi20_star_xx = self.get_derivative(self.psi20_star_x)
        self.psi20_star_xxx = self.get_derivative(self.psi20_star_xx)
        
        self.psi21_x = self.get_derivative(self.psi21)
        self.psi21_xx = self.get_derivative(self.psi21_x)
        
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
        
        self.A21_x = self.get_derivative(self.A21)
        self.A21_xx = self.get_derivative(self.A21_x)
        
        self.A22_x = self.get_derivative(self.A22)
        self.A22_xx = self.get_derivative(self.A22_x)
        self.A22_xxx = self.get_derivative(self.A22_xx)
        
       
class N3(MRI):

    """
    Solves the nonlinear vector N3
    Returns N3
    
    """
    
    def __init__(self, domain, o1 = None, o2 = None, ah = None, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        
        logger.info("initializing N3")
        
        if o1 == None:
            o1 = OrderE(domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            n2 = N2(domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
        else:
            MRI.__init__(self, domain, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
            n2 = N2(domain, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)

        if ah == None:
            ah = AdjointHomogenous(domain, o1 = o1, Q = self.Q, Rm = self.Rm, Pm = self.Pm, q = self.q, beta = self.beta, norm = self.norm)

        if o2 == None:
            o2 = OrderE2(domain, o1 = o1, ah = ah, Q = self.Q, Rm = self.Rm, Pm = self.Pm, q = self.q, beta = self.beta, norm = self.norm)
        
        # Components of N31
        # psi component
        N31_psi_my1 = 1j*self.Q*(o1.psi*o2.psi20_xxx) + 1j*self.Q*(o1.psi*o2.psi20_star_xxx) - 1j*self.Q*(o1.psi_star*o2.psi22_xxx) - 1j*2*self.Q*(o1.psi_star_x*o2.psi22_xx) + 1j*8*self.Q**3*(o1.psi_star_x*o2.psi22) + 1j*4*self.Q**3*(o1.psi_star*o2.psi22_x)
        N31_psi_my2 = -1j*self.Q*(2/self.beta)*(o1.A*o2.A20_xxx) - 1j*self.Q*(2/self.beta)*(o1.A*o2.A20_star_xxx) + 1j*self.Q*(2/self.beta)*(o1.A_star*o2.A22_xxx) + 1j*2*self.Q*(2/self.beta)*(o1.A_star_x*o2.A22_xx) - 1j*8*self.Q**3*(2/self.beta)*(o1.A_star_x*o2.A22) - 1j*4*self.Q**3*(2/self.beta)*(o1.A_star*o2.A22_x)
        N31_psi_my3 = 1j*2*self.Q*(o2.psi22*o1.psi_star_xxx) - 1j*2*self.Q**3*(o2.psi22*o1.psi_star_x) - 1j*self.Q*(o2.psi20_x*o1.psi_xx) + 1j*self.Q*(o2.psi22_x*o1.psi_star_xx) - 1j*self.Q*(o2.psi20_star_x*o1.psi_xx) + 1j*self.Q**3*(o2.psi20_x*o1.psi) + 1j*self.Q**3*(o2.psi20_star_x*o1.psi) - 1j*self.Q**3*(o2.psi22_x*o1.psi_star)
        N31_psi_my4 = -1j*2*self.Q*(2/self.beta)*(o2.A22*o1.A_star_xxx) + 1j*2*self.Q**3*(2/self.beta)*(o2.A22*o1.A_star_x) + 1j*self.Q*(2/self.beta)*(o2.A20_x*o1.A_xx) - 1j*self.Q*(2/self.beta)*(o2.A22_x*o1.A_star_xx) + 1j*self.Q*(2/self.beta)*(o2.A20_star_x*o1.A_xx) - 1j*self.Q**3*(2/self.beta)*(o2.A20_x*o1.A) - 1j*self.Q**3*(2/self.beta)*(o2.A20_star_x*o1.A) + 1j*self.Q**3*(2/self.beta)*(o2.A22_x*o1.A_star)
        
        
        N31_psi = N31_psi_my1 + N31_psi_my2 + N31_psi_my3 +  N31_psi_my4
        
        self.N31_psi = N31_psi.evaluate()
        
        # diagnostics reveal that my_2 and my_4 are nan with the new normalizations. A_star? A22?
        #print(self.N31_psi['g'])
        p1 = N31_psi_my1.evaluate()
        p2 = N31_psi_my2.evaluate()
        p3 = N31_psi_my3.evaluate()
        p4 = N31_psi_my4.evaluate()
        
        # u component
        N31_u_my1 = 1j*self.Q*(o1.psi*o2.u20_x) + 1j*self.Q*(o1.psi*o2.u20_star_x) - 1j*self.Q*(o1.psi_star*o2.u22_x) - 1j*2*self.Q*(o1.psi_star_x*o2.u22)
        N31_u_my2 = -1j*self.Q*(o1.u*o2.psi20_x) - 1j*self.Q*(o1.u*o2.psi20_star_x) + 1j*self.Q*(o1.u_star*o2.psi22_x) + 1j*2*self.Q*(o1.u_star_x*o2.psi22)
        N31_u_my3 = -1j*self.Q*(2/self.beta)*(o1.A*o2.B20_x) - 1j*self.Q*(2/self.beta)*(o1.A*o2.B20_star_x) + 1j*self.Q*(2/self.beta)*(o1.A_star*o2.B22_x) + 1j*2*self.Q*(2/self.beta)*(o1.A_star_x*o2.B22)
        N31_u_my4 = 1j*self.Q*(2/self.beta)*(o1.B*o2.A20_x) + 1j*self.Q*(2/self.beta)*(o1.B*o2.A20_star_x) - 1j*self.Q*(2/self.beta)*(o1.B_star*o2.A20_x) - 1j*2*self.Q*(2/self.beta)*(o1.B_star_x*o2.A22)
        
        
        N31_u = N31_u_my1 + N31_u_my2 + N31_u_my3 + N31_u_my4
        
        self.N31_u = N31_u.evaluate()
        
        # A component -- correct with all-positive V2 definition. Checked 11/14/15
        N31_A_my1 = -1j*self.Q*(o1.A*o2.psi20_x) - 1j*self.Q*(o1.A*o2.psi20_star_x) + 1j*self.Q*(o1.A_star*o2.psi22_x) + 1j*2*self.Q*(o1.A_star_x*o2.psi22)
        N31_A_my2 = 1j*self.Q*(o1.psi*o2.A20_x) + 1j*self.Q*(o1.psi*o2.A20_star_x) - 1j*self.Q*(o1.psi_star*o2.A22_x) - 1j*2*self.Q*(o1.psi_star_x*o2.A22)
        
        N31_A = N31_A_my1 + N31_A_my2
        
        self.N31_A = N31_A.evaluate()
        
        # B component -- correct with all-positive V2 definition. Checked 11/6/15
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
    
    def __init__(self, domain, o1 = None, o2 = None, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        
        logger.info("initializing Amplitude Alpha")
      
        if o1 == None:
            o1 = OrderE(domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            n2 = N2(domain, o1 = o1, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
        else:
            MRI.__init__(self, domain, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
            n2 = N2(domain, o1 = o1, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)

        ah = AdjointHomogenous(domain, o1 = o1, Q = self.Q, Rm = self.Rm, Pm = self.Pm, q = self.q, beta = self.beta, norm = self.norm)
        if o2 == None:
            o2 = OrderE2(domain, o1 = o1, ah=ah, n2=n2, Q = self.Q, Rm = self.Rm, Pm = self.Pm, q = self.q, beta = self.beta, norm = self.norm)
        
        n3 = N3(domain, o1 = o1, o2 = o2, ah=ah, Q = self.Q, Rm = self.Rm, Pm = self.Pm, q = self.q, beta = self.beta, norm = self.norm)

        
        magicnumberhack = False
        logger.info("magicnumberhack is {}".format(magicnumberhack))
        if magicnumberhack == True:
            o1scale = -0.55
            o1.psi['g'] = o1.psi['g']*o1scale
            o1.u['g'] = o1.u['g']*o1scale
            o1.A['g'] = o1.A['g']*o1scale
            o1.B['g'] = o1.B['g']*o1scale
            
        self.x = self.domain.grid(0)
        
        a_psi_rhs = o1.psi_xx - self.Q**2*o1.psi
        a_psi_rhs = a_psi_rhs.evaluate()
        
        a_psi_rhs2 = o1.psi_star_xx - self.Q**2*o1.psi #test
        a_psi_rhs2 = a_psi_rhs2.evaluate()
        
        u20_twiddle = self.domain.new_field()
        u20_twiddle.name = 'self.v20_utwiddle'
        u20_twiddle['g'] = 0.5*(2/self.beta)*self.R*(self.x**2 - 1)
        
        allzeros = self.domain.new_field()
        allzeros['g'] = np.zeros(len(self.x), np.complex_)
        
        u20_twiddle_x = self.get_derivative(u20_twiddle)
        
        c_twiddle_u_rhs = (1j*self.Q*o1.psi)*u20_twiddle_x
        c_twiddle_u_rhs = c_twiddle_u_rhs.evaluate()
        
        c_twiddle_B_rhs = (-1j*self.Q*o1.psi)*(u20_twiddle_x)
        c_twiddle_B_rhs = c_twiddle_B_rhs.evaluate()
        
        # Old definition of b
        #b_psi_rhs = (2/self.beta)*o1.A_xx
        #b_psi_rhs = b_psi_rhs.evaluate()
        
        # New definition :: b = <Gtwiddle | V+ >
        b_psi_rhs = 1j*self.Q**3*(2/self.beta)*o1.A - 1j*self.Q*(2/self.beta)*o1.A_xx
        b_psi_rhs = b_psi_rhs.evaluate()
        
        b_u_rhs = -1j*self.Q*(2/self.beta)*o1.B
        b_u_rhs = b_u_rhs.evaluate()
        
        b_A_rhs = -1j*self.Q*o1.psi
        b_A_rhs = b_A_rhs.evaluate()
        
        b_B_rhs = -1j*self.Q*o1.u
        b_B_rhs = b_B_rhs.evaluate()
                
        l2twiddlel1twiddle_psi = 3*1j*(2/self.beta)*self.Q*o1.A - 3*(2/self.beta)*self.Q**2*o2.A21 + (2/self.beta)*o2.A21_xx - 6*self.Q**2*self.iR*o1.psi + 2*self.iR*o1.psi_xx - 4*1j*self.iR*self.Q**3*o2.psi21 + 4*self.iR*1j*self.Q*o2.psi21_xx + 2*o2.u21
        l2twiddlel1twiddle_psi = l2twiddlel1twiddle_psi.evaluate()
        
        l2twiddlel1twiddle_psi_URM07 = 6*1j*(2/self.beta)*self.Q*o1.A - 3*(2/self.beta)*self.Q**2*o2.A21 + (2/self.beta)*o2.A21_xx - 12*self.Q**2*self.iR*o1.psi + 4*self.iR*o1.psi_xx - 4*1j*self.iR*self.Q**3*o2.psi21 + 4*self.iR*1j*self.Q*o2.psi21_xx + 2*o2.u21 # Umurhan+'s wrong definition of L2twiddle
        l2twiddlel1twiddle_psi_URM07 = l2twiddlel1twiddle_psi_URM07.evaluate()
        
        #l2twiddlel1twiddle_u = (2/self.beta)*o1.B - 1j*self.Q*(2/self.beta)*o2.B21 - 1j*self.Q*(self.q - 2)*o2.psi21 + self.iR*o1.u #what?
        l2twiddlel1twiddle_u = (2/self.beta)*o2.B21 + 2*1j*self.iR*self.Q*o2.u21 + (self.q - 2)*o2.psi21 + self.iR*o1.u #correct
        l2twiddlel1twiddle_u = l2twiddlel1twiddle_u.evaluate()
        
        l2twiddlel1twiddle_u_URM07 = (2/self.beta)*o2.B21 + 2*1j*self.iR*self.Q*o2.u21 + (self.q - 2)*o2.psi21 + 2*self.iR*o1.u #Umurhan+'s wrong def of L2twiddle
        l2twiddlel1twiddle_u_URM07 = l2twiddlel1twiddle_u_URM07.evaluate()
        
        l2twiddlel1twiddle_A = self.iRm*o1.A + 2*1j*self.iRm*self.Q*o2.A21 + o2.psi21
        l2twiddlel1twiddle_A = l2twiddlel1twiddle_A.evaluate()
        
        l2twiddlel1twiddle_A_URM07 = 2*self.iRm*o1.A + 2*1j*self.iRm*self.Q*o2.A21 + o2.psi21 #Umurhan+'s wrong def of L2twiddle
        l2twiddlel1twiddle_A_URM07 = l2twiddlel1twiddle_A_URM07.evaluate()
        
        l2twiddlel1twiddle_B = -self.q*o2.A21 + self.iRm*o1.B + 2*1j*self.iRm*self.Q*o2.B21 + o2.u21
        l2twiddlel1twiddle_B = l2twiddlel1twiddle_B.evaluate()
        
        l2twiddlel1twiddle_B_URM07 = -self.q*o2.A21 + 2*self.iRm*o1.B + 2*1j*self.iRm*self.Q*o2.B21 + o2.u21 #Umurhan+'s wrong def of L2twiddle
        l2twiddlel1twiddle_B_URM07 = l2twiddlel1twiddle_B_URM07.evaluate()
        
        self.l2twiddlel1twiddle_psi = l2twiddlel1twiddle_psi
        self.l2twiddlel1twiddle_u = l2twiddlel1twiddle_u
        self.l2twiddlel1twiddle_A = l2twiddlel1twiddle_A
        self.l2twiddlel1twiddle_B = l2twiddlel1twiddle_B
        
        g_psi = (2/self.beta)*o1.A
        g_psi = g_psi.evaluate()
        
        # Hack... normalize all the RHS's
        """
        a_psi_rhs['g'] = self.normalize_vector(a_psi_rhs['g'])
        b_psi_rhs['g'] = self.normalize_vector(b_psi_rhs['g'])
        n3.N31_psi['g'] = self.normalize_vector(n3.N31_psi['g'])
        n3.N31_u['g'] = self.normalize_vector(n3.N31_u['g'])
        n3.N31_A['g'] = self.normalize_vector(n3.N31_A['g'])
        n3.N31_B['g'] = self.normalize_vector(n3.N31_B['g'])
        l2twiddlel1twiddle_psi['g'] = self.normalize_vector(l2twiddlel1twiddle_psi['g'])
        l2twiddlel1twiddle_u['g'] = self.normalize_vector(l2twiddlel1twiddle_u['g'])
        l2twiddlel1twiddle_A['g'] = self.normalize_vector(l2twiddlel1twiddle_A['g'])
        l2twiddlel1twiddle_B['g'] = self.normalize_vector(l2twiddlel1twiddle_B['g'])
        g_psi['g'] = self.normalize_vector(g_psi['g'])
        """
        # Normalize s.t. a = 1
        if magicnumberhack == False:
            logger.info("Normalizing V^dagger s.t. a = 1")
            ah.psi, ah.u, ah.A, ah.B = self.normalize_inner_product_eq_1(ah.psi, ah.u, ah.A, ah.B, a_psi_rhs, o1.u, o1.A, o1.B)
        
        # a = <va . D V11*>
        self.a = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [a_psi_rhs, o1.u, o1.A, o1.B])
        
        # c = <va . N31*>
        self.c = -self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [n3.N31_psi, n3.N31_u, n3.N31_A, n3.N31_B])
        
        # ctwiddle = < va . N31_twiddle_star >. Should be zero.
        self.ctwiddle = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [allzeros, c_twiddle_u_rhs, allzeros, c_twiddle_B_rhs])
        
        # b = < va . (X v11)* > :: in new terminology, b = < va . (Gtwiddle v11)* >
        #self.b = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [b_psi_rhs, o1.B, o1.psi, o1.u])
        self.b = -self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [b_psi_rhs, b_u_rhs, b_A_rhs, b_B_rhs])
  
        # h = < va . (L2twiddle v11 + L1twiddle v21)* >
        self.h = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [l2twiddlel1twiddle_psi, l2twiddlel1twiddle_u, l2twiddlel1twiddle_A, l2twiddlel1twiddle_B])
        self.h_URM07 = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [l2twiddlel1twiddle_psi_URM07, l2twiddlel1twiddle_u_URM07, l2twiddlel1twiddle_A_URM07, l2twiddlel1twiddle_B_URM07])
  
        # With new definition of b, no need for g
        # g = < va . (L3 v11) * >
        #self.g = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [g_psi, allzeros, allzeros, allzeros])
    
        #self.linear_term = 1j*self.Q*self.b - 1j*self.Q**3*self.g
        self.linear_term = self.b
    

        self.sat_amp_coeffs = np.sqrt(self.b/self.c) #np.sqrt((-1j*self.Q*self.b + 1j*self.Q**3*self.g)/self.c)
        
        # For interactive diagnostic purposes only
        self.o1 = o1
        self.o2 = o2
        self.n3 = n3
        self.ah = ah
        self.n2 = n2

    def print_coeffs(self):
        logger.info("sat_amp_coeffs = b/c")
        logger.info("a = {}; c = {}; ctwiddle = {}; b = {}; h = {}".format(self.a, self.c, self.ctwiddle, self.b, self.h))
        logger.info("h with URM07's wrong definition of L2twiddle = {}".format(self.h_URM07))
        logger.info("saturation amp = {}".format(self.sat_amp_coeffs))

    def solve_IVP(self):
        # Actually solve the IVP
        Absolute = operators.Absolute
        
        problem = ParsedProblem(axis_names=['Z'],
                           field_names=['alpha', 'alphaZ'],
                           param_names=['ac', 'bc', 'hc', 'Absolute'])
        
        #problem.add_equation("ac*dt(alpha) + -bc*1j*Q*alpha - hc*dZ(alphaZ) - gc*1j*Q**3*alpha = alpha*Absolute(alpha**2)") #fixed to be gle
        problem.add_equation("-ac*dt(alpha) + bc*alpha + hc*dZ(alphaZ) = alpha*Absolute(alpha**2)") 
        problem.add_equation("alphaZ - dZ(alpha) = 0")
        
        problem.parameters['ac'] = self.a/self.c
        problem.parameters['bc'] = self.b/self.c
        problem.parameters['hc'] = self.h/self.c
        
        problem.parameters['Absolute'] = Absolute
        
        lambda_crit = 2*np.pi/self.Q
        
        Z_basis = Fourier(self.gridnum, interval=(-lambda_crit, lambda_crit), dealias=2/3)
        Zdomain = Domain([Z_basis], np.complex128)
        problem.expand(self.domain)
        
        solver = solvers.IVP(problem, self.domain, timesteppers.SBDF2)
        
        # stopping criteria
        solver.stop_sim_time = np.inf
        solver.stop_wall_time = np.inf
        solver.stop_iteration = 50000#0
        
        # reference local grid and state fields
        Z = Zdomain.grid(0)
        alpha = solver.state['alpha']
        alphaZ = solver.state['alphaZ']

        # initial conditions ... plus noise!
        #noise = np.array([random.uniform(-1E-15, 1E-15) for _ in range(len(Z))])
        alpha['g'] = 1.0#E-5 + 1.0E-5j #+ noise
        alpha.differentiate(0, out=alphaZ)
        
        # Setup storage
        alpha_list = [np.copy(alpha['g'])]
        t_list = [solver.sim_time]

        # Main loop
        dt = 2e-2#2e-3
        while solver.ok:
            solver.step(dt)
            if solver.iteration % 20 == 0:
                alpha_list.append(np.copy(alpha['g']))
                t_list.append(solver.sim_time)
                
        # Convert storage to arrays
        alpha_array = np.array(alpha_list)
        t_array = np.array(t_list)
        
        self.alpha = alpha
        self.alphaZ = alphaZ
        self.alpha_array = alpha_array
        self.t_array = t_array
    
        self.saturation_amplitude = alpha_array[-1, 0]
