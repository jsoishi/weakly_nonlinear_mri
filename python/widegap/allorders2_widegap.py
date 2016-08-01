import numpy as np
from mpi4py import MPI
import dedalus.public as de
from eigentools import Eigenproblem
import random

#import logging
#logger = logging.getLogger(__name__)

import logging
root = logging.root
#for h in root.handlers:
#    h.setLevel("DEBUG")
    
logger = logging.getLogger(__name__)


class MRI():

    """
    Base class for MRI equations.
    
    Defaults: For Pm of 0.001, beta = 25
              Omega1 = 313.55, Omega2 = 67.0631, R1 = 5, R2 = 15 yield q(r0) = 1.5
              xi = 0 (SMRI)
    
    CriticalFinder gives:
    critical wavenumber k =    0.01795
    critical Rm =    0.84043
    """

    def __init__(self, domain, Q = 0.01795, Rm = 0.84043, Pm = 0.001, beta = 25.0, Omega1 = 313.55, Omega2 = 67.0631, xi = 0, norm = True):

        self.domain = domain
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.beta = beta
        self.Omega1 = Omega1
        self.Omega2 = Omega2
        self.xi = xi
        self.norm = norm
        self.B0 = 1
        
        # Inverse magnetic reynolds number
        self.iRm = 1.0/self.Rm
        
        # Reynolds number
        self.R = self.Rm/self.Pm
        self.iR = 1.0/self.R
        
        self.gridnum = self.domain.bases[0].coeff_size
        self.r = self.domain.grid(0)
        
        self.R1 = self.domain.bases[0].interval[0]
        self.R2 = self.domain.bases[0].interval[1]

        self.Omega1 = Omega1
        self.Omega2 = Omega2
        
        self.c1 = (self.Omega2*self.R2**2 - self.Omega1*self.R1**2)/(self.R2**2 - self.R1**2)
        self.c2 = (self.R1**2*self.R2**2*(self.Omega1 - self.Omega2))/(self.R2**2 - self.R1**2)
        
        self.zeta_mean = 2*(self.R2**2*self.Omega2 - self.R1**2*self.Omega1)/((self.R2**2 - self.R1**2)*np.sqrt(self.Omega1*self.Omega2))
        self.R0 = (self.R1 + self.R2)/2.0
        self.q_R0 = 2*self.c2/(self.R0**2*self.c1 + self.c2)
        
        logger.info("MRI parameters: Q = {}; Rm = {}; Pm = {}; beta = {}; norm = {}, Re = {}".format(self.Q, self.Rm, self.Pm, self.beta, norm, self.R))
        logger.info("Effective shear parameter q(R0) = {}".format(self.q_R0))
        
        if self.xi != 0:
            logger.info("A nonzero xi means this is the HMRI")
        
    def set_boundary_conditions(self, problem, conducting = True):
        
        """
        Adds MRI problem boundary conditions to a ParsedProblem object.
        """
        
        problem.add_bc('left(u) = 0')
        problem.add_bc('right(u) = 0')
        problem.add_bc('left(psi) = 0')
        problem.add_bc('right(psi) = 0')
        problem.add_bc('left(A) = 0')
        problem.add_bc('right(A) = 0')
        problem.add_bc('left(psir) = 0')
        problem.add_bc('right(psir) = 0')
        problem.add_bc('left(B + r*Br) = 0')
        problem.add_bc('right(B + r*Br) = 0') # axial component of current = 0
        
        return problem

    def fastest_growing(self):
        gr, largest_eval_indx  = self.EP.growth_rate({})
        self.largest_eval_indx = largest_eval_indx
        self.EP.solver.set_state(largest_eval_indx)
        
        print("Fastest mode has growth rate {}".format(gr))
    
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
        #logger.warn("Normalizing according to norm(psi)")
        #norm = np.linalg.norm(psi['g'])
        
        logger.warn("Normalizing according to integral(psi)")
        intpsi = psi.integrate('r')
        norm = intpsi['g'][0]
        
        #logger.warn("Normalizing according to integral(u)")
        #intu = u.integrate('r')
        #norm = intu['g'][0]
        
        psi['g'] = psi['g']/norm
        u['g'] = u['g']/norm
        A['g'] = A['g']/norm
        B['g'] = B['g']/norm
        
        return psi, u, A, B

    def get_derivative(self, field):
    
        """
        Take derivative of a single field.
        """
        
        field_r = field.differentiate(0)
        
        if field.name.endswith("r"):
            field_r.name = field.name + "r"
        else:
            field_r.name = field.name + "_r"
            
        return field_r
        
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
        ip = ip.integrate('r')
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

    def __init__(self, domain, o1 = None, Q = 0.01795, Rm = 0.84043, Pm = 0.001, beta = 25.0, Omega1 = 313.55, Omega2 = 67.0631, xi = 0, norm = True, finalize=True):
        
        logger.info("initializing Adjoint Homogenous")
        
        if o1 == None:
            self.o1 = OrderE(domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
            MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
        else:
            MRI.__init__(self, domain, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, beta = o1.beta, Omega1 = o1.Omega1, Omega2 = o1.Omega2, xi = o1.xi, norm = o1.norm)
      
        # Set up problem object
        lv1 = de.EVP(self.domain,
                     ['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],'sigma')

        lv1.parameters['Q'] = self.Q
        lv1.parameters['iR'] = self.iR
        lv1.parameters['iRm'] = self.iRm
        lv1.parameters['xi'] = self.xi
        lv1.parameters['beta'] = self.beta
        lv1.parameters['c1'] = self.c1
        lv1.parameters['c2'] = self.c2
        lv1.parameters['B0'] = self.B0
        
        lv1.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
        lv1.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
        lv1.substitutions['twooverbeta'] = '(2.0/beta)'
        
        lv1.add_equation("sigma*(-r**3*Q**2*psi + r**3*psirr - r**2*psir) - r*1j*Q*rrdu0*u - r*ru0*1j*Q*u + r**4*1j*Q*A - iR*r**3*Q**4*psi + iR*r**3*2*Q**2*psirr - iR*r**2*2*Q**2*psir - iR*r**3*dr(psirrr) + iR*r**2*2*psirrr - iR*r*3*psirr + iR*3*psir + r*2*1j*Q*B0*xi*B = 0")
        lv1.add_equation("sigma*r**3*u + r*2*1j*Q*ru0*psi + r**3*1j*Q*B + iR*r**3*Q**2*u - iR*r**3*dr(ur) - iR*r**2*ur + iR*r*u = 0") 
        lv1.add_equation("sigma*r*A + rrdu0*1j*Q*B - ru0*1j*Q*B - twooverbeta*r**2*1j*Q**3*psi + twooverbeta*r**2*1j*Q*psirr - twooverbeta*r*1j*Q*psir + iRm*r**3*Q**2*A - iRm*r**3*dr(Ar) + iRm*r**2*Ar = 0")
        lv1.add_equation("sigma*r**2*B + r**2*twooverbeta*1j*Q*u + r**2*iRm*Q**2*B - r**2*iRm*dr(Br) - iRm*r*Br + iRm*B - twooverbeta*2*1j*Q*B0*xi*psi = 0") 

        lv1.add_equation("dr(psi) - psir = 0")
        lv1.add_equation("dr(psir) - psirr = 0")
        lv1.add_equation("dr(psirr) - psirrr = 0")
        lv1.add_equation("dr(u) - ur = 0")
        lv1.add_equation("dr(A) - Ar = 0")
        lv1.add_equation("dr(B) - Br = 0")

        # Set boundary conditions for MRI problem
        self.lv1 = self.set_boundary_conditions(lv1)
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
        self.psi_r = self.get_derivative(self.psi)
        self.psi_rr = self.get_derivative(self.psi_r)
        self.psi_rrr = self.get_derivative(self.psi_rr)
      
        self.u_r = self.get_derivative(self.u)
        
        self.A_r = self.get_derivative(self.A)
        self.A_rr = self.get_derivative(self.A_r)
        self.A_rrr = self.get_derivative(self.A_rr)
        
        self.B_r = self.get_derivative(self.B)
        
class OrderE(MRI):

    """
    Solves the order(epsilon) equation L V_1 = 0
    This is simply the linearized MRI.
    Returns V_1
    """

    def __init__(self, domain, Q = 0.01795, Rm = 0.84043, Pm = 0.001, beta = 25.0, Omega1 = 313.55, Omega2 = 67.0631, xi = 0, norm = True, finalize=True):
        
        logger.info("initializing Order E")
        
        MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
        
        lv1 = de.EVP(self.domain,
                     ['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],'sigma')

        lv1.parameters['Q'] = self.Q
        lv1.parameters['iR'] = self.iR
        lv1.parameters['iRm'] = self.iRm
        lv1.parameters['xi'] = self.xi
        lv1.parameters['beta'] = self.beta
        lv1.parameters['c1'] = self.c1
        lv1.parameters['c2'] = self.c2
        lv1.parameters['B0'] = self.B0
        
        lv1.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
        lv1.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
        lv1.substitutions['twooverbeta'] = '(2.0/beta)'
        
        lv1.substitutions['psivisc'] = '(2*r**2*Q**2*psir - 2*r**3*Q**2*psirr + r**3*Q**4*psi + r**3*dr(psirrr) - 3*psir + 3*r*psirr - 2*r**2*psirrr)'
        lv1.substitutions['uvisc'] = '(-r**3*Q**2*u + r**3*dr(ur) + r**2*ur - r*u)'
        lv1.substitutions['Avisc'] = '(r*dr(Ar) - r*Q**2*A - Ar)' 
        lv1.substitutions['Bvisc'] = '(-r**3*Q**2*B + r**3*dr(Br) + r**2*Br - r*B)'
    
        lv1.add_equation("sigma*(-r**3*Q**2*psi + r**3*psirr - r**2*psir) - r**2*2*ru0*1j*Q*u + r**3*twooverbeta*B0*1j*Q**3*A + twooverbeta*B0*r**2*1j*Q*Ar - twooverbeta*r**3*B0*1j*Q*dr(Ar) - iR*psivisc + twooverbeta*r**2*2*xi*1j*Q*B = 0") #corrected on whiteboard 5/6
        lv1.add_equation("sigma*r**3*u + 1j*Q*ru0*psi + 1j*Q*rrdu0*psi - 1j*Q*r**3*twooverbeta*B0*B - iR*uvisc = 0") 
        lv1.add_equation("sigma*r*A - r*B0*1j*Q*psi - iRm*Avisc = 0")
        lv1.add_equation("sigma*r**3*B + ru0*1j*Q*A - r**3*B0*1j*Q*u - 1j*Q*rrdu0*A - iRm*Bvisc - 2*xi*1j*Q*psi = 0") 
        
        # Substitutions (temporarily?) not working with EP
        #lv1.add_equation("sigma*(-r**3*Q**2*psi + r**3*psirr - r**2*psir) - r**2*2*(r*r*c1 + c2)*1j*Q*u + r**3*(2.0/beta)*B0*1j*Q**3*A + (2.0/beta)*B0*r**2*1j*Q*Ar - (2.0/beta)*r**3*B0*1j*Q*dr(Ar) - iR*(2*r**2*Q**2*psir - 2*r**3*Q**2*psirr + r**3*Q**4*psi + r**3*dr(psirrr) - 3*psir + 3*r*psirr - 2*r**2*psirrr) + (2.0/beta)*r**2*2*xi*1j*Q*B = 0") #corrected on whiteboard 5/6
        #lv1.add_equation("sigma*r**3*u + 1j*Q*(r*r*c1 + c2)*psi + 1j*Q*(c1*r*r-c2)*psi - 1j*Q*r**3*(2.0/beta)*B0*B - iR*(-r**3*Q**2*u + r**3*dr(ur) + r**2*ur - r*u) = 0") 
        #lv1.add_equation("sigma*r*A - r*B0*1j*Q*psi - iRm*(r*dr(Ar) - r*Q**2*A - Ar) = 0")
        #lv1.add_equation("sigma*r**3*B + (r*r*c1 + c2)*1j*Q*A - r**3*B0*1j*Q*u - 1j*Q*(c1*r*r-c2)*A - iRm*(-r**3*Q**2*B + r**3*dr(Br) + r**2*Br - r*B) - 2*xi*1j*Q*psi = 0") 

        lv1.add_equation("dr(psi) - psir = 0")
        lv1.add_equation("dr(psir) - psirr = 0")
        lv1.add_equation("dr(psirr) - psirrr = 0")
        lv1.add_equation("dr(u) - ur = 0")
        lv1.add_equation("dr(A) - Ar = 0")
        lv1.add_equation("dr(B) - Br = 0")

        self.lv1 = self.set_boundary_conditions(lv1)
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
            
        self.psi.name = "psi"
        self.u.name = "u"
        self.A.name = "A"
        self.B.name = "B"
            
        # Take all relevant derivates for use with higher order terms
        self.psi_r = self.get_derivative(self.psi)
        self.psi_rr = self.get_derivative(self.psi_r)
        self.psi_rrr = self.get_derivative(self.psi_rr)
      
        self.u_r = self.get_derivative(self.u)
        
        # relevant for alternate O2 calculation
        self.u_rr = self.get_derivative(self.u_r)
        
        self.A_r = self.get_derivative(self.A)
        self.A_rr = self.get_derivative(self.A_r)
        self.A_rrr = self.get_derivative(self.A_rr)
        
        self.B_r = self.get_derivative(self.B)
        
        # relevant for alternate O2 calculation
        self.B_rr = self.get_derivative(self.B_r)
        
        # Also take relevant complex conjugates
        self.psi_star = self.get_complex_conjugate(self.psi)
        self.psi_star_r = self.get_derivative(self.psi_star)
        self.psi_star_rr = self.get_derivative(self.psi_star_r)
        self.psi_star_rrr = self.get_derivative(self.psi_star_rr)
        
        self.u_star = self.get_complex_conjugate(self.u)
        self.u_star_r = self.get_derivative(self.u_star)
        
        self.A_star = self.get_complex_conjugate(self.A)
        self.A_star_r = self.get_derivative(self.A_star)
        self.A_star_rr = self.get_derivative(self.A_star_r)
        self.A_star_rrr = self.get_derivative(self.A_star_rr)
        
        self.B_star = self.get_complex_conjugate(self.B)
        self.B_star_r = self.get_derivative(self.B_star)
        
class N2(MRI):

    """
    Solves the nonlinear term N2
    Returns N2
    
    """
    
    def __init__(self, domain, o1 = None, Q = 0.01795, Rm = 0.84043, Pm = 0.001, beta = 25.0, Omega1 = 313.55, Omega2 = 67.0631, xi = 0, norm = True):
    
        logger.info("initializing N2")
    
        if o1 is None:
            o1 = OrderE(domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
            MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
        else:
            MRI.__init__(self, domain, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, beta = o1.beta, Omega1 = o1.Omega1, Omega2 = o1.Omega2, xi = o1.xi, norm = o1.norm)
    
        rfield = domain.new_field()
        rfield['g'] = o1.r
    
        N22_psi_jacobians = ((1j*Q*o1.psi)*((1/rfield**2)*(-Q**2)*o1.psi_r - (3/rfield**3)*o1.psi_rr + (1/rfield**2)*o1.psi_rrr - (2/rfield**3)*(-Q**2)*o1.psi + (3/rfield**4)*o1.psi_r)
                - (o1.psi_r)*((1/rfield**2)*(1j*Q)*o1.psi_rr - (1/rfield**3)*(1j*Q)*o1.psi_r + (1/rfield**2)*(1j*Q)**3*o1.psi)
                - (2/beta)*((1j*Q*o1.A)*((1/rfield**2)*(-Q**2)*o1.A_r - (3/rfield**3)*o1.A_rr + (1/rfield**2)*o1.A_rrr - (2/rfield**3)*(-Q**2)*o1.A + (3/rfield**4)*o1.A_r)
                - (o1.A_r)*((1/rfield**2)*(1j*Q)*o1.A_rr - (1/rfield**3)*(1j*Q)*o1.A_r + (1/rfield**2)*(1j*Q)**3*o1.A)))
        N22_psi_advectives = (-(2/rfield)*o1.u*(1j*Q)*o1.u + (2/beta)*(2/rfield)*o1.B*(1j*Q)*o1.B)
        self.N22_psi = (N22_psi_jacobians + N22_psi_advectives).evaluate()
        self.N22_psi.name = "N22_psi"
        
        N22_psi_r4 = (((1j*Q*o1.psi)*(-2*rfield*o1.psi_rr + rfield**2*o1.psi_rrr - 2*rfield*(1j*Q)**2*o1.psi + rfield**2*(1j*Q)**2*o1.psi_r - 3*o1.psi_r + rfield*o1.psi_r + 6*o1.psi_r - 2*rfield*o1.psi_r)
                        - o1.psi_r*(1j*Q)*(rfield**2*o1.psi_rr + rfield**2*(1j*Q)**2*o1.psi + rfield*o1.psi_r - 2*rfield*o1.psi_r))
                        -(2/beta)*((1j*Q*o1.A)*(-2*rfield*o1.A_rr + rfield**2*o1.A_rrr - 2*rfield*(1j*Q)**2*o1.A + rfield**2*(1j*Q)**2*o1.A_r - 3*o1.A_r + rfield*o1.A_r + 6*o1.A_r - 2*rfield*o1.A_r)
                        - o1.A_r*(1j*Q)*(rfield**2*o1.A_rr + rfield**2*(1j*Q)**2*o1.A + rfield*o1.A_r - 2*rfield*o1.A_r))
                        -2*rfield**2*o1.u*(1j*Q)*o1.u + (2/beta)*2*rfield**3*o1.B*(1j*Q)*o1.B)
        self.N22_psi_r4 = N22_psi_r4.evaluate()
        self.N22_psi_r4.name = "N22_psi_r4"
        
        N22_psi_r4_rederived = ((-2*rfield*(1j*Q)**2*o1.psi*(1j*Q)*o1.psi + rfield**2*(1j*Q)*o1.psi*(1j*Q)**2*o1.psi_r - rfield**2*o1.psi_r*(1j*Q)**3*o1.psi
                               + 3*(1j*Q)*o1.psi*o1.psi_r + rfield*(1j*Q)*o1.psi_r*o1.psi_r - 3*rfield*(1j*Q)*o1.psi*o1.psi_rr 
                               + rfield**2*(1j*Q)*o1.psi*o1.psi_rrr - rfield**2*o1.psi_r*(1j*Q)*o1.psi_rr
                               - 2*rfield**3*o1.u*(1j*Q)*o1.u)
                               - (2/beta)*(-2*rfield*(1j*Q)**2*o1.A*(1j*Q)*o1.A + rfield**2*(1j*Q)*o1.A*(1j*Q)**2*o1.A_r - rfield**2*o1.A_r*(1j*Q)**3*o1.A
                               + 3*(1j*Q)*o1.A*o1.A_r + rfield*(1j*Q)*o1.A_r*o1.A_r - 3*rfield*(1j*Q)*o1.A*o1.A_rr 
                               + rfield**2*(1j*Q)*o1.A*o1.A_rrr - rfield**2*o1.A_r*(1j*Q)*o1.A_rr
                               - 2*rfield**3*o1.B*(1j*Q)*o1.B))
        self.N22_psi_r4_rederived = N22_psi_r4_rederived.evaluate()
        self.N22_psi_r4_rederived.name = "N22_psi_r4_rederived"
                               
        N20_psi_jacobians = ((1j*Q*o1.psi)*((1/rfield**2)*((-1j*Q)**2)*o1.psi_star_r - (3/rfield**3)*o1.psi_star_rr + (1/rfield**2)*o1.psi_star_rrr - (2/rfield**3)*((-1j*Q)**2)*o1.psi_star + (3/rfield**4)*o1.psi_star_r)
                - (o1.psi_r)*((1/rfield**2)*(-1j*Q)*o1.psi_star_rr - (1/rfield**3)*(-1j*Q)*o1.psi_star_r + (1/rfield**2)*(-1j*Q)**3*o1.psi_star)
                - (2/beta)*((1j*Q*o1.A)*((1/rfield**2)*((-1j*Q)**2)*o1.A_star_r - (3/rfield**3)*o1.A_star_rr + (1/rfield**2)*o1.A_star_rrr - (2/rfield**3)*((-1j*Q)**2)*o1.A_star + (3/rfield**4)*o1.A_star_r)
                - (o1.A_r)*((1/rfield**2)*(-1j*Q)*o1.A_star_rr - (1/rfield**3)*(-1j*Q)*o1.A_star_r + (1/rfield**2)*(-1j*Q)**3*o1.A_star)))
        N20_psi_advectives = (-(2/rfield)*o1.u*(-1j*Q)*o1.u_star + (2/beta)*(2/rfield)*o1.B*(-1j*Q)*o1.B_star)
        self.N20_psi = (N20_psi_jacobians + N20_psi_advectives).evaluate()
        self.N20_psi.name = "N20_psi"
        
        """
        # explicitly include (psi1, psi1star) and (psi1star, psi1)
        N20_psi_r4_rederived = ((-2*rfield*(1j*Q)**2*o1.psi*(-1j*Q)*o1.psi_star + rfield**2*(1j*Q)*o1.psi*(-1*Q)**2*o1.psi_star_r
                               - rfield**2*o1.psi_r*(-1j*Q)**3*o1.psi_star + 3*(1j*Q)*o1.psi*o1.psi_star_r
                               + rfield*(-1j*Q)*o1.psi_r*o1.psi_star_r - 3*rfield*(1j*Q)*o1.psi*o1.psi_star_rr
                               + rfield**2*(1j*Q)*o1.psi*o1.psi_star_rrr - rfield**2*o1.psi_r*(-1j*Q)*o1.psi_star_rr
                               + 2*rfield**3*1j*Q*o1.u*o1.u_star
                               - 2*rfield*(-1j*Q)**2*o1.psi_star*(1j*Q)*o1.psi + rfield**2*(-1j*Q)*o1.psi_star*(1j*Q)**2*o1.psi_r
                               - rfield**2*o1.psi_star_r*(1j*Q)**3*o1.psi + 3*(-1j*Q)*o1.psi_star*o1.psi_r
                               + rfield*(-1j*Q)*o1.psi_star_r*o1.psi_r - 3*rfield*(-1j*Q)*o1.psi_star*o1.psi_rr
                               + rfield**2*(-1j*Q)*o1.psi_star*o1.psi_rrr - rfield**2*o1.psi_star_r*(1j*Q)*o1.psi_rr
                               - 2*rfield**3*1j*Q*o1.u*o1.u_star)
                   - (2/beta)*(-2*rfield*(1j*Q)**2*o1.A*(-1j*Q)*o1.A_star + rfield**2*(1j*Q)*o1.A*(-1*Q)**2*o1.A_star_r
                               - rfield**2*o1.A_star_r*(-1j*Q)**3*o1.A_star + 3*(1j*Q)*o1.A*o1.A_star_r
                               + rfield*(-1j*Q)*o1.A_r*o1.A_star_r - 3*rfield*(1j*Q)*o1.A*o1.A_star_rr
                               + rfield**2*(1j*Q)*o1.A*o1.A_star_rrr - rfield**2*o1.A_r*(-1j*Q)*o1.A_star_rr
                               + 2*rfield**3*1j*Q*o1.B*o1.B_star
                               - 2*rfield*(-1j*Q)**2*o1.A_star*(1j*Q)*o1.A + rfield**2*(-1j*Q)*o1.A_star*(1j*Q)**2*o1.A_r
                               - rfield**2*o1.A_star_r*(1j*Q)**3*o1.A + 3*(-1j*Q)*o1.A_star*o1.A_r
                               + rfield*(-1j*Q)*o1.A_star_r*o1.A_r - 3*rfield*(-1j*Q)*o1.A_star*o1.A_rr
                               + rfield**2*(-1j*Q)*o1.A_star*o1.A_rrr - rfield**2*o1.A_star_r*(1j*Q)*o1.A_rr
                               - 2*rfield**3*1j*Q*o1.B*o1.B_star))
                               
        self.N20_psi_r4_rederived = N20_psi_r4_rederived.evaluate()
        self.N20_psi_r4_rederived.name = "N20_psi_r4_rederived"
        """
        # hack this to be zero because it's < 1E-15
        print("setting N20_psi to zero because N20_psi + N20_psi* = 0")
        allzeros = np.zeros(len(rfield['g']), np.complex)
        all_zeros_field = domain.new_field()
        all_zeros_field['g'] = allzeros 
        self.N20_psi_r4_rederived = all_zeros_field
        self.N20_psi_r4_rederived.name = "N20_psi_r4_rederived"
        
        # Note: doesn't matter which of (psi1, psi1) is chosen to be psi1_star -- here i am using opposite from above
        N20_psi_r4 = (((-1j*Q*o1.psi_star)*(-2*rfield*o1.psi_rr + rfield**2*o1.psi_rrr - 2*rfield*(1j*Q)**2*o1.psi + rfield**2*(1j*Q)**2*o1.psi_r - 3*o1.psi_r + rfield*o1.psi_r + 6*o1.psi_r - 2*rfield*o1.psi_r)
                        - o1.psi_star_r*(1j*Q)*(rfield**2*o1.psi_rr + rfield**2*(1j*Q)**2*o1.psi + rfield*o1.psi_r - 2*rfield*o1.psi_r))
                        -(2/beta)*((-1j*Q*o1.A_star)*(-2*rfield*o1.A_rr + rfield**2*o1.A_rrr - 2*rfield*(1j*Q)**2*o1.A + rfield**2*(1j*Q)**2*o1.A_r - 3*o1.A_r + rfield*o1.A_r + 6*o1.A_r - 2*rfield*o1.A_r)
                        - o1.A_star_r*(1j*Q)*(rfield**2*o1.A_rr + rfield**2*(1j*Q)**2*o1.A + rfield*o1.A_r - 2*rfield*o1.A_r))
                        -2*rfield**2*o1.u*(1j*Q)*o1.u_star + (2/beta)*2*rfield**3*o1.B*(1j*Q)*o1.B_star)
        self.N20_psi_r4 = N20_psi_r4.evaluate()
        self.N20_psi_r4.name = "N20_psi_r4"
        
        print("N20 psi r4 + cc:", self.N20_psi_r4['g'] + self.N20_psi_r4['g'].conj())
        print("rederived:", self.N20_psi_r4_rederived['g'])
        
        N22_u_jacobians = ((1/rfield)*((1j*Q*o1.psi)*o1.u_r - (1j*Q*o1.u)*o1.psi_r) - ((1/rfield)*(2/beta)*((1j*Q*o1.A)*o1.B_r - (1j*Q*o1.B)*o1.A_r)))
        N22_u_advectives = ((1/rfield**2)*o1.u*1j*Q*o1.psi - (2/beta)*(1/rfield**2)*o1.B*1j*Q*o1.A)
        self.N22_u = (N22_u_jacobians + N22_u_advectives).evaluate()
        self.N22_u.name = "N22_u"
        
        N22_u_r3 = rfield**2*(1j*Q*o1.psi*o1.u_r - 1j*Q*o1.u*o1.psi_r) - rfield**2*(2/beta)*(1j*Q*o1.A*o1.B_r - 1j*Q*o1.B*o1.A_r) + rfield*o1.u*(1j*Q)*o1.psi - rfield*(2/beta)*o1.B*(1j*Q)*o1.A
        self.N22_u_r3 = N22_u_r3.evaluate()
        self.N22_u_r3.name = "N22_u_r3"
        
        N20_u_jacobians = ((1/rfield)*((1j*Q*o1.psi)*o1.u_star_r - (-1j*Q*o1.u_star)*o1.psi_r) - ((1/rfield)*(2/beta)*((1j*Q*o1.A)*o1.B_star_r - (-1j*Q*o1.B_star)*o1.A_r)))
        N20_u_advectives = ((1/rfield**2)*o1.u*(-1j*Q)*o1.psi_star - (2/beta)*(1/rfield**2)*o1.B*(-1j*Q)*o1.A_star)
        self.N20_u = (N20_u_jacobians + N20_u_advectives).evaluate()
        self.N20_u.name = "N20_u"
        
        N20_u_r2 = rfield*(1j*Q*o1.psi*o1.u_star_r - -1j*Q*o1.u_star*o1.psi_r) - rfield*(2/beta)*(1j*Q*o1.A*o1.B_star_r - -1j*Q*o1.B_star*o1.A_r) + o1.u*(-1j*Q)*o1.psi_star - (2/beta)*o1.B*(-1j*Q)*o1.A_star
        self.N20_u_r2 = N20_u_r2.evaluate()
        self.N20_u_r2.name = "N20_u_r2"
        
        N20_u_r2_rederived = (rfield*(1j*Q)*o1.psi*o1.u_star_r - rfield*(-1j*Q)*o1.u_star*o1.psi_r + o1.u*(-1j*Q)*o1.psi_star
                            - rfield*(2/beta)*(1j*Q)*o1.A*o1.B_star_r + rfield*(2/beta)*(-1j*Q)*o1.B_star*o1.A_r - (2/beta)*o1.B*(-1j*Q)*o1.A_star
                            + rfield*(-1j*Q)*o1.psi_star*o1.u_r - rfield*(1j*Q)*o1.u*o1.psi_star_r + o1.u_star*(1j*Q)*o1.psi
                            - rfield*(2/beta)*(-1j*Q)*o1.A_star*o1.B_r + rfield*(2/beta)*(1j*Q)*o1.B*o1.A_star_r - (2/beta)*o1.B_star*(1j*Q)*o1.A)
        self.N20_u_r2_rederived = N20_u_r2_rederived.evaluate()
        self.N20_u_r2_rederived.name = "N20_u_r2_rederived"
                
        N22_A = (1/rfield)*((1j*Q*o1.psi)*o1.A_r - (1j*Q*o1.A)*o1.psi_r) # correct - checked 7/13/16
        self.N22_A = N22_A.evaluate()
        self.N22_A.name = "N22_A"
        
        N22_A_r = 1j*Q*o1.psi*o1.A_r - 1j*Q*o1.A*o1.psi_r
        self.N22_A_r = N22_A_r.evaluate()
        self.N22_A_r.name = "N22_A_r"
        
        N20_A = (1/rfield)*((1j*Q*o1.psi)*o1.A_star_r - (-1j*Q*o1.A_star)*o1.psi_r) # correct - checked 7/13/16
        self.N20_A = N20_A.evaluate()
        self.N20_A.name = "N20_A"
        
        N20_A_r = 1j*Q*o1.psi*o1.A_star_r - -1j*Q*o1.A_star*o1.psi_r
        self.N20_A_r = N20_A_r.evaluate()
        self.N20_A_r.name = "N20_A_r"
        
        N20_A_r_rederived = (-(1j*Q)*o1.A*o1.psi_star_r + (-1j*Q)*o1.psi_star*o1.A_r - (-1j*Q)*o1.A_star*o1.psi_r + (1j*Q)*o1.psi*o1.A_star_r)
        self.N20_A_r_rederived = N20_A_r_rederived.evaluate()
        self.N20_A_r_rederived.name = "N20_A_r_rederived"
        
        N22_B_jacobians = ((1/rfield)*((1j*Q*o1.u)*o1.A_r - (1j*Q*o1.A)*o1.u_r) + (1/rfield)*((1j*Q*o1.psi)*o1.B_r - (1j*Q*o1.B)*o1.psi_r))
        N22_B_advectives = (-(1/rfield**2)*o1.B*(1j*Q)*o1.psi + (1/rfield**2)*o1.u*(1j*Q)*o1.A)
        self.N22_B = (N22_B_jacobians + N22_B_advectives).evaluate()
        self.N22_B.name = "N22_B"
        
        N22_B_r3 = rfield**2*(1j*Q*o1.u*o1.A_r - 1j*Q*o1.A*o1.u_r) - rfield**2*(1j*Q*o1.B*o1.psi_r - 1j*Q*o1.psi*o1.B_r) + rfield*o1.u*1j*Q*o1.A - rfield*o1.B*1j*Q*o1.psi
        self.N22_B_r3 = N22_B_r3.evaluate()
        self.N22_B_r3.name = "N22_B_r3"
         
        N20_B_jacobians = ((1/rfield)*((1j*Q*o1.u)*o1.A_star_r - (-1j*Q*o1.A_star)*o1.u_r) + (1/rfield)*((1j*Q*o1.psi)*o1.B_star_r - (-1j*Q*o1.B_star)*o1.psi_r))
        N20_B_advectives = (-(1/rfield**2)*o1.B*(-1j*Q)*o1.psi_star + (1/rfield**2)*o1.u*(-1j*Q)*o1.A_star)
        self.N20_B = (N20_B_jacobians + N20_B_advectives).evaluate()
        self.N20_B.name = "N20_B"
        
        N20_B_r2 = rfield*(1j*Q*o1.u*o1.A_star_r - -1j*Q*o1.A_star*o1.u_r) - rfield*(1j*Q*o1.B*o1.psi_star_r - -1j*Q*o1.psi_star*o1.B_r) + o1.u*1j*Q*o1.A_star - o1.B*1j*Q*o1.psi_star
        self.N20_B_r2 = N20_B_r2.evaluate()
        self.N20_B_r2.name = "N20_B_r2"
        
        """
        N20_B_r2_rederived = (-rfield*(1j*Q)*o1.A*o1.u_star_r + rfield*(-1j*Q)*o1.u_star*o1.A_r - rfield*(-1j*Q)*o1.A*o1.u_r + rfield*(1j*Q)*o1.u*o1.A_star_r
                             -rfield*(1j*Q)*o1.B*o1.psi_star_r + rfield*(-1j*Q)*o1.psi_star*o1.B_r - rfield*(-1j*Q)*o1.B_star*o1.psi_r + rfield*(1j*Q)*o1.psi*o1.B_star_r
                             - o1.B*(-1j*Q)*o1.psi_star - o1.B_star*(1j*Q)*o1.psi
                             + o1.u*(-1j*Q)*o1.A_star + o1.u_star*(1j*Q)*o1.A)
        self.N20_B_r2_rederived = N20_B_r2_rederived.evaluate()
        self.N20_B_r2_rederived.name = "N20_B_r2_rederived"
        """
        """
        N20_B_r2_rederived = (rfield*(1j*Q)*o1.psi*o1.B_star_r - rfield*o1.psi_r*(-1j*Q)*o1.B_star + o1.u*(-1j*Q)*o1.A_star - rfield*(1j*Q)*o1.A*o1.u_star_r
                               + rfield*o1.A_r*(-1j*Q)*o1.u_star - o1.B*(-1j*Q)*o1.psi_star
                               + rfield*(-1j*Q)*o1.psi_star*o1.B_r - rfield*o1.psi_star_r*(1j*Q)*o1.B + o1.u_star*(1j*Q)*o1.A - rfield*(-1j*Q)*o1.A_star*o1.u_r
                               + rfield*o1.A_star_r*(1j*Q)*o1.u - o1.B_star*(1j*Q)*o1.psi)
        self.N20_B_r2_rederived = N20_B_r2_rederived.evaluate()
        self.N20_B_r2_rederived.name = "N20_B_r2_rederived"
        """
        
        # hack this to be zero because it's < 1E-15
        print("setting N20_B to zero because N20_B + N20_B* = 0")
        self.N20_B_r2_rederived = all_zeros_field
        self.N20_B_r2_rederived.name = "N20_B_r2_rederived"
        
class OrderE2(MRI):

    """
    Solves the second order equation L V2 = N2 - Ltwiddle V1
    Returns V2
    
    """
    
    def __init__(self, domain, o1 = None, ah = None, Q = 0.01795, Rm = 0.84043, Pm = 0.001, beta = 25.0, Omega1 = 313.55, Omega2 = 67.0631, xi = 0, norm = True):
    
        logger.info("initializing Order E2")
        
        if o1 is None:
            o1 = OrderE(domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
            MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
            n2 = N2(domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
        else:
            MRI.__init__(self, domain, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, beta = o1.beta, Omega1 = o1.Omega1, Omega2 = o1.Omega2, xi = o1.xi, norm = o1.norm)
            n2 = N2(domain, o1 = o1, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, beta = o1.beta, Omega1 = o1.Omega1, Omega2 = o1.Omega2, xi = o1.xi, norm = o1.norm)
        
        self.o1 = o1
        self.n2 = n2
        
        rfield = self.domain.new_field()
        rfield['g'] = o1.r
        
        # righthand side for the 20 terms (e^0)
        # need N20 + N20* on RHS
        # LV20 = -N20 - N20*
        N20_psi_cc = self.domain.new_field()
        N20_psi_cc['g'] = n2.N20_psi['g'].conj()
        N20_u_cc = self.domain.new_field()
        N20_u_cc['g'] = n2.N20_u['g'].conj()
        N20_A_cc = self.domain.new_field()
        N20_A_cc['g'] = n2.N20_A['g'].conj()
        N20_B_cc = self.domain.new_field()
        N20_B_cc['g'] = n2.N20_B['g'].conj()
        
        # multiplied by multiples of r
        N20_psi_r4_cc = self.domain.new_field()
        N20_psi_r4_cc['g'] = n2.N20_psi_r4['g'].conj()
        N20_u_r2_cc = self.domain.new_field()
        N20_u_r2_cc['g'] = n2.N20_u_r2['g'].conj()
        N20_A_r_cc = self.domain.new_field()
        N20_A_r_cc['g'] = n2.N20_A_r['g'].conj()
        N20_B_r2_cc = self.domain.new_field()
        N20_B_r2_cc['g'] = n2.N20_B_r2['g'].conj()
        
        #self.rhs_psi20 = ((-n2.N20_psi - N20_psi_cc)*rfield**4).evaluate()
        #self.rhs_u20 = ((-n2.N20_u - N20_u_cc)*rfield**2).evaluate()
        #self.rhs_A20 = ((-n2.N20_A - N20_A_cc)*rfield).evaluate()
        #self.rhs_B20 = ((-n2.N20_B - N20_B_cc)*rfield**2).evaluate()
    
        #self.rhs_psi20 = (-n2.N20_psi_r4 - N20_psi_r4_cc).evaluate()
        #self.rhs_u20 = (-n2.N20_u_r2 - N20_u_r2_cc).evaluate()
        #self.rhs_A20 = (-n2.N20_A_r - N20_A_r_cc).evaluate()
        #self.rhs_B20 = (-n2.N20_B_r2 - N20_B_r2_cc).evaluate()
        
        self.rhs_psi20 = (-n2.N20_psi_r4_rederived).evaluate() # trying explicit "cc" term included 
        self.rhs_u20 = (-n2.N20_u_r2_rederived).evaluate()
        self.rhs_A20 = (-n2.N20_A_r_rederived).evaluate()
        self.rhs_B20 = (-n2.N20_B_r2_rederived).evaluate()
    
    
        # V20 equations are separable because dz terms -> 0, but we'll solve them coupled anyway.
        bv20 = de.LBVP(self.domain,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'])
        
        bv20.parameters['rhs_psi20'] = self.rhs_psi20
        bv20.parameters['rhs_u20'] = self.rhs_u20
        bv20.parameters['rhs_A20'] = self.rhs_A20
        bv20.parameters['rhs_B20'] = self.rhs_B20
        bv20.parameters['iR'] = self.iR
        bv20.parameters['iRm'] = self.iRm
        
        bv20.add_equation("-iR*(r**3*dr(psirrr) - r**2*2*psirrr + r*3*psirr - 3*psir) = rhs_psi20")
        bv20.add_equation("-iR*(r**2*dr(ur) + r*ur - u) = rhs_u20")
        bv20.add_equation("-iRm*(r*dr(Ar) - Ar) = rhs_A20")
        bv20.add_equation("-iRm*(r**2*dr(Br) + r*Br - B) = rhs_B20")
        
        bv20.add_equation("dr(psi) - psir = 0")
        bv20.add_equation("dr(psir) - psirr = 0")
        bv20.add_equation("dr(psirr) - psirrr = 0")
        bv20.add_equation("dr(u) - ur = 0")
        bv20.add_equation("dr(A) - Ar = 0")
        bv20.add_equation("dr(B) - Br = 0")
        
        bv20 = self.set_boundary_conditions(bv20)
        self.BVP20 = self.solve_BVP(bv20)
        
        self.psi20 = self.BVP20.state['psi']
        self.u20 = self.BVP20.state['u']
        self.A20 = self.BVP20.state['A']
        self.B20 = self.BVP20.state['B']
        
        print('B20', self.B20['g'])
        print('N20_B rhs', self.rhs_B20['g'])
        
        # V21 equations are coupled
        # RHS of V21 = -L1twiddle V11
        rfield = self.domain.new_field()
        rfield['g'] = o1.r
        
        u0field = self.domain.new_field()
        u0field['g'] = self.c1*rfield['g'] + self.c2*(1/rfield['g'])
        
        ru0field = self.domain.new_field()
        ru0field['g'] = self.c1*rfield['g']**2 + self.c2
        
        du0field = self.domain.new_field()
        du0field['g'] = self.c1 - self.c2*(1/rfield['g']**2)
        
        rrdu0field = self.domain.new_field()
        rrdu0field['g'] = rfield['g']**2*self.c1 - self.c2
        
        # multiplied by r^4
        rhs_psi21 = (-self.iR*4*1j*self.Q**3*rfield**3*o1.psi - (2/self.beta)*3*rfield**3*self.Q**2*o1.A + self.iR*4*1j*self.Q*rfield**3*o1.psi_rr
                    -self.iR*4*1j*self.Q*rfield**2*o1.psi_r + 2*rfield**2*ru0field*o1.u + (2/self.beta)*rfield**3*o1.A_rr - (2/self.beta)*rfield**2*o1.A_r
                    -(2/self.beta)*2*xi*rfield**2*o1.B)
        self.rhs_psi21 = rhs_psi21.evaluate()
        
        # multiplied by r^3
        #rhs_u21 = (2*self.iR*1j*self.Q*rfield**3*o1.u - du0field*rfield**2*o1.psi - rfield*u0field*o1.psi + (2/self.beta)*rfield**3*o1.B)
        #self.rhs_u21 = rhs_u21.evaluate()
        
        rhs_u21 = (2*self.iR*1j*self.Q*rfield**3*o1.u - rrdu0field*o1.psi - ru0field*o1.psi + (2/self.beta)*rfield**3*o1.B)
        self.rhs_u21 = rhs_u21.evaluate()
        
        # multiplied by r
        rhs_A21 = self.iRm*2*1j*Q*rfield*o1.A + rfield*o1.psi
        self.rhs_A21 = rhs_A21.evaluate()
        
        # multiplied by r^3
        rhs_B21 = (self.iRm*2*1j*self.Q*rfield**3*o1.B + rrdu0field*o1.A + rfield**3*o1.u  - ru0field*o1.A + 2*xi*o1.psi)
        self.rhs_B21 = rhs_B21.evaluate()
        
        # These RHS terms must satisfy the solvability condition <V^dagger | RHS> = 0. Test that:
        if ah == None:
            self.ah = AdjointHomogenous(domain, o1 = o1, Q = self.Q, Rm = self.Rm, Pm = self.Pm, beta = self.beta, Omega1 = self.Omega1, Omega2 = self.Omega2, xi = self.xi, norm = self.norm)
            ah = self.ah
        else:
            self.ah = ah
        
        # RHS terms have been multiplied by [r^4, r^3, r, r^3]^T, so do the same for AH terms
        ah_psi_r4 = (rfield**4*self.ah.psi).evaluate()
        ah_u_r3 = (rfield**3*self.ah.u).evaluate()
        ah_A_r = (rfield*self.ah.A).evaluate()
        ah_B_r3 = (rfield**3*self.ah.B).evaluate()
        
        self.rhs_psi21_nor = (self.rhs_psi21/rfield**4).evaluate()
        self.rhs_u21_nor = (self.rhs_u21/rfield**3).evaluate()
        self.rhs_A21_nor = (self.rhs_A21/rfield).evaluate()
        self.rhs_B21_nor = (self.rhs_B21/rfield**3).evaluate()
        
        #sctest = self.take_inner_product_real((self.rhs_psi21, self.rhs_u21, self.rhs_A21, self.rhs_B21),(ah_psi_r4, ah_u_r3, ah_A_r, ah_B_r3))
        sctest = self.take_inner_product_real((self.rhs_psi21_nor, self.rhs_u21_nor, self.rhs_A21_nor, self.rhs_B21_nor),(ah.psi, ah.u, ah.A, ah.B))
        logger.info("solvability condition satisfied? {}".format(sctest))
        if np.abs(sctest) > 1E-10:
            logger.warn("CAUTION: solvability condition <V^dagger | RHS> = 0 failed for V21")
        
        rfield = self.domain.new_field()
        rfield['g'] = o1.r
                
        # define problem using righthand side as nonconstant coefficients
        bv21 = de.LBVP(self.domain,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'])
              
        # righthand side for the 21 terms (e^iQz dependence)
        bv21.parameters['rhs_psi21'] = self.rhs_psi21
        bv21.parameters['rhs_u21'] = self.rhs_u21
        bv21.parameters['rhs_A21'] = self.rhs_A21
        bv21.parameters['rhs_B21'] = self.rhs_B21
              
        # parameters
        bv21.parameters['Q'] = self.Q
        bv21.parameters['iR'] = self.iR
        bv21.parameters['iRm'] = self.iRm
        bv21.parameters['beta'] = self.beta
        bv21.parameters['c1'] = self.c1
        bv21.parameters['c2'] = self.c2
        bv21.parameters['xi'] = self.xi
        bv21.parameters['B0'] = self.B0
        
        bv21.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
        bv21.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
        bv21.substitutions['twooverbeta'] = '(2.0/beta)'
        bv21.substitutions['psivisc'] = '(2*r**2*Q**2*psir - 2*r**3*Q**2*psirr + r**3*Q**4*psi + r**3*dr(psirrr) - 3*psir + 3*r*psirr - 2*r**2*psirrr)'
        bv21.substitutions['uvisc'] = '(-r**3*Q**2*u + r**3*dr(ur) + r**2*ur - r*u)'
        bv21.substitutions['Avisc'] = '(r*dr(Ar) - r*Q**2*A - Ar)' 
        bv21.substitutions['Bvisc'] = '(-r**3*Q**2*B + r**3*dr(Br) + r**2*Br - r*B)'
    
        bv21.add_equation("-r**2*2*ru0*1j*Q*u + r**3*twooverbeta*1j*Q**3*A + twooverbeta*r**2*1j*Q*Ar - twooverbeta*r**3*1j*Q*dr(Ar) - iR*psivisc + twooverbeta*r**2*2*xi*1j*Q*B = rhs_psi21") #corrected on whiteboard 5/6
        bv21.add_equation("1j*Q*ru0*psi + 1j*Q*rrdu0*psi - 1j*Q*r**3*twooverbeta*B0*B - iR*uvisc = rhs_u21") 
        bv21.add_equation("r*B0*1j*Q*psi - iRm*Avisc = rhs_A21")
        bv21.add_equation("ru0*1j*Q*A - r**3*B0*1j*Q*u - 1j*Q*rrdu0*A - iRm*Bvisc - 2*xi*1j*Q*psi = rhs_B21") 

        bv21.add_equation("dr(psi) - psir = 0")
        bv21.add_equation("dr(psir) - psirr = 0")
        bv21.add_equation("dr(psirr) - psirrr = 0")
        bv21.add_equation("dr(u) - ur = 0")
        bv21.add_equation("dr(A) - Ar = 0")
        bv21.add_equation("dr(B) - Br = 0")

        # boundary conditions
        bv21 = self.set_boundary_conditions(bv21)

        self.BVP21 = self.solve_BVP(bv21)
        self.psi21 = self.BVP21.state['psi']
        self.u21 = self.BVP21.state['u']
        self.A21 = self.BVP21.state['A']
        self.B21 = self.BVP21.state['B']
        
        # define problem using righthand side as nonconstant coefficients
        bv22 = de.LBVP(self.domain,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'])
        
        # LV22 = -N22
        self.rhs_psi22 = -n2.N22_psi
        self.rhs_u22 = -n2.N22_u
        self.rhs_A22 = -n2.N22_A
        self.rhs_B22 = -n2.N22_B
        
        #bv22.parameters['rhs_psi22'] = (self.rhs_psi22*rfield**4).evaluate()
        #bv22.parameters['rhs_u22'] = (self.rhs_u22*rfield**3).evaluate() # multiplied by r^3 because of (1/r)*dr(u0) term
        #bv22.parameters['rhs_A22'] = (self.rhs_A22*rfield).evaluate()
        #bv22.parameters['rhs_B22'] = (self.rhs_B22*rfield**3).evaluate() # multiplied by r^3 because of (1/r)*dr(u0) term
        
        # multiples of r
        bv22.parameters['rhs_psi22'] = (-n2.N22_psi_r4).evaluate()
        bv22.parameters['rhs_u22'] = (-n2.N22_u_r3).evaluate() # multiplied by r^3 because of (1/r)*dr(u0) term
        bv22.parameters['rhs_A22'] = (-n2.N22_A_r).evaluate()
        bv22.parameters['rhs_B22'] = (-n2.N22_B_r3).evaluate() # multiplied by r^3 because of (1/r)*dr(u0) term
        
        # parameters
        bv22.parameters['Q'] = self.Q
        bv22.parameters['iR'] = self.iR
        bv22.parameters['iRm'] = self.iRm
        bv22.parameters['beta'] = self.beta
        bv22.parameters['xi'] = self.xi
        bv22.parameters['c1'] = self.c1
        bv22.parameters['c2'] = self.c2
        
        bv22.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
        bv22.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
        bv22.substitutions['twooverbeta'] = '(2.0/beta)'
        bv22.substitutions['dz'] = '(1j*2*Q)'
        
        bv22.add_equation("-r**2*ru0*dz*2*u - twooverbeta*r**3*dz*dr(Ar) + twooverbeta*r**2*dz*Ar - twooverbeta*r**3*dz**3*A - iR*r**3*dr(psirrr) + 2*iR*r**2*psirrr - 2*iR*r**3*dz**2*psirr - 3*iR*r*psirr + 2*iR*r**2*dz**2*psir + 3*iR*psir - iR*r**3*dz**4*psi + twooverbeta*r**2*2*xi*(2*1j*Q)*B = rhs_psi22")
        bv22.add_equation("rrdu0*dz*psi + ru0*dz*psi - r**3*twooverbeta*dz*B - r**3*iR*dr(ur) - r**2*iR*ur - r**3*iR*dz**2*u + r*iR*u = rhs_u22")
        bv22.add_equation("-dz*psi*r - r*iRm*dr(Ar) + iRm*Ar - r*iRm*dz**2*A = rhs_A22")
        bv22.add_equation("-dz*rrdu0*A - dz*r**3*u + ru0*dz*A - iRm*r**3*dr(Br) - r**2*iRm*Br - iRm*r**3*dz**2*B + r*iRm*B - 2*xi*(2*1j*Q)*psi = rhs_B22") #checked 7/14/16 - these are all right
        
        bv22.add_equation("dr(psi) - psir = 0")
        bv22.add_equation("dr(psir) - psirr = 0")
        bv22.add_equation("dr(psirr) - psirrr = 0")
        bv22.add_equation("dr(u) - ur = 0")
        bv22.add_equation("dr(A) - Ar = 0")
        bv22.add_equation("dr(B) - Br = 0")
        
        # boundary conditions
        bv22 = self.set_boundary_conditions(bv22)
        
        self.BVP22 = self.solve_BVP(bv22)
        self.psi22 = self.BVP22.state['psi']
        self.u22 = self.BVP22.state['u']
        self.A22 = self.BVP22.state['A']
        self.B22 = self.BVP22.state['B']
        
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
        self.psi20_r = self.get_derivative(self.psi20)
        self.psi20_rr = self.get_derivative(self.psi20_r)
        self.psi20_rrr = self.get_derivative(self.psi20_rr)
        
        self.psi20_star = self.get_complex_conjugate(self.psi20)
        
        self.psi20_star_r = self.get_derivative(self.psi20_star)
        self.psi20_star_rr = self.get_derivative(self.psi20_star_r)
        self.psi20_star_rrr = self.get_derivative(self.psi20_star_rr)
        
        self.psi21_r = self.get_derivative(self.psi21)
        self.psi21_rr = self.get_derivative(self.psi21_r)
        
        self.psi22_r = self.get_derivative(self.psi22)
        self.psi22_rr = self.get_derivative(self.psi22_r)
        self.psi22_rrr = self.get_derivative(self.psi22_rr)
        
        # u
        self.u20_r = self.get_derivative(self.u20)
        self.u20_star = self.get_complex_conjugate(self.u20)
        self.u20_star_r = self.get_derivative(self.u20_star)
        
        self.u22_r = self.get_derivative(self.u22)
        self.u22_star = self.get_complex_conjugate(self.u22)
        self.u22_star_r = self.get_derivative(self.u22_star)
        
        # B 
        self.B20_r = self.get_derivative(self.B20)
        self.B20_star = self.get_complex_conjugate(self.B20)
        self.B20_star_r = self.get_derivative(self.B20_star)
        
        self.B22_r = self.get_derivative(self.B22)
        self.B22_star = self.get_complex_conjugate(self.B22)
        self.B22_star_r = self.get_derivative(self.B22_star)
        
        # A 
        self.A20_r = self.get_derivative(self.A20)
        self.A20_rr = self.get_derivative(self.A20_r)
        self.A20_rrr = self.get_derivative(self.A20_rr)
        
        self.A20_star = self.get_complex_conjugate(self.A20)
        
        self.A20_star_r = self.get_derivative(self.A20_star)
        self.A20_star_rr = self.get_derivative(self.A20_star_r)
        self.A20_star_rrr = self.get_derivative(self.A20_star_rr)
        
        self.A21_r = self.get_derivative(self.A21)
        self.A21_rr = self.get_derivative(self.A21_r)
        
        self.A22_r = self.get_derivative(self.A22)
        self.A22_rr = self.get_derivative(self.A22_r)
        self.A22_rrr = self.get_derivative(self.A22_rr)
        
       
class N3(MRI):

    """
    Solves the nonlinear vector N3
    Returns N3
    
    """
    
    def __init__(self, domain, o1 = None, o2 = None, ah = None, Q = 0.01795, Rm = 0.84043, Pm = 0.001, beta = 25.0, Omega1 = 313.55, Omega2 = 67.0631, xi = 0, norm = True):
        
        logger.info("initializing N3")
        
        if o1 == None:
            o1 = OrderE(domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
            MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
            
        else:
            MRI.__init__(self, domain, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, beta = o1.beta, Omega1 = o1.Omega1, Omega2 = o1.Omega2, xi = o1.xi, norm = o1.norm)
            
        if ah == None:
            ah = AdjointHomogenous(domain, o1 = o1, Q = self.Q, Rm = self.Rm, Pm = self.Pm, beta = self.beta, Omega1 = self.Omega1, Omega2 = self.Omega2, xi = self.xi, norm = self.norm)

        if o2 == None:
            o2 = OrderE2(domain, o1 = o1, ah = ah, Q = self.Q, Rm = self.Rm, Pm = self.Pm, beta = self.beta, Omega1 = self.Omega1, Omega2 = self.Omega2, xi = self.xi, norm = self.norm)
        
        rfield = self.domain.new_field()
        rfield['g'] = o1.r
        
        invr = self.domain.new_field()
        invr['g'] = (1/o1.r)
        
        k = self.Q
        
        # J(psi1, (1/r^2)del^2 psi2)
        Jacobian1_part1 = ((1j*k*o1.psi)*(-2*invr**3*o2.psi20_rr + invr**2*o2.psi20_rrr - 3*invr**4*o2.psi20_r + invr**3*o2.psi20_rr)
                        + (1j*k*o1.psi)*(-2*invr**3*o2.psi20_star_rr + invr**2*o2.psi20_star_rrr - 3*invr**4*o2.psi20_star_r + invr**3*o2.psi20_star_rr)
                        + (-1j*k*o1.psi_star)*(-2*invr**3*o2.psi22_rr + invr**2*o2.psi22_rrr - 3*invr**4*o2.psi22_r + invr**3*o2.psi22_rr + (2*1j*k)**2*(-2*invr**3*o2.psi22 + invr**2*o2.psi22_r))
                        + (-2*1j*k*o1.psi_star_r)*(invr**2*o2.psi22_rr + invr**3*o2.psi22_r + (2*1j*k)**2*invr**2*o2.psi22)) # checked
        
        # J(psi1, (-2/r^3) dr psi2)
        Jacobian1_part2 = ((1j*k*o1.psi)*(6*invr**4*o2.psi20_r - 2*invr**3*o2.psi20_rr) + (1j*k*o1.psi)*(6*invr**4*o2.psi20_star_r - 2*invr**3*o2.psi20_star_rr)
                        + (-1j*k*o1.psi_star)*(6*invr**4*o2.psi22_r - 2*invr**3*o2.psi22_rr) - (o1.psi_star_r)*(2*1j*k*(-2)*invr**2*o2.psi22_r))
        
        # -(2/beta)J(A1, (1/r^2)del^2 A2)
        Jacobian2_part1 = (-2/self.beta)*((1j*k*o1.A)*(-2*invr**3*o2.A20_rr + invr**2*o2.A20_rrr - 3*invr**4*o2.A20_r + invr**3*o2.A20_rr)
                        + (1j*k*o1.A)*(-2*invr**3*o2.A20_star_rr + invr**2*o2.A20_star_rrr - 3*invr**4*o2.A20_star_r + invr**3*o2.A20_star_rr)
                        + (-1j*k*o1.A_star)*(-2*invr**3*o2.A22_rr + invr**2*o2.A22_rrr - 3*invr**4*o2.A22_r + invr**3*o2.A22_rr + (2*1j*k)**2*(-2*invr**3*o2.A22 + invr**2*o2.A22_r))
                        + (-2*1j*k*o1.A_star_r)*(invr**2*o2.A22_rr + invr**3*o2.A22_r + (2*1j*k)**2*invr**2*o2.A22))
        
        # -(2/beta) J(A1, -(2/r^3) dr A2)
        Jacobian2_part2 = (-2/self.beta)*((1j*k*o1.A)*(6*invr**4*o2.A20_r - 2*invr**3*o2.A20_rr) + (1j*k*o1.A)*(6*invr**4*o2.A20_star_r - 2*invr**3*o2.A20_star_rr)
                        + (-1j*k*o1.A_star)*(6*invr**4*o2.A22_r - 2*invr**3*o2.A22_rr) - (o1.A_star_r)*(2*1j*k*(-2)*invr**2*o2.A22_r))
        
        # J(psi2, (1/r^2) del^2 psi1)
        Jacobian3_part1 = (-o2.psi20_r*(1j*k)*(invr**2*o1.psi_rr + invr**2*(1j*k)**2*o1.psi + invr**3*o1.psi_r)
                          -o2.psi20_star_r*(1j*k)*(invr**2*o1.psi_rr + invr**2*(1j*k)**2*o1.psi + invr**3*o1.psi_r)
                          + (1j*2*k)*o2.psi22*(-2*invr**3*o1.psi_star_rr + invr**2*o1.psi_star_rrr + (-1j*k)**2*(-2*invr**3*o1.psi_star + invr**2*o1.psi_star_r) - 3*invr**4*o1.psi_star_r + invr**3*o1.psi_star_rr) # added missing derivative 7/14/16
                          - o2.psi22_r*(-1j*k)*(invr**2*o1.psi_star_rr + invr**2*(-1j*k)**2*o1.psi_star + invr**3*o1.psi_star_r))
        
        # J(psi2, -(2/r^3) dr psi1)
        Jacobian3_part2 = (-o2.psi20_r*(1j*k)*(-2*invr**3*o1.psi_r) - o2.psi20_star_r*(1j*k)*(-2*invr**3*o1.psi_r)
                          + (1j*2*k)*o2.psi22*(6*invr**4*o1.psi_star_r - 2*invr**3*o1.psi_star_rr)
                          - o2.psi22_r*(-1j*k)*(-2*invr**3*o1.psi_star_r)) # added missing negative sign 7/14/15
        
        # -(2/beta) J(A2, (1/r^2) del^2 A1)                  
        Jacobian4_part1 = (-2/self.beta)*(-o2.A20_r*(1j*k)*(invr**2*o1.A_rr + invr**2*(1j*k)**2*o1.A + invr**3*o1.A_r)
                          -o2.A20_star_r*(1j*k)*(invr**2*o1.A_rr + invr**2*(1j*k)**2*o1.A + invr**3*o1.A_r)
                          + (1j*2*k)*o2.A22*(-2*invr**3*o1.A_star_rr + invr**2*o1.A_star_rrr + (-1j*k)**2*(-2*invr**3*o1.A_star + invr**2*o1.A_star_r) - 3*invr**4*o1.A_star_r + invr**3*o1.A_star_rr)  # added missing derivative 7/14/16
                          - o2.A22_r*(-1j*k)*(invr**2*o1.A_star_rr + invr**2*(-1j*k)**2*o1.A_star + invr**3*o1.A_star_r))
        
        # -(2/beta) J(A2, -(2/r^3) dr A1)
        Jacobian4_part2 = (-2/self.beta)*(-o2.A20_r*(1j*k)*(-2*invr**3*o1.A_r) - o2.A20_star_r*(1j*k)*(-2*invr**3*o1.A_r)
                          + (1j*2*k)*o2.A22*(6*invr**4*o1.A_star_r - 2*invr**3*o1.A_star_rr)
                          - o2.A22_r*(-1j*k)*(-2*invr**3*o1.A_star_r)) # added missing negative sign 7/14/15
        
        # -(2/r) u1 dz u2 
        advective1 = -2*invr*o1.u_star*2*1j*k*o2.u22
        
        # -(2/r) u2 dz u1
        advective2 = -2*invr*(o2.u20*1j*k*o1.u + o2.u20_star*1j*k*o1.u + o2.u22*(-1j*k)*o1.u_star)
        
        # (2/beta) (2/r) B1 dz B2
        advective3 = 2*invr*(2/self.beta)*o1.B_star*2*1j*k*o2.B22
        
        # (2/beta) (2/r) B2 dz B1
        advective4 = 2*invr*(2/self.beta)*(o2.B20*1j*k*o1.B + o2.B20_star*1j*k*o1.B + o2.B22*(-1j*k)*o1.B_star)
        
        # diagnostics
        self.N31_psi_Jacobian1_part1 = Jacobian1_part1.evaluate()
        self.N31_psi_Jacobian1_part2 = Jacobian1_part2.evaluate()
        self.N31_psi_Jacobian2_part1 = Jacobian2_part1.evaluate()
        self.N31_psi_Jacobian2_part2 = Jacobian2_part2.evaluate()
        self.N31_psi_Jacobian3_part1 = Jacobian3_part1.evaluate()
        self.N31_psi_Jacobian3_part2 = Jacobian3_part2.evaluate()
        self.N31_psi_Jacobian4_part1 = Jacobian4_part1.evaluate()
        self.N31_psi_Jacobian4_part2 = Jacobian4_part2.evaluate()
        self.N31_psi_advective1 = advective1.evaluate()
        self.N31_psi_advective2 = advective2.evaluate()
        self.N31_psi_advective3 = advective3.evaluate()
        self.N31_psi_advective4 = advective4.evaluate()
        
        self.N31_psi = (Jacobian1_part1 + Jacobian1_part2 + Jacobian2_part1 + Jacobian2_part2 + Jacobian3_part1 + Jacobian3_part2 
                       + Jacobian4_part1 + Jacobian4_part2 + advective1 + advective2 + advective3 + advective4).evaluate()
                       
        self.N31_psi_noadvective = (Jacobian1_part1 + Jacobian1_part2 + Jacobian2_part1 + Jacobian2_part2 + Jacobian3_part1 + Jacobian3_part2 
                       + Jacobian4_part1 + Jacobian4_part2).evaluate()
                       
        # re-derivation, N31_psi
        # u dot del u
        n31_psi_psi11_dot_psi20 = (3*invr**4*(1j*Q)*o1.psi*o2.psi20_r + invr**3*(1j*Q)*o1.psi_r*o2.psi20_r - 3*invr**3*(1j*Q)*o1.psi*o2.psi20_rr
                                 + invr**2*(1j*Q)*o1.psi*o2.psi20_rrr)
        n31_psi_psi11_dot_psi20_star = (3*invr**4*(1j*Q)*o1.psi*o2.psi20_star_r + invr**3*(1j*Q)*o1.psi_r*o2.psi20_star_r - 3*invr**3*(1j*Q)*o1.psi*o2.psi20_star_rr
                                 + invr**2*(1j*Q)*o1.psi*o2.psi20_star_rrr)
        n31_psi_psi11_star_dot_psi22 = (-2*invr**3*(-1j*Q)**2*o1.psi_star*(2*1j*Q)*o2.psi22 + invr**2*(-1j*Q)*o1.psi_star*(2*1j*Q)**2*o2.psi22_r - invr**2*o1.psi_star_r*(2*1j*Q)**3*o2.psi22
                                        + 3*invr**4*(-1j*Q)*o1.psi_star*o2.psi22_r + invr**3*(-1j*Q)*o1.psi_star_r*o2.psi22_r - 3*invr**3*(-1j*Q)*o1.psi_star*o2.psi22_rr
                                        + invr**2*(-1j*Q)*o1.psi_star*o2.psi22_rrr - invr**2*o1.psi_star_r*(2*1j*Q)*o2.psi22_rr 
                                        - 2*invr*o1.u_star*(2*1j*Q)*o2.u22)
        n31_psi_psi20_dot_psi11 = (-invr**2*o2.psi20_r*(1j*Q)**3*o1.psi - invr**2*o2.psi20_r*(1j*Q)*o1.psi_rr
                                   - 2*invr*o2.u20*(1j*Q)*o1.u)
        n31_psi_psi20_star_dot_psi11 = (-invr**2*o2.psi20_star_r*(1j*Q)**3*o1.psi - invr**2*o2.psi20_star_r*(1j*Q)*o1.psi_rr
                                   - 2*invr*o2.u20_star*(1j*Q)*o1.u)
        n31_psi_psi22_dot_psi11_star = (-2*invr**3*(2*1j*Q)**2*o2.psi22*(-1j*Q)*o1.psi_star + invr**2*(2*1j*Q)*o2.psi22*(-1j*Q)**2*o1.psi_star_r - invr**2*o2.psi22_r*(-1j*Q)**3*o1.psi_star
                                        + 3*invr**4*(2*1j*Q)*o2.psi22*o1.psi_star_r + invr**3*(2*1j*Q)*o2.psi22_r*o1.psi_star_r - 3*invr**3*(2*1j*Q)*o2.psi22*o1.psi_star_rr
                                        + invr**2*(2*1j*Q)*o2.psi22*o1.psi_star_rrr - invr**2*o2.psi22_r*(-1j*Q)*o1.psi_star_rr 
                                        - 2*invr*o2.u22*(-1j*Q)*o1.u_star)
        # - (2/beta) B dot del B                                
        n31_psi_A11_dot_A20 = (3*invr**4*(1j*Q)*o1.A*o2.A20_r + invr**3*(1j*Q)*o1.A_r*o2.A20_r - 3*invr**3*(1j*Q)*o1.A*o2.A20_rr
                                 + invr**2*(1j*Q)*o1.A*o2.A20_rrr)
        n31_psi_A11_dot_A20_star = (3*invr**4*(1j*Q)*o1.A*o2.A20_star_r + invr**3*(1j*Q)*o1.A_r*o2.A20_star_r - 3*invr**3*(1j*Q)*o1.A*o2.A20_star_rr
                                 + invr**2*(1j*Q)*o1.A*o2.A20_star_rrr)
        n31_psi_A11_star_dot_A22 = (-2*invr**3*(-1j*Q)**2*o1.A_star*(2*1j*Q)*o2.A22 + invr**2*(-1j*Q)*o1.A_star*(2*1j*Q)**2*o2.A22_r - invr**2*o1.A_star_r*(2*1j*Q)**3*o2.A22
                                        + 3*invr**4*(-1j*Q)*o1.A_star*o2.A22_r + invr**3*(-1j*Q)*o1.A_star_r*o2.A22_r - 3*invr**3*(-1j*Q)*o1.A_star*o2.A22_rr
                                        + invr**2*(-1j*Q)*o1.A_star*o2.A22_rrr - invr**2*o1.A_star_r*(2*1j*Q)*o2.A22_rr 
                                        - 2*invr*o1.B_star*(2*1j*Q)*o2.B22)
        n31_psi_A20_dot_A11 = (-invr**2*o2.A20_r*(1j*Q)**3*o1.A - invr**2*o2.A20_r*(1j*Q)*o1.A_rr
                                   - 2*invr*o2.B20*(1j*Q)*o1.B)
        n31_psi_A20_star_dot_A11 = (-invr**2*o2.A20_star_r*(1j*Q)**3*o1.A - invr**2*o2.A20_star_r*(1j*Q)*o1.A_rr
                                   - 2*invr*o2.B20_star*(1j*Q)*o1.B)
        n31_psi_A22_dot_A11_star = (-2*invr**3*(2*1j*Q)**2*o2.A22*(-1j*Q)*o1.A_star + invr**2*(2*1j*Q)*o2.A22*(-1j*Q)**2*o1.A_star_r - invr**2*o2.A22_r*(-1j*Q)**3*o1.A_star
                                        + 3*invr**4*(2*1j*Q)*o2.A22*o1.A_star_r + invr**3*(2*1j*Q)*o2.A22_r*o1.A_star_r - 3*invr**3*(2*1j*Q)*o2.A22*o1.A_star_rr
                                        + invr**2*(2*1j*Q)*o2.A22*o1.A_star_rrr - invr**2*o2.A22_r*(-1j*Q)*o1.A_star_rr 
                                        - 2*invr*o2.B22*(-1j*Q)*o1.B_star)
        self.N31_psi = (n31_psi_psi11_dot_psi20 + n31_psi_psi11_dot_psi20_star + n31_psi_psi11_star_dot_psi22
                + n31_psi_psi20_dot_psi11 + n31_psi_psi20_star_dot_psi11 + n31_psi_psi22_dot_psi11_star
                -(2/beta)*(n31_psi_A11_dot_A20 + n31_psi_A11_dot_A20_star + n31_psi_A11_star_dot_A22
                + n31_psi_A20_dot_A11 + n31_psi_A20_star_dot_A11 + n31_psi_A22_dot_A11_star)).evaluate()
        
        
        # (1/r) J(psi1, u2)               
        Jacobian1 = invr*(1j*k*o1.psi*(o2.u20_r + o2.u20_star_r) + (-1j*k)*o1.psi_star*o2.u22_r - o1.psi_star_r*(2*1j*k)*o2.u22)
        
        # -(1/r) (2/beta) J(A1, B2)
        Jacobian2 = -invr*(2/self.beta)*(1j*k*o1.A*(o2.B20_r + o2.B20_star_r) + (-1j*k)*o1.A_star*o2.B22_r - o1.A_star_r*(2*1j*k)*o2.B22)
        
        # (1/r) J(psi2, u1)
        Jacobian3 = invr*(-o2.psi20_r*1j*k*o1.u - o2.psi20_star_r*1j*k*o1.u + 2*1j*k*o2.psi22*o1.u_star_r - o2.psi22_r*(-1j*k)*o1.u_star)
        
        # -(1/r) (2/beta) J(A2, B1)
        Jacobian4 = -invr*(2/self.beta)*(-o2.A20_r*1j*k*o1.B - o2.A20_star_r*1j*k*o1.B + 2*1j*k*o2.A22*o1.B_star_r - o2.A22_r*(-1j*k)*o1.B_star)
        
        # (1/r^2) u1 dz psi2
        advective1 = invr**2*(o1.u_star*2*1j*k*o2.psi22)
        
        # (1/r^2) u2 dz psi1
        advective2 = invr**2*(o2.u20*1j*k*o1.psi + o2.u20_star*1j*k*o1.psi + o2.u22*(-1j*k)*o1.psi_star)
        
        # -(2/beta)(1/r^2) B1 dz A2
        advective3 = -(2/self.beta)*invr**2*(o1.B_star*2*1j*k*o2.A22)
        
        # - (2/beta)(1/r^2) B2 dz A1
        advective4 = -(2/self.beta)*invr**2*(o2.B20*1j*k*o1.A + o2.B20_star*1j*k*o1.A + o2.B22*(-1j*k)*o1.A_star)
        
        # diagnostics
        self.N31_u_Jacobian1 = Jacobian1.evaluate()
        self.N31_u_Jacobian2 = Jacobian2.evaluate()
        self.N31_u_Jacobian3 = Jacobian3.evaluate()
        self.N31_u_Jacobian4 = Jacobian4.evaluate()
        self.N31_u_advective1 = advective1.evaluate()
        self.N31_u_advective2 = advective2.evaluate()
        self.N31_u_advective3 = advective3.evaluate()
        self.N31_u_advective4 = advective4.evaluate()
        
        self.N31_u = (Jacobian1 + Jacobian2 + Jacobian3 + Jacobian4 + advective1 + advective2 + advective3 + advective4).evaluate()
        
        self.N31_u_noadvective = (Jacobian1 + Jacobian2 + Jacobian3 + Jacobian4).evaluate()
        
        # -(1/r) J(A1, psi2)
        Jacobian1 = -invr*(1j*k*o1.A*(o2.psi20_r + o2.psi20_star_r) + (-1j*k)*o1.A_star*o2.psi22_r - o1.A_star_r*(2*1j*k)*o2.psi22)
        
        # -(1/r) J(A2, psi1)
        Jacobian2 = -invr*(-o2.A20_r*1j*k*o1.psi - o2.A20_star_r*1j*k*o1.psi + 2*1j*k*o2.A22*o1.psi_star_r - o2.A22_r*(-1j*k)*o1.psi_star)
        
        self.N31_A_Jacobian1 = Jacobian1.evaluate()
        self.N31_A_Jacobian2 = Jacobian2.evaluate()
        
        self.N31_A = (Jacobian1 + Jacobian2).evaluate()
        
        # -(1/r) J(A1, u2)
        Jacobian1 = -invr*(1j*k*o1.A*(o2.u20_r + o2.u20_star_r) + (-1j*k)*o1.A_star*o2.u22_r - o1.A_star_r*(2*1j*k)*o2.u22)
        
        # -(1/r) J(B1, psi2)
        Jacobian2 = -invr*(1j*k*o1.B*(o2.psi20_r + o2.psi20_star_r) + (-1j*k)*o1.B_star*o2.psi22_r - o1.B_star_r*(2*1j*k)*o2.psi22)
        
        # -(1/r) J(A2, u1)
        Jacobian3 = -invr*(-o2.A20_r*1j*k*o1.u - o2.A20_star_r*1j*k*o1.u + 2*1j*k*o2.A22*o1.u_star_r - o2.A22_r*(-1j*k)*o1.u_star)
        
        # -(1/r) J(B2, psi1)
        Jacobian4 = -invr*(-o2.B20_r*1j*k*o1.psi - o2.B20_star_r*1j*k*o1.psi + 2*1j*k*o2.B22*o1.psi_star_r - o2.B22_r*(-1j*k)*o1.psi_star)
        
        # -(1/r^2) B1 partial_z psi2
        advective1 = -invr**2*(o1.B_star*2*1j*k*o2.psi22)
        
        # +(1/r^2) u1 partial_z A2
        advective2 = invr**2*(o1.u_star*2*1j*k*o2.A22) # fixed sign 7/14/16
        
        # -(1/r^2) B2 partial_z psi1
        advective3 = -invr**2*(o2.B20*1j*k*o1.psi + o2.B20_star*1j*k*o1.psi + o2.B22*(-1j*k)*o1.psi_star)
        
        # +(1/r^2) u2 partial_z A1
        advective4 = invr**2*(o2.u20*1j*k*o1.A + o2.u20_star*1j*k*o1.A + o2.u22*(-1j*k)*o1.A_star) # fixed sign 7/14/16
        
        # diagnostics
        self.N31_B_Jacobian1 = Jacobian1.evaluate()
        self.N31_B_Jacobian2 = Jacobian2.evaluate()
        self.N31_B_Jacobian3 = Jacobian3.evaluate()
        self.N31_B_Jacobian4 = Jacobian4.evaluate()
        self.N31_B_advective1 = advective1.evaluate()
        self.N31_B_advective2 = advective2.evaluate()
        self.N31_B_advective3 = advective3.evaluate()
        self.N31_B_advective4 = advective4.evaluate()
        
        self.N31_B = (Jacobian1 + Jacobian2 + Jacobian3 + Jacobian4 + advective1 + advective2 + advective3 + advective4).evaluate()
        
        self.N31_B_noadvective = (Jacobian1 + Jacobian2 + Jacobian3 + Jacobian4).evaluate()


class AmplitudeAlpha(MRI):

    """
    Solves the coefficients of the first amplitude equation for alpha -- e^(iQz) terms.
    
    """
    
    def __init__(self, domain, o1 = None, o2 = None, Q = 0.01795, Rm = 0.84043, Pm = 0.001, beta = 25.0, Omega1 = 313.55, Omega2 = 67.0631, xi = 0, norm = True):
        
        logger.info("initializing Amplitude Alpha")
      
        if o1 == None:
            o1 = OrderE(domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
            MRI.__init__(self, domain, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
            n2 = N2(domain, o1 = o1, Q = Q, Rm = Rm, Pm = Pm, beta = beta, Omega1 = Omega1, Omega2 = Omega2, xi = xi, norm = norm)
        else:
            MRI.__init__(self, domain, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, beta = o1.beta, Omega1 = o1.Omega1, Omega2 = o1.Omega2, xi = o1.xi, norm = o1.norm)
            n2 = N2(domain, o1 = o1, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, beta = o1.beta, Omega1 = o1.Omega1, Omega2 = o1.Omega2, xi = o1.xi, norm = o1.norm)

        ah = AdjointHomogenous(domain, o1 = o1, Q = self.Q, Rm = self.Rm, Pm = self.Pm, beta = self.beta, Omega1 = self.Omega1, Omega2 = self.Omega2, xi = self.xi, norm = self.norm)
        if o2 == None:
            o2 = OrderE2(domain, o1 = o1, ah=ah, Q = self.Q, Rm = self.Rm, Pm = self.Pm, beta = self.beta, Omega1 = self.Omega1, Omega2 = self.Omega2, xi = self.xi, norm = self.norm)
        
        n3 = N3(domain, o1 = o1, o2 = o2, ah=ah, Q = self.Q, Rm = self.Rm, Pm = self.Pm, beta = self.beta, Omega1 = self.Omega1, Omega2 = self.Omega2, xi = self.xi, norm = self.norm)
            
        rfield = self.domain.new_field()
        rfield['g'] = o1.r
        
        invr = self.domain.new_field()
        invr['g'] = (1/o1.r)
        
        # u0 = r Omega(r) = Ar + B/r
        u0 = self.domain.new_field()
        u0['g'] = self.c1*rfield['g'] + self.c2*(1/rfield['g'])
        
        du0 = self.domain.new_field()
        du0['g'] = self.c1 - self.c2*(1/rfield['g']**2)
        
        # D . V11
        a_psi_rhs = invr*o1.psi_rr - invr**2*o1.psi_r - invr*self.Q**2*o1.psi
        a_psi_rhs = a_psi_rhs.evaluate()
        
        # Gtwiddle . V11 - xi dz H . V11
        b_psi_rhs = -(2/self.beta)*invr*1j*self.Q**3*o1.A + (2/self.beta)*invr*1j*self.Q*o1.A_rr - (2/self.beta)*invr**2*1j*self.Q*o1.A_r - (2/self.beta)*invr**2*2*1j*self.Q*self.xi*o1.B
        b_psi_rhs = b_psi_rhs.evaluate()
        
        b_u_rhs = (2/self.beta)*1j*self.Q*o1.B
        b_u_rhs = b_u_rhs.evaluate()
        
        b_A_rhs = 1j*self.Q*o1.psi
        b_A_rhs = b_A_rhs.evaluate()
        
        b_B_rhs = 1j*self.Q*o1.u + invr**3*2*1j*self.Q*self.xi*o1.psi
        b_B_rhs = b_B_rhs.evaluate()
        
        # (L1twiddle V21 + L2twiddle V11 + xi H V21)
        h_psi_rhs = (self.iR*invr*4*1j*self.Q**3*o2.psi21 + (2/self.beta)*invr*3*self.Q**2*o2.A21 + self.iR*invr*6*self.Q**2*o1.psi - (2/self.beta)*invr*3*1j*self.Q*o1.A
                    - self.iR*invr*4*1j*self.Q*o2.psi21_rr + self.iR*invr**2*4*1j*self.Q*o2.psi21_r - invr*2*u0*o2.u21 - (2/self.beta)*invr*o2.A21_rr
                    + (2/self.beta)*invr**2*o2.A21_r + (2/self.beta)*invr**2*2*self.xi*o2.B21 - self.iR*invr*2*o1.psi_rr + self.iR*invr**2*2*o1.psi_r)
        h_psi_rhs = h_psi_rhs.evaluate()
        
        h_u_rhs = -self.iR*2*1j*self.Q*o2.u21 + invr*du0*o2.psi21 + invr**2*u0*o2.psi21 - (2/self.beta)*o2.B21 - self.iR*o1.u
        h_u_rhs = h_u_rhs.evaluate()
        
        h_A_rhs = -self.iRm*2*1j*self.Q*o2.A21 - o2.psi21 - self.iRm*o1.A
        h_A_rhs = h_A_rhs.evaluate()
        
        h_B_rhs = -self.iRm*2*1j*self.Q*o2.B21 - invr*du0*o2.A21 - o2.u21 + invr**2*u0*o2.A21 - invr**3*2*self.xi*o2.psi21 - self.iRm*o1.B
        h_B_rhs = h_B_rhs.evaluate()
        
        # Normalize s.t. a = 1
        if norm is True:
            logger.info("Normalizing V^dagger s.t. a = 1")
            ah.psi, ah.u, ah.A, ah.B = self.normalize_inner_product_eq_1(ah.psi, ah.u, ah.A, ah.B, a_psi_rhs, o1.u, o1.A, o1.B)
        
        # a = <va . D V11*>
        self.a = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [a_psi_rhs, o1.u, o1.A, o1.B])
        
        # c = <va . N31*>
        self.c = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [n3.N31_psi, n3.N31_u, n3.N31_A, n3.N31_B])
        
        # b = < va . (Gtwiddle v11 - xi*dz*H v11)* > 
        self.b = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [b_psi_rhs, b_u_rhs, b_A_rhs, b_B_rhs])
  
        # h = < va . ( L1twiddle V21 + L2twiddle V11 + xi H V21)* > 
        self.h = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [h_psi_rhs, h_u_rhs, h_A_rhs, h_B_rhs])
  
        # linear term alias
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
        logger.info("a = {}; c = {}; b = {}; h = {}".format(self.a, self.c, self.b, self.h))
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
