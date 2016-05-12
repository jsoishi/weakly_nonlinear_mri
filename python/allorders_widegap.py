import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
from scipy.linalg import eig, norm
import pylab
import copy
import pickle
#import plot_tools
import streamplot_uneven as su
import random

import logging 
#root = logging.root 
#for h in root.handlers: 
#    h.setLevel("DEBUG") 

logger = logging.getLogger(__name__)

import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

nr1 = 16#128#64
r1 = de.Chebyshev('r', nr1, interval=(5, 15))
d1 = de.Domain([r1])

nr2 = 24#192#128
r2 = de.Chebyshev('r', nr2, interval=(5, 15))
d2 = de.Domain([r2])

print("grid number {}, spurious eigenvalue check at {}".format(nr1, nr2))

class MRI():

    """
    Base class for MRI equations.
    
    Defaults: For Pm of 0.001 critical Rm is 4.879  critical Q is 0.748
    """

    def __init__(self, Q = np.pi/10, Rm = 4.052, Pm = 1.6E-6, beta = 0.4378, R1 = 9.5, R2 = 10.5, Omega1 = 314, Omega2 = 37.9, norm = True):
    
        self.Q = Q
        self.Rm = Rm
        self.Pm = Pm
        self.beta = beta
        self.norm = norm
        
        # Inverse magnetic reynolds number
        self.iRm = 1.0/self.Rm
        
        # Reynolds number
        self.R = self.Rm/self.Pm
        self.iR = 1.0/self.R
        
        self.gridnum1 = nr1
        self.gridnum2 = nr2
        self.r = r1.grid()
        
        self.R1 = R1
        self.R2 = R2
        self.Omega1 = Omega1
        self.Omega2 = Omega2
        
        self.R0 = (self.R1 + self.R2)/2.0 # center of channel

        self.c1 = (self.Omega2*self.R2**2 - self.Omega1*self.R1**2)/(self.R2**2 - self.R1**2)
        self.c2 = (self.R1**2*self.R2**2*(self.Omega1 - self.Omega2))/(self.R2**2 - self.R1**2)
        
        self.zeta_mean = 2*(self.R2**2*self.Omega2 - self.R1**2*self.Omega1)/((self.R2**2 - self.R1**2)*np.sqrt(self.Omega1*self.Omega2))
    
        print("MRI parameters: ", self.Q, self.Rm, self.Pm, self.beta, "R1 = ", self.R1, "R2 = ", self.R2, 'norm = ', norm, "Reynolds number", self.R)
        
        
    def set_boundary_conditions(self, problem):
        
        """
        Adds MRI problem boundary conditions to a ParsedProblem object.
        """
        
        problem.add_bc('left(u) = 0')
        problem.add_bc('right(u) = 0')
        problem.add_bc('left(psi) = 0')
        problem.add_bc('right(psi) = 0')
        problem.add_bc('left(A) = 0')
        problem.add_bc('right(A) = 0')
        #problem.add_bc('left(psi + r*psir) = 0')
        #problem.add_bc('right(psi + r*psir) = 0')
        problem.add_bc('left(psir) = 0')
        problem.add_bc('right(psir) = 0')
        problem.add_bc('left(B + r*Br) = 0')
        problem.add_bc('right(B + r*Br) = 0') # axial component of current = 0
        
        return problem
    
    def solve_LEV(self, problem):
    
        """
        Solves the linear eigenvalue problem for a ParsedProblem object.
        """
        
        problem.expand(domain)
        LEV = LinearEigenvalue(problem, domain)
        print(LEV.pencils[0].L)
        LEV.solve(LEV.pencils[0])
        
        return LEV
        
    def solve_LEV_secondgrid(self, problem):
    
        """
        Solves the linear eigenvalue problem for a ParsedProblem object.
        Uses gridnum = 192 domain. For use in discarding spurious eigenvalues.
        """
        
        problem.expand(domain192)
        LEV = LinearEigenvalue(problem, domain192)
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
        
        
    def discard_spurious_eigenvalues(self, lambda1, lambda2):
    
        """
        lambda1 :: eigenvalues from low res run
        lambda2 :: eigenvalues from high res run
        
        Solves the linear eigenvalue problem for two different resolutions.
        Returns trustworthy eigenvalues using nearest delta, from Boyd chapter 7.
        """

        # Reverse engineer correct indices to make unsorted list from sorted
        reverse_lambda1_indx = np.arange(len(lambda1)) 
        reverse_lambda2_indx = np.arange(len(lambda2))
    
        lambda1_and_indx = np.asarray(list(zip(lambda1, reverse_lambda1_indx)))
        lambda2_and_indx = np.asarray(list(zip(lambda2, reverse_lambda2_indx)))
        
        # remove nans
        lambda1_and_indx = lambda1_and_indx[np.isfinite(lambda1)]
        lambda2_and_indx = lambda2_and_indx[np.isfinite(lambda2)]
    
        # Sort lambda1 and lambda2 by real parts
        lambda1_and_indx = lambda1_and_indx[np.argsort(lambda1_and_indx[:, 0].real)]
        lambda2_and_indx = lambda2_and_indx[np.argsort(lambda2_and_indx[:, 0].real)]
        
        lambda1_sorted = lambda1_and_indx[:, 0]
        lambda2_sorted = lambda2_and_indx[:, 0]
    
        # Compute sigmas from lower resolution run (gridnum = N1)
        sigmas = np.zeros(len(lambda1_sorted))
        sigmas[0] = np.abs(lambda1_sorted[0] - lambda1_sorted[1])
        sigmas[1:-1] = [0.5*(np.abs(lambda1_sorted[j] - lambda1_sorted[j - 1]) + np.abs(lambda1_sorted[j + 1] - lambda1_sorted[j])) for j in range(1, len(lambda1_sorted) - 1)]
        sigmas[-1] = np.abs(lambda1_sorted[-2] - lambda1_sorted[-1])

        if not (np.isfinite(sigmas)).all():
            print("WARNING: at least one eigenvalue spacings (sigmas) is non-finite (np.inf or np.nan)!")
    
        # Nearest delta
        delta_near = np.array([np.nanmin(np.abs(lambda1_sorted[j] - lambda2_sorted)/sigmas[j]) for j in range(len(lambda1_sorted))])
    
        # Discard eigenvalues with 1/delta_near < 10^6
        lambda1_and_indx = lambda1_and_indx[np.where((1.0/delta_near) > 1E6)]
        #print(lambda1_and_indx)
        
        lambda1 = lambda1_and_indx[:, 0]
        indx = lambda1_and_indx[:, 1]
        
        return lambda1, indx

    def get_largest_real_eigenvalue_index(self, LEV, goodevals = None, goodevals_indx = None):
        
        """
        Return index of largest eigenvalue. Can be positive or negative.
        """
        
        if goodevals == None:
            evals = LEV.eigenvalues
        else:
            evals = goodevals
        
        largest_eval_pseudo_indx = np.nanargmax(goodevals.real)
        largest_eval_indx = goodevals_indx[largest_eval_pseudo_indx]
        
        print("largest eigenvalue indx", largest_eval_indx)
        
        return largest_eval_indx
        
    def get_derivative(self, field):
    
        """
        Take derivative of a single field.
        """
        
        field_x = field.differentiate(0)
        
        #if field.name.endswith("x"):
        #    field_x.name = field.name + "x"
        #else:
        #    field_x.name = field.name + "_x"
            
        return field_x
        
    def get_complex_conjugate(self, field):
        
        """
        Take complex conjugate of a single field.
        """
        
        field_star = d1.new_field()
        #field_star.name = field.name + "_star"
        field_star['g'] = field['g'].conj()
        
        return field_star
        
class OrderE(MRI):

    """
    Solves the order(epsilon) equation L V_1 = 0
    This is simply the linearized wide-gap MRI.
    Returns V_1
    """

    def __init__(self, Q = np.pi/10, Rm = 4.052, Pm = 1.6E-6, beta = 0.4378, R1 = 9.5, R2 = 10.5, Omega1 = 314, Omega2 = 37.9, norm = True, inviscid = False):
        
        print("initializing wide gap Order epsilon")
        
        MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, beta = beta, R1 = R1, R2 = R2, Omega1 = Omega1, Omega2 = Omega2, norm = norm)
    
        # widegap order epsilon
        widegap1 = de.EVP(d1,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],'sigma')
        widegap2 = de.EVP(d2,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],'sigma')
        
        # Rm and Pm are input parameters
        iRm = 1./Rm
        R = Rm/Pm
        iR = 1./R
        
        # Add equations
        for widegap in [widegap1, widegap2]:
            
            widegap.parameters['k'] = Q
            widegap.parameters['iR'] = iR
            widegap.parameters['iRm'] = iRm
            widegap.parameters['c1'] = self.c1
            widegap.parameters['c2'] = self.c2
            widegap.parameters['beta'] = beta
            widegap.parameters['B0'] = 1

            widegap.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
            widegap.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
            widegap.substitutions['twooverbeta'] = '(2.0/beta)'
            widegap.substitutions['psivisc'] = '(2*r**2*k**2*psir - 2*r**3*k**2*psirr + r**3*k**4*psi + r**3*dr(psirrr) - 3*psir + 3*r*psirr - 2*r**2*psirrr)'
            widegap.substitutions['uvisc'] = '(-r**3*k**2*u + r**3*dr(ur) + r**2*ur - r*u)'
            widegap.substitutions['Avisc'] = '(r*dr(Ar) - r*k**2*A - Ar)' 
            widegap.substitutions['Bvisc'] = '(-r**3*k**2*B + r**3*dr(Br) + r**2*Br - r*B)'
        
            widegap.add_equation("sigma*(-r**3*k**2*psi + r**3*psirr - r**2*psir) - r**2*2*ru0*1j*k*u + r**3*twooverbeta*B0*1j*k**3*A + twooverbeta*B0*r**2*1j*k*Ar - twooverbeta*r**3*B0*1j*k*dr(Ar) - iR*psivisc = 0") #corrected on whiteboard 5/6
            widegap.add_equation("sigma*r**3*u + 1j*k*ru0*psi + 1j*k*rrdu0*psi - 1j*k*r**3*twooverbeta*B0*B - iR*uvisc = 0") 
            widegap.add_equation("sigma*r*A - r*B0*1j*k*psi - iRm*Avisc = 0")
            widegap.add_equation("sigma*r**3*B + ru0*1j*k*A - r**3*B0*1j*k*u - 1j*k*rrdu0*A - iRm*Bvisc = 0") 

            widegap.add_equation("dr(psi) - psir = 0")
            widegap.add_equation("dr(psir) - psirr = 0")
            widegap.add_equation("dr(psirr) - psirrr = 0")
            widegap.add_equation("dr(u) - ur = 0")
            widegap.add_equation("dr(A) - Ar = 0")
            widegap.add_equation("dr(B) - Br = 0")
        
            widegap = self.set_boundary_conditions(widegap)
        
        solver1 = widegap1.build_solver()
        solver2 = widegap2.build_solver()
        
        solver1.solve(solver1.pencils[0])
        solver2.solve(solver2.pencils[0])
        
        # Discard spurious eigenvalues
        ev1 = solver1.eigenvalues
        ev2 = solver2.eigenvalues
        self.goodeigs, self.goodeigs_indices = self.discard_spurious_eigenvalues(ev1, ev2)

        #goodeigs_index = np.where(self.goodeigs.real == np.nanmax(self.goodeigs.real))[0][0]
        print(self.goodeigs)
        try:
            goodeigs_index = np.nanargmax(self.goodeigs.real)
            self.marginal_mode_index = int(self.goodeigs_indices[goodeigs_index])
        
            solver1.set_state(self.marginal_mode_index)
        
            self.solver1 = solver1
        
            self.psi = self.solver1.state['psi']
            self.u = self.solver1.state['u']
            self.A = self.solver1.state['A']
            self.B = self.solver1.state['B']
            
            # Take all relevant derivates for use with higher order terms
            self.psi_r = self.solver1.state['psir']
            self.psi_rr = self.solver1.state['psirr']
            self.psi_rrr = self.solver1.state['psirrr']
            #self.psi_rrrr = self.get_derivative(self.psi_rrr)
      
            self.u_r = self.solver1.state['ur']
        
            self.A_r = self.solver1.state['Ar']
            self.A_rr = self.get_derivative(self.A_r)
            self.A_rrr = self.get_derivative(self.A_rr)
        
            self.B_r = self.solver1.state['Br']
            
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
            
        except ValueError:
            print("No good eigenvalues found!!!")
            self.solver1 = solver1
    
class N2(MRI):

    """
    Solves the nonlinear term N2
    Returns N2
    
    """
    
    def __init__(self, o1 = None, Q = np.pi/10, Rm = 4.052, Pm = 1.6E-6, beta = 0.4378, R1 = 9.5, R2 = 10.5, Omega1 = 314, Omega2 = 37.9, norm = True):
    
        print("initializing N2")
    
        if o1 == None:
            o1 = OrderE(Q = Q, Rm = Rm, Pm = Pm, beta = beta, R1 = R1, R2 = R2, Omega1 = Omega1, Omega2 = Omega2, norm = norm)
            self.o1 = o1
            MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, beta = beta, R1 = R1, R2 = R2, Omega1 = Omega1, Omega2 = Omega2, norm = norm)
        else:
            MRI.__init__(self, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, beta = o1.beta, R1 = o1.R1, R2 = o1.R2, Omega1 = o1.Omega1, Omega2 = o1.Omega2, norm = o1.norm)
    
        rfield = d1.new_field()
        rfield['g'] = o1.r
    
        N22_psi = ((1j*Q*o1.psi)*((1/rfield**2)*(-Q**2)*o1.psi_r - (3/rfield**3)*o1.psi_rr + (1/rfield**2)*o1.psi_rrr - (2/rfield**3)*(-Q**2)*o1.psi + (3/rfield**4)*o1.psi_r)
                - (o1.psi_r)*((1/rfield**2)*(1j*Q)*o1.psi_rr - (1/rfield**3)*(1j*Q)*o1.psi_r + (1/rfield**2)*(1j*Q)**3*o1.psi)
                - (2/beta)*((1j*Q*o1.A)*((1/rfield**2)*(-Q**2)*o1.A_r - (3/rfield**3)*o1.A_rr + (1/rfield**2)*o1.A_rrr - (2/rfield**3)*(-Q**2)*o1.A + (3/rfield**4)*o1.A_r)
                - (o1.A_r)*((1/rfield**2)*(1j*Q)*o1.A_rr - (1/rfield**3)*(1j*Q)*o1.A_r + (1/rfield**2)*(1j*Q)**3*o1.A))
                -(2/rfield)*o1.u*(1j*Q)*o1.u + (2/beta)*(2/rfield)*o1.B*(1j*Q)*o1.B)
        self.N22_psi = N22_psi.evaluate()
        
        N20_psi = ((1j*Q*o1.psi)*((1/rfield**2)*((-1j*Q)**2)*o1.psi_star_r - (3/rfield**3)*o1.psi_star_rr + (1/rfield**2)*o1.psi_star_rrr - (2/rfield**3)*((-1j*Q)**2)*o1.psi_star + (3/rfield**4)*o1.psi_star_r)
                - (o1.psi_r)*((1/rfield**2)*(-1j*Q)*o1.psi_star_rr - (1/rfield**3)*(-1j*Q)*o1.psi_star_r + (1/rfield**2)*(-1j*Q)**3*o1.psi_star)
                - (2/beta)*((1j*Q*o1.A)*((1/rfield**2)*((-1j*Q)**2)*o1.A_star_r - (3/rfield**3)*o1.A_star_rr + (1/rfield**2)*o1.A_star_rrr - (2/rfield**3)*((-1j*Q)**2)*o1.A_star + (3/rfield**4)*o1.A_star_r)
                - (o1.A_r)*((1/rfield**2)*(-1j*Q)*o1.A_star_rr - (1/rfield**3)*(-1j*Q)*o1.A_star_r + (1/rfield**2)*(-1j*Q)**3*o1.A_star))
                -(2/rfield)*o1.u*(-1j*Q)*o1.u_star + (2/beta)*(2/rfield)*o1.B*(-1j*Q)*o1.B_star)
        self.N20_psi = N20_psi.evaluate()
                
        N22_u = ((1/rfield)*((1j*Q*o1.psi)*o1.u_r - (1j*Q*o1.u)*o1.psi_r) - ((1/rfield)*(2/beta)*((1j*Q*o1.A)*o1.B_r - (1j*Q*o1.B)*o1.A_r))
                + (1/rfield**2)*o1.u*1j*Q*o1.psi - (2/beta)*(1/rfield**2)*o1.B*1j*Q*o1.A)
        self.N22_u = N22_u.evaluate()
        
        N20_u = ((1/rfield)*((1j*Q*o1.psi)*o1.u_star_r - (-1j*Q*o1.u_star)*o1.psi_r) - ((1/rfield)*(2/beta)*((1j*Q*o1.A)*o1.B_star_r - (-1j*Q*o1.B_star)*o1.A_r))
                + (1/rfield**2)*o1.u*(-1j*Q)*o1.psi_star - (2/beta)*(1/rfield**2)*o1.B*(-1j*Q)*o1.A_star)
        self.N20_u = N20_u.evaluate()
                
        N22_A = (1/rfield)*((1j*Q*o1.psi)*o1.A_r - (1j*Q*o1.A)*o1.psi_r)
        self.N22_A = N22_A.evaluate()
        
        N20_A = (1/rfield)*((1j*Q*o1.psi)*o1.A_star_r - (-1j*Q*o1.A_star)*o1.psi_r)
        self.N20_A = N20_A.evaluate()
        
        N22_B = ((1/rfield)*((1j*Q*o1.u)*o1.A_r - (1j*Q*o1.A)*o1.u_r) + (1/rfield)*((1j*Q*o1.psi)*o1.B_r - (1j*Q*o1.B)*o1.psi_r)
                - (1/rfield**2)*o1.B*(1j*Q)*o1.psi + (1/rfield**2)*o1.u*(1j*Q)*o1.A)
        self.N22_B = N22_B.evaluate()
        
        N20_B = ((1/rfield)*((1j*Q*o1.u)*o1.A_star_r - (-1j*Q*o1.A_star)*o1.u_r) + (1/rfield)*((1j*Q*o1.psi)*o1.B_star_r - (-1j*Q*o1.B_star)*o1.psi_r)
                - (1/rfield**2)*o1.B*(-1j*Q)*o1.psi_star + (1/rfield**2)*o1.u*(-1j*Q)*o1.A_star)
        self.N20_B = N20_B.evaluate()
        
        
class OrderE2(MRI):

    """
    Solves the second order equation L V2 = -N2 - Ltwiddle V1 (note matrices are defined for LHS of eqn).
    Returns V2
    
    """
    
    def __init__(self, o1 = None, Q = np.pi/10, Rm = 4.052, Pm = 1.6E-6, beta = 0.4378, R1 = 9.5, R2 = 10.5, Omega1 = 314, Omega2 = 37.9, norm = True):
    
        print("initializing Order E2")
        
        if o1 == None:
            #o1 = OrderE(Q = Q, Rm = Rm, Pm = Pm, beta = beta, R1 = R1, R2 = R2, Omega1 = Omega1, Omega2 = Omega2, norm = norm)
            MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, beta = beta, R1 = R1, R2 = R2, Omega1 = Omega1, Omega2 = Omega2, norm = norm)
            n2 = N2(Q = Q, Rm = Rm, Pm = Pm, beta = beta, R1 = R1, R2 = R2, Omega1 = Omega1, Omega2 = Omega2, norm = norm)
            o1 = n2.o1
        else:
            MRI.__init__(self, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, beta = o1.beta, R1 = o1.R1, R2 = o1.R2, Omega1 = o1.Omega1, Omega2 = o1.Omega2, norm = o1.norm)
            n2 = N2(o1 = o1, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, beta = o1.beta, R1 = o1.R1, R2 = o1.R2, Omega1 = o1.Omega1, Omega2 = o1.Omega2, norm = o1.norm)
    
        V21 = de.LBVP(d1,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'])
        V20 = de.LBVP(d1,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'])
        V22 = de.LBVP(d1,['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'])
    
        for V in [V21, V20, V22]:
            V.parameters['k'] = self.Q
            V.parameters['iRm'] = self.iRm
            V.parameters['beta'] = self.beta
            V.parameters['c1'] = self.c1
            V.parameters['c2'] = self.c2
            V.parameters['iR'] = self.iR
            V.parameters['B0'] = 1
        
        # RHS of V21 = -L1twiddle V11
        rfield = d1.new_field()
        rfield['g'] = o1.r
        
        u0field = d1.new_field()
        u0field['g'] = self.c1*rfield['g'] + self.c2*(1/rfield['g'])
        
        du0field = d1.new_field()
        du0field['g'] = self.c1 - self.c2*(1/rfield['g']**2)
        
        rhs_psi21 = ((rfield**4)*((2/rfield)*u0field*o1.u + (2/self.beta)*(1/rfield)*o1.A_rr - (2/self.beta)*(1/rfield**2)*o1.A_r 
                    - (2/self.beta)*(3/rfield)*self.Q**2*o1.A + self.iR*4*1j*self.Q*(1/rfield)*o1.psi_rr - self.iR*(1/rfield**2)*4*1j*self.Q*o1.psi_r
                    - self.iR*(1/rfield)*4*1j*self.Q**3*o1.psi))
        self.rhs_psi21 = rhs_psi21.evaluate()
        
        rhs_u21 = ((rfield**3)*(-(1/rfield)*du0field*o1.psi - (1/rfield**2)*u0field*o1.psi + (2/self.beta)*o1.B + self.iR*2*1j*self.Q*o1.u))
        self.rhs_u21 = rhs_u21.evaluate()
        
        rhs_A21 = (rfield)*(o1.psi + self.iRm*2*1j*self.Q*o1.A)
        self.rhs_A21 = rhs_A21.evaluate()
        
        rhs_B21 = (rfield**3)*((1/rfield)*du0field*o1.A + o1.u - (1/rfield**2)*u0field*o1.A + self.iRm*2*1j*self.Q*o1.B)
        self.rhs_B21 = rhs_B21.evaluate()
        
        V21.parameters['rhs_psi21'] = self.rhs_psi21
        V21.parameters['rhs_u21'] = self.rhs_u21
        V21.parameters['rhs_A21'] = self.rhs_A21
        V21.parameters['rhs_B21'] = self.rhs_B21
        
        # LHS of V21 is the same as LHS of order (epsilon)
        V21.substitutions['ru0'] = '(r*r*c1 + c2)' # u0 = r Omega(r) = Ar + B/r
        V21.substitutions['rrdu0'] = '(c1*r*r-c2)' # du0/dr = A - B/r^2
        V21.substitutions['twooverbeta'] = '(2.0/beta)'
        V21.substitutions['psivisc'] = '(2*r**2*k**2*psir - 2*r**3*k**2*psirr + r**3*k**4*psi + r**3*dr(psirrr) - 3*psir + 3*r*psirr - 2*r**2*psirrr)'
        V21.substitutions['uvisc'] = '(-r**3*k**2*u + r**3*dr(ur) + r**2*ur - r*u)'
        V21.substitutions['Avisc'] = '(r*dr(Ar) - r*k**2*A - Ar)' 
        V21.substitutions['Bvisc'] = '(-r**3*k**2*B + r**3*dr(Br) + r**2*Br - r*B)'
    
        V21.add_equation("-r**2*2*ru0*1j*k*u + r**3*twooverbeta*B0*1j*k**3*A + twooverbeta*B0*r**2*1j*k*Ar - twooverbeta*r**3*B0*1j*k*dr(Ar) - iR*psivisc = rhs_psi21") #corrected on whiteboard 5/6
        V21.add_equation("1j*k*ru0*psi + 1j*k*rrdu0*psi - 1j*k*r**3*twooverbeta*B0*B - iR*uvisc = rhs_u21") 
        V21.add_equation("r*B0*1j*k*psi - iRm*Avisc = rhs_A21")
        V21.add_equation("ru0*1j*k*A - r**3*B0*1j*k*u - 1j*k*rrdu0*A - iRm*Bvisc = rhs_B21") 

        V21.add_equation("dr(psi) - psir = 0")
        V21.add_equation("dr(psir) - psirr = 0")
        V21.add_equation("dr(psirr) - psirrr = 0")
        V21.add_equation("dr(u) - ur = 0")
        V21.add_equation("dr(A) - Ar = 0")
        V21.add_equation("dr(B) - Br = 0")
        
        V21 = self.set_boundary_conditions(V21)
        V21solver = V21.build_solver()
        V21solver.solve()
        
        self.psi21 = V21solver.state['psi']
        self.u21 = V21solver.state['u']
        self.A21 = V21solver.state['A']
        self.B21 = V21solver.state['B']
        
        # LV20 = -N20
        
        self.rhs_psi20 = -n2.N20_psi
        self.rhs_u20 = -n2.N20_u
        self.rhs_A20 = -n2.N20_A
        self.rhs_B20 = -n2.N20_B
        
        V20.parameters['rhs_psi20'] = self.rhs_psi20
        V20.parameters['rhs_u20'] = self.rhs_u20
        V20.parameters['rhs_A20'] = self.rhs_A20
        V20.parameters['rhs_B20'] = self.rhs_B20
        
        V20.add_equation("-iR*(r**3*dr(psirrr) - r**2*2*psirrr + r*3*psirr - 3*psir) = rhs_psi20")
        V20.add_equation("-iR*(r**2*dr(ur) + r*ur - u) = rhs_u20")
        V20.add_equation("-iRm*(r*dr(Ar) - Ar) = rhs_A20")
        V20.add_equation("-iRm*(r**2*dr(Br) + r*Br - B) = rhs_B20")
        
        V20.add_equation("dr(psi) - psir = 0")
        V20.add_equation("dr(psir) - psirr = 0")
        V20.add_equation("dr(psirr) - psirrr = 0")
        V20.add_equation("dr(u) - ur = 0")
        V20.add_equation("dr(A) - Ar = 0")
        V20.add_equation("dr(B) - Br = 0")
        
        V20 = self.set_boundary_conditions(V20)
        V20solver = V20.build_solver()
        V20solver.solve()
        
        self.psi20 = V20solver.state['psi']
        self.u20 = V20solver.state['u']
        self.A20 = V20solver.state['A']
        self.B20 = V20solver.state['B']
        
        
        
        
"""
if __name__ == '__main__':

    #oe = OrderE()

    #root = "/home/sec2170/weakly_nonlinear_mri/weakly_nonlinear_mri/python/"
    #outfn = root + "multirun/widegap_orderEobj_"+str(nr1)+"_grid2_"+str(nr2)+"_Pm_"+str(oe.Pm)+"_Q_"+str(oe.Q)+"_Rm_"+str(oe.Rm)+"_beta"+str(oe.beta)+"_allgoodeigs.p"

    #pickle.dump(result_dict, open(outfn, "wb"))
    
    # Parameters approximating Umurhan+ 2007
    
    Pm = 1.0E-3
    beta = 25.0
    R1 = 5
    R2 = 15
    Omega1 = 313.55
    Omega2 = 67.0631
    Q = 0.748
    Rm = 4.879
    
    
    # Parameters approximating Umurhan+ 2007 "thin gap"
    
    #Pm = 1.0E-3
    #beta = 25.0
    #R1 = 9.5
    #R2 = 10.5
    #Omega1 = 314
    #Omega2 = 270.25
    #Q = 0.748
    #Rm = 4.879
    
    
    # Parameters approximating Goodman & Ji 2001    
    #Pm = 1.6E-6
    #beta = 0.43783886002604167#25.0
    #R1 = 5
    #R2 = 15
    #Omega1 = 314
    #Omega2 = 37.9

    c1 = (Omega2*R2**2 - Omega1*R1**2)/(R2**2 - R1**2)
    c2 = (R1**2*R2**2*(Omega1 - Omega2))/(R2**2 - R1**2)
    
    zeta_mean = 2*(R2**2*Omega2 - R1**2*Omega1)/((R2**2 - R1**2)*np.sqrt(Omega1*Omega2))
    
    print("mean zeta is {}, meaning q = 2 - zeta = {}".format(zeta_mean, 2 - zeta_mean))
    
    oe = OrderE(Q = Q, Rm = Rm, Pm = Pm, beta = beta, R1 = R1, R2 = R2, Omega1 = Omega1, Omega2 = Omega2)
"""
        