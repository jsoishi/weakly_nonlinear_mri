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

gridnum = 128
print("running at gridnum", gridnum)
x_basis = Chebyshev(gridnum)
domain = Domain([x_basis], grid_dtype=np.complex128)

# Second basis for checking eigenvalues
#x_basis192 = Chebyshev(64)
#x_basis192 = Chebyshev(64)
x_basis192 = Chebyshev(192)
domain192 = Domain([x_basis192], grid_dtype = np.complex128)

class MRI():

    """
    Base class for MRI equations.
    
    Defaults: For Pm of 0.001 critical Rm is 4.879  critical Q is 0.748
    """

    def __init__(self, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
    
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
        self.x = domain.grid(0)
    
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
        problem.add_left_bc("psir = 0")
        problem.add_right_bc("psir = 0")
        problem.add_left_bc("Br = 0")
        problem.add_right_bc("Br = 0")
        
        return problem
    
    def solve_LEV(self, problem):
    
        """
        Solves the linear eigenvalue problem for a ParsedProblem object.
        """
        
        problem.expand(domain)
        LEV = LinearEigenvalue(problem, domain)
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
        
        
    def discard_spurious_eigenvalues(self, problem):
    
        """
        Solves the linear eigenvalue problem for two different resolutions.
        Returns trustworthy eigenvalues using nearest delta, from Boyd chapter 7.
        """

        # Solve the linear eigenvalue problem at two different resolutions.
        LEV1 = self.solve_LEV(problem)
        LEV2 = self.solve_LEV_secondgrid(problem)
    
        # Eigenvalues returned by dedalus must be multiplied by -1
        lambda1 = -LEV1.eigenvalues
        lambda2 = -LEV2.eigenvalues

        # Reverse engineer correct indices to make unsorted list from sorted
        reverse_lambda1_indx = np.arange(len(lambda1)) 
        reverse_lambda2_indx = np.arange(len(lambda2))
    
        lambda1_and_indx = np.asarray(list(zip(lambda1, reverse_lambda1_indx)))
        lambda2_and_indx = np.asarray(list(zip(lambda2, reverse_lambda2_indx)))
        
        #print(lambda1_and_indx, lambda1_and_indx.shape, lambda1, len(lambda1))

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
        
        return lambda1, indx, LEV1

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
        
class OrderE(MRI):

    """
    Solves the order(epsilon) equation L V_1 = 0
    This is simply the linearized wide-gap MRI.
    Returns V_1
    """

    def __init__(self, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        
        print("initializing Order epsilon")
        
        MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)

        # Add in r terms as nonconstant coefficients
        lv1 = ParsedProblem(['r'], 
                field_names=['psi','u', 'A', 'B', 'psir', 'psirr', 'psirrr', 'ur', 'Ar', 'Br'],
                param_names=['Q', 'iR', 'iRm', 'q', 'beta', 'rvar', 'rvarsq'])
        
        r = domain.grid(0)
                
        # equations defined in wide_gap_eqns.ipynb
        # -1j*((D*dt)*V).subs(dz, 1j*Q) - (L*V).subs(dz, 1j*Q)
        # Note dt terms multiplied by -1j as necessitated by Dedalus hard-coded temporal eigenvalue definition
        #lv1.add_equation("1j*(1/r)*Q**2*dt(psi) + -1j*(1/r)*dt(psirr) + 1j*(1/r**2)*dt(psir) - 3*1j*Q*r**(-q)*u - iR*(Q**4/r)*psi + (2/beta)*1j*Q**3*(1/r)*A + 2*Q**2*iR*(1/r)*psirr - iR*2*Q**2*(1/r**2)*psir - (2/beta)*1j*Q*(1/r)*dr(Ar) + (2/beta)*1j*Q*(1/r**2)*Ar - iR*(1/r)*dr(psirrr) + iR*(2/r**2)*psirrr - iR*(3/r**3)*psirr + iR*(3/r**4)*psir = 0")
        #lv1.add_equation("-1j*dt(u) - (1j/r)*Q*q*r**(-q)*psi + 1j*(4/r)*Q*r**(-q)*psi + iR*Q**2*u - (2/beta)*1j*Q*B - iR*dr(ur) - iR*(1/r)*ur + iR*(1/r)*u = 0")
        #lv1.add_equation("-1j*dt(A) + iRm*Q**2*A - 1j*Q*psi - iRm*dr(Ar) + iRm*(1/r)*Ar = 0")
        #lv1.add_equation("-1j*dt(B) + 1j*(1/r)*Q*q*r**(-q)*A - 2*1j*(1/r)*Q*r**(-q)*A + iRm*Q**2*B - 1j*Q*u - iRm*dr(Br) - iRm*(1/r)*Br + iRm*(1/r**2)*B = 0")

        # Multiply through by r**2
        lv1.add_equation("1j*rvar*Q**2*dt(psi) + -1j*rvar*dt(psirr) + 1j*dt(psir) - 3*1j*Q*rvarsq*rvar**(-q)*u - iR*(Q**4)*rvar*psi + (2/beta)*1j*Q**3*rvar*A + 2*Q**2*iR*rvar*psirr - iR*2*Q**2*psir - (2/beta)*1j*Q*rvar*dr(Ar) + (2/beta)*1j*Q*Ar - iR*rvar*dr(psirrr) + iR*2*psirrr - iR*(3/rvar)*psirr + iR*(3/rvarsq)*psir = 0")
        lv1.add_equation("-1j*rvarsq*dt(u) - 1j*rvar*Q*q*rvar**(-q)*psi + 1j*4*rvar*Q*rvar**(-q)*psi + iR*rvarsq*Q**2*u - (2/beta)*1j*Q*rvarsq*B - iR*rvarsq*dr(ur) - iR*rvar*ur + iR*rvar*u = 0")
        lv1.add_equation("-1j*rvarsq*dt(A) + iRm*rvarsq*Q**2*A - 1j*rvarsq*Q*psi - iRm*rvarsq*dr(Ar) + iRm*rvar*Ar = 0")
        lv1.add_equation("-1j*rvarsq*dt(B) + 1j*rvar*Q*q*rvar**(-q)*A - 2*1j*rvar*Q*rvar**(-q)*A + iRm*rvarsq*Q**2*B - 1j*rvarsq*Q*u - iRm*rvarsq*dr(Br) - iRm*rvar*Br + iRm*B = 0")


        lv1.add_equation("dr(psi) - psir = 0")
        lv1.add_equation("dr(psir) - psirr = 0")
        lv1.add_equation("dr(psirr) - psirrr = 0")
        lv1.add_equation("dr(u) - ur = 0")
        lv1.add_equation("dr(A) - Ar = 0")
        lv1.add_equation("dr(B) - Br = 0")

        lv1.parameters['Q'] = self.Q
        lv1.parameters['iR'] = self.iR
        lv1.parameters['iRm'] = self.iRm
        lv1.parameters['q'] = self.q
        lv1.parameters['beta'] = self.beta
        lv1.parameters['rvar'] = r
        lv1.parameters['rvarsq'] = r**2
    
        self.lv1 = self.set_boundary_conditions(lv1)
        

    
        # Discard spurious eigenvalues
        self.goodevals, self.goodevals_indx, self.LEV = self.discard_spurious_eigenvalues(lv1)
       
        # Find the largest eigenvalue (fastest growing mode).
        largest_eval_indx = self.get_largest_real_eigenvalue_index(self.LEV, goodevals = self.goodevals, goodevals_indx = self.goodevals_indx)
        
        self.LEV.set_state(largest_eval_indx)
        
        # All eigenfunctions must be scaled s.t. their max is 1
        self.psi = self.LEV.state['psi']
        self.u = self.LEV.state['u']
        self.A = self.LEV.state['A']
        self.B = self.LEV.state['B']
    

        