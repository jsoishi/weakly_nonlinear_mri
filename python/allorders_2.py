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

gridnum = 64#128
print("running at gridnum", gridnum)
x_basis = Chebyshev(gridnum)
domain = Domain([x_basis], grid_dtype=np.complex128)

# Second basis for checking eigenvalues
x_basis192 = Chebyshev(96)
#x_basis192 = Chebyshev(64)
#x_basis192 = Chebyshev(192)
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
        
        #delta_near_unsorted = delta_near[reverse_lambda1_indx]
        #lambda1[np.where((1.0/delta_near_unsorted) < 1E6)] = None
        #lambda1[np.where(np.isnan(1.0/delta_near_unsorted) == True)] = None
    
        return lambda1, indx, LEV1
        
    def discard_spurious_eigenvalues2(self, problem):
    
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
    
        # Sorted indices for lambda1 and lambda2 by real parts
        lambda1_indx = np.argsort(lambda1.real)
        lambda2_indx = np.argsort(lambda2.real)
        
        # Reverse engineer correct indices to make unsorted list from sorted
        reverse_lambda1_indx = sorted(range(len(lambda1_indx)), key=lambda1_indx.__getitem__)
        reverse_lambda2_indx = sorted(range(len(lambda2_indx)), key=lambda2_indx.__getitem__)
        
        self.lambda1_indx = lambda1_indx
        self.reverse_lambda1_indx = reverse_lambda1_indx
        self.lambda1 = lambda1
        
        # remove nans
        lambda1_indx = lambda1_indx[np.isfinite(lambda1)]
        reverse_lambda1_indx = np.asarray(reverse_lambda1_indx)
        reverse_lambda1_indx = reverse_lambda1_indx[np.isfinite(lambda1) == True]
        #lambda1 = lambda1[np.isfinite(lambda1)]
        
        lambda2_indx = lambda2_indx[np.isfinite(lambda2)]
        reverse_lambda2_indx = np.asarray(reverse_lambda2_indx)
        reverse_lambda2_indx = reverse_lambda2_indx[np.isfinite(lambda2)]
        #lambda2 = lambda2[np.isfinite(lambda2)]
        
        # Actually sort the eigenvalues by their real parts
        lambda1_sorted = lambda1[lambda1_indx]
        lambda2_sorted = lambda2[lambda2_indx]
        
        self.lambda1_sorted = lambda1_sorted
        #print(lambda1_sorted)
        #print(len(lambda1_sorted), len(np.where(np.isfinite(lambda1) == True)))
    
        # Compute sigmas from lower resolution run (gridnum = N1)
        sigmas = np.zeros(len(lambda1_sorted))
        sigmas[0] = np.abs(lambda1_sorted[0] - lambda1_sorted[1])
        sigmas[1:-1] = [0.5*(np.abs(lambda1_sorted[j] - lambda1_sorted[j - 1]) + np.abs(lambda1_sorted[j + 1] - lambda1_sorted[j])) for j in range(1, len(lambda1_sorted) - 1)]
        sigmas[-1] = np.abs(lambda1_sorted[-2] - lambda1_sorted[-1])

        if not (np.isfinite(sigmas)).all():
            print("WARNING: at least one eigenvalue spacings (sigmas) is non-finite (np.inf or np.nan)!")
    
        # Nearest delta
        delta_near = np.array([np.nanmin(np.abs(lambda1_sorted[j] - lambda2_sorted)/sigmas[j]) for j in range(len(lambda1_sorted))])
    
        #print(len(delta_near), len(reverse_lambda1_indx), len(LEV1.eigenvalues))
        # Discard eigenvalues with 1/delta_near < 10^6
        delta_near_unsorted = np.zeros(len(LEV1.eigenvalues))
        for i in range(len(delta_near)):
            delta_near_unsorted[reverse_lambda1_indx[i]] = delta_near[i]
        #delta_near_unsorted[reverse_lambda1_indx] = delta_near#[reverse_lambda1_indx]
        #print(delta_near_unsorted)
        
        self.delta_near_unsorted = delta_near_unsorted
        self.delta_near = delta_near
        
        goodeigs = copy.copy(LEV1.eigenvalues)
        goodeigs[np.where((1.0/delta_near_unsorted) < 1E6)] = None
        goodeigs[np.where(np.isfinite(1.0/delta_near_unsorted) == False)] = None
    
        return goodeigs, LEV1
        
    def find_spurious_eigenvalues(self, problem):
    
        """
        Solves the linear eigenvalue problem for two different resolutions.
        Returns drift ratios, from Boyd chapter 7.
        """
    
        # Solve the linear eigenvalue problem at two different resolutions.
        LEV1 = self.solve_LEV(problem)
        LEV2 = self.solve_LEV_secondgrid(problem)
        
        lambda1 = LEV1.eigenvalues
        lambda2 = LEV2.eigenvalues
        
        # Make sure argsort treats complex infs correctly
        for i in range(len(lambda1)):
            if (np.isnan(lambda1[i]) == True) or (np.isinf(lambda1[i]) == True):
                lambda1[i] = None
        for i in range(len(lambda2)):
            if (np.isnan(lambda2[i]) == True) or (np.isinf(lambda2[i]) == True):
                lambda2[i] = None        
        
        #lambda1[np.where(np.isnan(lambda1) == True)] = None
        #lambda2[np.where(np.isnan(lambda2) == True)] = None
                
        # Sort lambda1 and lambda2 by real parts
        lambda1_indx = np.argsort(lambda1.real)
        lambda1 = lambda1[lambda1_indx]
        lambda2_indx = np.argsort(lambda2.real)
        lambda2 = lambda2[lambda2_indx]
        
        # try using lower res (gridnum = N1) instead
        sigmas = np.zeros(len(lambda1))
        sigmas[0] = np.abs(lambda1[0] - lambda1[1])
        sigmas[1:-1] = [0.5*(np.abs(lambda1[j] - lambda1[j - 1]) + np.abs(lambda1[j + 1] - lambda1[j])) for j in range(1, len(lambda1) - 1)]
        sigmas[-1] = np.abs(lambda1[-2] - lambda1[-1])
        
        # Ordinal delta, calculated for the number of lambda1's.
        delta_ord = (lambda1 - lambda2[:len(lambda1)])/sigmas
        
        # Nearest delta
        delta_near = [np.nanmin(np.abs(lambda1[j] - lambda2)) for j in range(len(lambda1))]/sigmas
        
        # Discard eigenvalues with 1/delta_near < 10^6
        goodevals1 = lambda1[1/delta_near > 1E6]
        
        return delta_ord, delta_near, lambda1, lambda2, sigmas
        
    def get_largest_eigenvalue_index(self, LEV, goodevals = None):
        
        """
        Return index of largest eigenvalue. Can be positive or negative.
        """
        if goodevals == None:
            evals = LEV.eigenvalues
        else:
            evals = goodevals
            
        indx = np.arange(len(evals))
        largest_eval_indx = indx[evals == np.nanmax(evals)]
        
        return largest_eval_indx
        
    def get_largest_real_eigenvalue_index(self, LEV, goodevals = None, goodevals_indx = None):
        
        """
        Return index of largest eigenvalue. Can be positive or negative.
        """
        
        if goodevals == None:
            evals = LEV.eigenvalues
        else:
            evals = goodevals
        
        #goodevals_and_indx = zip(goodevals, goodevals_indx)
        #largest_eval_pseudo_indx = np.nanargmax(goodevals_and_indx[:, 0].real)
        #largest_eval_indx = goodevals_and_indx[largest_eval_pseudo_indx, 1]
        largest_eval_pseudo_indx = np.nanargmax(goodevals.real)
        largest_eval_indx = goodevals_indx[largest_eval_pseudo_indx]
        
        print("largest eigenvalue indx", largest_eval_indx)
        
        return largest_eval_indx
        
    def get_largest_real_eigenvalue_index2(self, LEV, goodevals = None):
        
        """
        Return index of largest eigenvalue. Can be positive or negative.
        """
        if goodevals == None:
            evals = LEV.eigenvalues
        else:
            evals = goodevals
            
        #indx = np.arange(len(evals))
        #largest_eval_indx = indx[evals.real == np.nanmax(evals.real)]
        largest_eval_indx = np.nanargmax(evals.real)
        
        #print(largest_eval_indx)
        
        return largest_eval_indx[0]
        
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
        
        print("begin norm")
        veclen = len(psi['g'])
        evector = np.zeros(veclen*4, np.complex_)
        evector[0:veclen] = psi['g'] #+ psi['g'].conj()
        evector[veclen:veclen*2] = u['g'] #+ u['g'].conj()
        evector[veclen*2:veclen*3] = A['g'] #+ A['g'].conj()
        evector[veclen*3:] = B['g'] #+ B['g'].conj()
        
        norm = np.linalg.norm(evector)
        print("end norm")
        
        psi['g'] = psi['g']/norm
        u['g'] = u['g']/norm
        A['g'] = A['g']/norm
        B['g'] = B['g']/norm
        
        return psi, u, A, B
        
    def normalize_state_vector2(self, psi0, u0, A0, B0, psi1, u1, A1, B1, psi2, u2, A2, B2):
        
        """
        Normalize total state vector.
        """
        
        veclen = len(psi0['g'])
        evector = np.zeros(veclen*12, np.complex_)
        evector[0:veclen] = psi0['g']
        evector[veclen:veclen*2] = u0['g']
        evector[veclen*2:veclen*3] = A0['g']
        evector[veclen*3:veclen*4] = B0['g']
        
        evector[veclen*4:veclen*5] = psi1['g']
        evector[veclen*5:veclen*6] = u1['g']
        evector[veclen*6:veclen*7] = A1['g']
        evector[veclen*7:veclen*8] = B1['g']
        
        evector[veclen*8:veclen*9] = psi2['g']
        evector[veclen*9:veclen*10] = u2['g']
        evector[veclen*10:veclen*11] = A2['g']
        evector[veclen*11:] = B2['g']
        
        norm = np.linalg.norm(evector)
        #print(evector)
        print(norm)
        
        psi0['g'] = psi0['g']/norm
        u0['g'] = u0['g']/norm
        A0['g'] = A0['g']/norm
        B0['g'] = B0['g']/norm
        
        psi1['g'] = psi1['g']/norm
        u1['g'] = u1['g']/norm
        A1['g'] = A1['g']/norm
        B1['g'] = B1['g']/norm
        
        psi2['g'] = psi2['g']/norm
        u2['g'] = u2['g']/norm
        A2['g'] = A2['g']/norm
        B2['g'] = B2['g']/norm
        
        return psi0, u0, A0, B0, psi1, u1, A1, B1, psi2, u2, A2, B2
        
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
        Take inner product < vector2 | vector1 > 
        Defined as integral of (vector2.conj * vector1)
        """
        print(vector1)
        
        inner_product = vector1[0]['g']*vector2[0]['g'].conj() + vector1[1]['g']*vector2[1]['g'].conj() + vector1[2]['g']*vector2[2]['g'].conj() + vector1[3]['g']*vector2[3]['g'].conj()
        print(inner_product)
        
        ip = domain.new_field()
        ip.name = "inner product"
        ip['g'] = inner_product
        ip = ip.integrate(x_basis)
        ip = ip['g'][0]/2.0
        
        return ip
        
    def take_inner_product2(self, vector1, vector2):
        
        """
        Take inner product < vector2 | vector1 > 
        Defined as integral of (vector2.conj * vector1)
        """

        inner_product = vector1[0]['g']*vector2[0]['g'] + vector1[1]['g']*vector2[1]['g'] + vector1[2]['g']*vector2[2]['g'] + vector1[3]['g']*vector2[3]['g']
        
        
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

    def __init__(self, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        
        print("initializing Adjoint Homogenous")
        
        MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
      
        # Set up problem object
        lv1 = ParsedProblem(['x'],
                field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                param_names=['Q', 'iR', 'iRm', 'q', 'beta'])

        #switched q-2 to 2-q
        #lv1.add_equation("1j*Q**2*dt(psi) - 1j*dt(psixx) + 1j*Q*A + 1j*(q - 2)*Q*u - iR*Q**4*psi + 2*iR*Q**2*psixx - iR*dx(psixxx) = 0")
        #lv1.add_equation("-1j*dt(u) + 1j*Q*B + 2*1j*Q*psi + iR*Q**2*u - iR*dx(ux) = 0")
        #lv1.add_equation("-1j*dt(A) + iRm*Q**2*A - iRm*dx(Ax) - 1j*Q*q*B - 1j*(2/beta)*Q**3*psi + 1j*(2/beta)*Q*psixx = 0")
        #lv1.add_equation("-1j*dt(B) + iRm*Q**2*B - iRm*dx(Bx) + 1j*(2/beta)*Q*u = 0")
        
        #psi and A are upside down
        #lv1.add_equation("1j*Q**2*dt(psi) - 1j*dt(psixx) - 1j*Q*A - 1j*(q - 2)*Q*u - iR*Q**4*psi + 2*iR*Q**2*psixx - iR*dx(psixxx) = 0")
        #lv1.add_equation("-1j*dt(u) - 1j*Q*B - 2*1j*Q*psi + iR*Q**2*u - iR*dx(ux) = 0")
        #lv1.add_equation("-1j*dt(A) + iRm*Q**2*A - iRm*dx(Ax) + 1j*Q*q*B + 1j*(2/beta)*Q**3*psi - 1j*(2/beta)*Q*psixx = 0")
        #lv1.add_equation("-1j*dt(B) + iRm*Q**2*B - iRm*dx(Bx) - 1j*(2/beta)*Q*u = 0")
        
        #wait - adjoint should be taken as (L.subs(dz, 1j*Q)).adjoint() -- the following are correct.
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
        
        # Discard spurious eigenvalues
        self.goodevals, self.goodevals_indx, self.LEV = self.discard_spurious_eigenvalues(lv1)
       
        # Find the largest eigenvalue (fastest growing mode).
        largest_eval_indx = self.get_largest_real_eigenvalue_index(self.LEV, goodevals = self.goodevals, goodevals_indx = self.goodevals_indx)
        
        self.LEV.set_state(largest_eval_indx)
        
        self.psi = self.LEV.state['psi']
        self.u = self.LEV.state['u']
        self.A = self.LEV.state['A']
        self.B = self.LEV.state['B']
               
        #self.psi['g'] = self.normalize_vector(self.psi['g'])
        #self.u['g'] = self.normalize_vector(self.u['g'])
        #self.A['g'] = self.normalize_vector(self.A['g'])
        #self.B['g'] = self.normalize_vector(self.B['g'])
        
        if self.norm == True:
            scale = self.normalize_all_real_or_imag(self.LEV)
            
            self.psi = (self.psi*scale).evaluate()
            self.u = (self.u*scale).evaluate()
            self.A = (self.A*scale).evaluate()
            self.B = (self.B*scale).evaluate()
            
        self.psi, self.u, self.A, self.B = self.normalize_state_vector(self.psi, self.u, self.A, self.B)
        
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
        
        # I guess we should be normalizing derivatives too?
        """
        self.psi_x['g'] = self.normalize_vector(self.psi_x['g'])
        self.psi_xx['g'] = self.normalize_vector(self.psi_xx['g'])
        self.psi_xxx['g'] = self.normalize_vector(self.psi_xxx['g'])
        
        self.u_x['g'] = self.normalize_vector(self.u_x['g'])
        
        self.A_x['g'] = self.normalize_vector(self.A_x['g'])
        self.A_xx['g'] = self.normalize_vector(self.A_xx['g'])
        self.A_xxx['g'] = self.normalize_vector(self.A_xxx['g'])
        
        self.B_x['g'] = self.normalize_vector(self.B_x['g'])
        """
            
class OrderE(MRI):

    """
    Solves the order(epsilon) equation L V_1 = 0
    This is simply the linearized MRI.
    Returns V_1
    """

    def __init__(self, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        
        print("initializing Order E")
        
        MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)

        lv1 = ParsedProblem(['x'],
                field_names=['psi','u', 'A', 'B', 'psix', 'psixx', 'psixxx', 'ux', 'Ax', 'Bx'],
                param_names=['Q', 'iR', 'iRm', 'q', 'beta'])

        lv1.add_equation("-1j*dt(psixx) + 1j*Q**2*dt(psi) - iR*dx(psixxx) + 2*iR*Q**2*psixx - iR*Q**4*psi - 2*1j*Q*u - (2/beta)*1j*Q*dx(Ax) + (2/beta)*Q**3*1j*A = 0")
        lv1.add_equation("-1j*dt(u) - iR*dx(ux) + iR*Q**2*u - (q - 2)*1j*Q*psi - (2/beta)*1j*Q*B = 0") 
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
       
        # Discard spurious eigenvalues
        self.goodevals, self.goodevals_indx, self.LEV = self.discard_spurious_eigenvalues(lv1)
       
        # Find the largest eigenvalue (fastest growing mode).
        largest_eval_indx = self.get_largest_real_eigenvalue_index(self.LEV, goodevals = self.goodevals, goodevals_indx = self.goodevals_indx)
        
        #print(largest_eval_indx)
        #print(largest_eval_indx.shape)
        self.LEV.set_state(largest_eval_indx)
        
        # All eigenfunctions must be scaled s.t. their max is 1
        self.psi = self.LEV.state['psi']
        self.u = self.LEV.state['u']
        self.A = self.LEV.state['A']
        self.B = self.LEV.state['B']
        
        # This does nothing
        #self.psi, self.u, self.A, self.B = self.normalize_state_vector(self.psi, self.u, self.A, self.B)
        
        #self.psi['g'] = self.normalize_vector(self.psi['g'])
        #self.u['g'] = self.normalize_vector(self.u['g'])
        #self.A['g'] = self.normalize_vector(self.A['g'])
        #self.B['g'] = self.normalize_vector(self.B['g'])
        print("pre norm", self.psi['g'])
        self.prenormpsi = self.psi
        
        if self.norm == True:
            scale = self.normalize_all_real_or_imag(self.LEV)
            
            self.psi = (self.psi*scale).evaluate()
            print(self.psi['g'])
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
        
        # Normalize derivatives and cc's as well... 
        """
        self.psi_x['g'] = self.normalize_vector(self.psi_x['g'])
        self.psi_xx['g'] = self.normalize_vector(self.psi_xx['g'])
        self.psi_xxx['g'] = self.normalize_vector(self.psi_xxx['g'])
        
        self.u_x['g'] = self.normalize_vector(self.u_x['g'])
        
        self.A_x['g'] = self.normalize_vector(self.A_x['g'])
        self.A_xx['g'] = self.normalize_vector(self.A_xx['g'])
        self.A_xxx['g'] = self.normalize_vector(self.A_xxx['g'])
        
        self.B_x['g'] = self.normalize_vector(self.B_x['g'])
        
        self.psi_star['g'] = self.normalize_vector(self.psi_star['g'])
        self.psi_star_x['g'] = self.normalize_vector(self.psi_star_x['g'])
        self.psi_star_xx['g'] = self.normalize_vector(self.psi_star_xx['g'])
        self.psi_star_xxx['g'] = self.normalize_vector(self.psi_star_xxx['g'])
        
        self.u_star['g'] = self.normalize_vector(self.u_star['g'])
        self.u_star_x['g'] = self.normalize_vector(self.u_star_x['g'])
        
        self.A_star['g'] = self.normalize_vector(self.A_star['g'])
        self.A_star_x['g'] = self.normalize_vector(self.A_star_x['g'])
        self.A_star_xx['g'] = self.normalize_vector(self.A_star_xx['g'])
        self.A_star_xxx['g'] = self.normalize_vector(self.A_star_xxx['g'])
        
        self.B_star['g'] = self.normalize_vector(self.B_star['g'])
        self.B_star_x['g'] = self.normalize_vector(self.B_star_x['g'])
        """
        
class N2(MRI):

    """
    Solves the nonlinear term N2
    Returns N2
    
    """
    
    def __init__(self, o1 = None, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
    
        print("initializing N2")
    
        if o1 == None:
            o1 = OrderE(Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
            MRI.__init__(self, Q = Q, Rm = Rm, Pm = Pm, q = q, beta = beta, norm = norm)
        else:
            MRI.__init__(self, Q = o1.Q, Rm = o1.Rm, Pm = o1.Pm, q = o1.q, beta = o1.beta, norm = o1.norm)
    
        N22psi = 1j*self.Q*o1.psi*(o1.psi_xxx - self.Q**2*o1.psi_x) - o1.psi_x*(1j*self.Q*o1.psi_xx - 1j*self.Q**3*o1.psi) + (2/self.beta)*o1.A_x*(1j*self.Q*o1.A_xx - 1j*self.Q**3*o1.A) - (2/self.beta)*1j*self.Q*o1.A*(o1.A_xxx - self.Q**2*o1.A_x) # Confirmed 11/2/15
        self.N22_psi = N22psi.evaluate()
        self.N22_psi.name = "N22_psi"
    
        N20psi = 1j*self.Q*o1.psi*(o1.psi_star_xxx - self.Q**2*o1.psi_star_x) - o1.psi_x*(-1j*self.Q*o1.psi_star_xx + 1j*self.Q**3*o1.psi_star) + (2/self.beta)*o1.A_x*(-1j*self.Q*o1.A_star_xx + 1j*self.Q**3*o1.A_star) - (2/self.beta)*1j*self.Q*o1.A*(o1.A_star_xxx - self.Q**2*o1.A_star_x) # Confirmed 11/2/15
        #N20psi = 1j*self.Q*o1.psi*(o1.psi_star_xxx - self.Q**2*o1.psi_star_x) - o1.psi_x*(1j*self.Q*o1.psi_star_xx - 1j*self.Q**3*o1.psi_star) + (2/self.beta)*o1.A_x*(1j*self.Q*o1.A_star_xx - 1j*self.Q**3*o1.A_star) - (2/self.beta)*1j*self.Q*o1.A*(o1.A_star_xxx - self.Q**2*o1.A_star_x)
        self.N20_psi = N20psi.evaluate()
        self.N20_psi.name = "N20_psi"
        
        N22u = 1j*self.Q*o1.psi*o1.u_x - o1.psi_x*1j*self.Q*o1.u - (2/self.beta)*1j*self.Q*o1.A*o1.B_x + (2/self.beta)*o1.A_x*1j*self.Q*o1.B # confirmed 10/30/15
        self.N22_u = N22u.evaluate()
        self.N22_u.name = "N22_u"
        
        N20u = 1j*self.Q*o1.psi*o1.u_star_x + o1.psi_x*1j*self.Q*o1.u_star - (2/self.beta)*1j*self.Q*o1.A*o1.B_star_x - (2/self.beta)*o1.A_x*1j*self.Q*o1.B_star # confirmed 10/30/15
        #N20u = 1j*self.Q*o1.psi*o1.u_star_x - o1.psi_x*1j*self.Q*o1.u_star - (2/self.beta)*1j*self.Q*o1.A*o1.B_star_x + (2/self.beta)*o1.A_x*1j*self.Q*o1.B_star
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
        #N20B = 1j*self.Q*o1.psi*o1.B_star_x - o1.psi_x*1j*self.Q*o1.B_star - 1j*self.Q*o1.A*o1.u_star_x + o1.A_x*1j*self.Q*o1.u_star
        self.N20_B = N20B.evaluate()
        self.N20_B.name = "N20_B"
        
        # Let's try normalizing N2 eigenvectors.
        """
        self.N22_psi['g'] = self.normalize_vector(self.N22_psi['g'])
        self.N22_u['g'] = self.normalize_vector(self.N22_u['g'])
        self.N22_A['g'] = self.normalize_vector(self.N22_A['g'])
        self.N22_B['g'] = self.normalize_vector(self.N22_B['g'])
        
        self.N20_psi['g'] = self.normalize_vector(self.N20_psi['g'])
        self.N20_u['g'] = self.normalize_vector(self.N20_u['g'])
        self.N20_A['g'] = self.normalize_vector(self.N20_A['g'])
        self.N20_B['g'] = self.normalize_vector(self.N20_B['g'])
        """
       
        
class OrderE2(MRI):

    """
    Solves the second order equation L V2 = N2 - Ltwiddle V1
    Returns V2
    
    """
    
    def __init__(self, o1 = None, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
    
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
        #rhs20_psi = n2.N20_psi['g']
        #rhs20_u = n2.N20_u['g'] - (2.0/beta)
        #rhs20_A = n2.N20_A['g'] 
        #rhs20_B = n2.N20_B['g'] - 2*1j*Q*self.iRm
        
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
        # second term: -L1twiddle V1
        term2_psi = 3*(2/self.beta)*self.Q**2*o1.A - (2/self.beta)*o1.A_xx + 4*self.iR*1j*self.Q**3*o1.psi - 4*self.iR*1j*self.Q*o1.psi_xx - 2*o1.u
        self.term2_psi = term2_psi.evaluate()
        
        #term2_u = (2/self.beta)*o1.B + 2*self.iR*self.Q*o1.u + (self.q - 2)*o1.psi ## why does (q - 2) and (2 - q) make no diff here??
        term2_u = -(2/self.beta)*o1.B - 2*self.iR*1j*self.Q*o1.u - (self.q - 2)*o1.psi #added missing 1j in second term 10/14/15
        self.term2_u = term2_u.evaluate()
        
        term2_A = -2*self.iRm*1j*self.Q*o1.A - o1.psi
        self.term2_A = term2_A.evaluate()
        
        term2_B = self.q*o1.A - 2*self.iRm*1j*self.Q*o1.B - o1.u
        self.term2_B = term2_B.evaluate()
        
        # righthand side for the 21 terms (e^iQz dependence)
        rhs21_psi = self.term2_psi['g']
        rhs21_u = self.term2_u['g']
        rhs21_A = self.term2_A['g']
        rhs21_B = self.term2_B['g']
                
        # define problem using righthand side as nonconstant coefficients
        
        bv21 = ParsedProblem(['x'],
              field_names=['psi21', 'psi21x', 'psi21xx', 'psi21xxx', 'u21', 'u21x', 'A21', 'A21x', 'B21', 'B21x'],
              param_names=['Q', 'iR', 'iRm', 'q', 'beta', 'rhs21_psi', 'rhs21_u', 'rhs21_A', 'rhs21_B'])
          
        #bv21.add_equation("1j*(2/beta)*Q**3*A21 - 1j*(2/beta)*Q*dx(A21x) - 2*1j*Q*u21 - iR*Q**4*psi21 + 2*iR*Q**2*psi21xx - iR*dx(psi21xxx) = rhs21_psi")
        #bv21.add_equation("-1j*(2/beta)*Q*B21 - 1j*Q*(q - 2)*psi21 + iR*Q**2*u21 - iR*dx(u21x) = rhs21_u")
        #bv21.add_equation("iRm*Q**2*A21 - iRm*dx(A21x) - 1j*Q*psi21 = rhs21_A")
        #bv21.add_equation("1j*Q*q*A21 + iRm*Q**2*B21 - iRm*dx(B21x) - 1j*Q*u21 = rhs21_B")   
        
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
        
        #wild alternative version
        #self.psi21['g'] = o1.psi['g']
        #self.u21['g'] = o1.u['g']
        #self.A21['g'] = o1.A['g']
        #self.B21['g'] = o1.B['g']
        
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
        
        
        if self.norm == True:
            print("not normalize_all_real_or_imag - ing V21 or V22")
            """
            scale21 = self.normalize_all_real_or_imag_bystate(self.psi21)
            scale22 = self.normalize_all_real_or_imag_bystate(self.psi22)
    
            self.psi21 = (self.psi21*scale21).evaluate()
            self.u21 = (self.u21*scale21).evaluate()
            self.A21 = (self.A21*scale21).evaluate()
            self.B21 = (self.B21*scale21).evaluate()
            
            self.psi22 = (self.psi22*scale22).evaluate()
            self.u22 = (self.u22*scale22).evaluate()
            self.A22 = (self.A22*scale22).evaluate()
            self.B22 = (self.B22*scale22).evaluate()
            """
            
        #self.psi21, self.u21, self.A21, self.B21 = self.normalize_state_vector(self.psi21, self.u21, self.A21, self.B21)
        #self.psi22, self.u22, self.A22, self.B22 = self.normalize_state_vector(self.psi22, self.u22, self.A22, self.B22)
            
        # These should be zero... 
        self.psi20['g'] = np.zeros(gridnum, np.int_)
        self.B20['g'] = np.zeros(gridnum, np.int_)
        
        if self.norm == True:
            print("not normalize_all_real_or_imag - ing u20 and A20")
            #scale20 = self.normalize_all_real_or_imag_bystate(self.u20)
            #self.u20 = (self.u20*scale20).evaluate()
            #self.A20 = (self.A20*scale20).evaluate()
        
        #self.psi20, self.u20, self.A20, self.B20 = self.normalize_state_vector(self.psi20, self.u20, self.A20, self.B20)
        #self.psi20, self.u20, self.A20, self.B20, self.psi21, self.u21, self.A21, self.B21, self.psi22, self.u22, self.A22, self.B22 = self.normalize_state_vector2(self.psi20, self.u20, self.A20, self.B20, self.psi21, self.u21, self.A21, self.B21, self.psi22, self.u22, self.A22, self.B22)

        #self.u20['g'] = self.normalize_vector(self.u20['g'])
        #self.A20['g'] = self.normalize_vector(self.A20['g'])
        
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
    
    def __init__(self, o1 = None, o2 = None, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        
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
        
        #N31_psi_my1 = 1j*self.Q*(o1.psi*o2.psi20_xxx) - 1j*self.Q*(o1.psi_star*o2.psi22_xxx) - 1j*2*self.Q*(o1.psi_star_x*o2.psi22_xx) + 1j*8*self.Q**3*(o1.psi_star_x*o2.psi22) + 1j*4*self.Q**3*(o1.psi_star*o2.psi22_x)
        #N31_psi_my2 = -1j*self.Q*(2/self.beta)*(o1.A*o2.A20_xxx) + 1j*self.Q*(2/self.beta)*(o1.A_star*o2.A22_xxx) + 1j*2*self.Q*(2/self.beta)*(o1.A_star_x*o2.A22_xx) - 1j*8*self.Q**3*(2/self.beta)*(o1.A_star_x*o2.A22) - 1j*4*self.Q**3*(2/self.beta)*(o1.A_star*o2.A22_x)
        #N31_psi_my3 = 1j*2*self.Q*(o2.psi22*o1.psi_star_xxx) - 1j*2*self.Q**3*(o2.psi22*o1.psi_star_x) - 1j*self.Q*(o2.psi20_x*o1.psi_xx) + 1j*self.Q*(o2.psi22_x*o1.psi_star_xx) + 1j*self.Q**3*(o2.psi20_x*o1.psi) - 1j*self.Q**3*(o2.psi22_x*o1.psi_star)
        #N31_psi_my4 = -1j*2*self.Q*(2/self.beta)*(o2.A22*o1.A_star_xxx) + 1j*2*self.Q**3*(2/self.beta)*(o2.A22*o1.A_star_x) + 1j*self.Q*(2/self.beta)*(o2.A20_x*o1.A_xx) - 1j*self.Q*(2/self.beta)*(o2.A22_x*o1.A_star_xx) - 1j*self.Q**3*(2/self.beta)*(o2.A20_x*o1.A) + 1j*self.Q**3*(2/self.beta)*(o2.A22_x*o1.A_star)
        
        
        N31_psi = N31_psi_my1 + N31_psi_my2 + N31_psi_my3 +  N31_psi_my4
        
        self.N31_psi = N31_psi.evaluate()
        
        # diagnostics reveal that my_2 and my_4 are nan with the new normalizations. A_star? A22?
        #print(self.N31_psi['g'])
        p1 = N31_psi_my1.evaluate()
        p2 = N31_psi_my2.evaluate()
        p3 = N31_psi_my3.evaluate()
        p4 = N31_psi_my4.evaluate()
        #print("N31_psi_my1", p1['g'])
        #print("N31_psi_my2", p2['g'])
        #print("N31_psi_my3", p3['g'])
        #print("N31_psi_my4", p4['g'])
        
        # u component
        N31_u_my1 = 1j*self.Q*(o1.psi*o2.u20_x) + 1j*self.Q*(o1.psi*o2.u20_star_x) - 1j*self.Q*(o1.psi_star*o2.u22_x) - 1j*2*self.Q*(o1.psi_star_x*o2.u22)
        N31_u_my2 = -1j*self.Q*(o1.u*o2.psi20_x) - 1j*self.Q*(o1.u*o2.psi20_star_x) + 1j*self.Q*(o1.u_star*o2.psi22_x) + 1j*2*self.Q*(o1.u_star_x*o2.psi22)
        N31_u_my3 = -1j*self.Q*(2/self.beta)*(o1.A*o2.B20_x) - 1j*self.Q*(2/self.beta)*(o1.A*o2.B20_star_x) + 1j*self.Q*(2/self.beta)*(o1.A_star*o2.B22_x) + 1j*2*self.Q*(2/self.beta)*(o1.A_star_x*o2.B22)
        N31_u_my4 = 1j*self.Q*(2/self.beta)*(o1.B*o2.A20_x) + 1j*self.Q*(2/self.beta)*(o1.B*o2.A20_star_x) - 1j*self.Q*(2/self.beta)*(o1.B_star*o2.A20_x) - 1j*2*self.Q*(2/self.beta)*(o1.B_star_x*o2.A22)
        
        #N31_u_my1 = 1j*self.Q*(o1.psi*o2.u20_x) - 1j*self.Q*(o1.psi_star*o2.u22_x) - 1j*2*self.Q*(o1.psi_star_x*o2.u22)
        #N31_u_my2 = -1j*self.Q*(o1.u*o2.psi20_x) + 1j*self.Q*(o1.u_star*o2.psi22_x) + 1j*2*self.Q*(o1.u_star_x*o2.psi22)
        #N31_u_my3 = -1j*self.Q*(2/self.beta)*(o1.A*o2.B20_x) + 1j*self.Q*(2/self.beta)*(o1.A_star*o2.B22_x) + 1j*2*self.Q*(2/self.beta)*(o1.A_star_x*o2.B22)
        #N31_u_my4 = 1j*self.Q*(2/self.beta)*(o1.B*o2.A20_x) - 1j*self.Q*(2/self.beta)*(o1.B_star*o2.A20_x) - 1j*2*self.Q*(2/self.beta)*(o1.B_star_x*o2.A22)
        
        
        N31_u = N31_u_my1 + N31_u_my2 + N31_u_my3 + N31_u_my4
        
        self.N31_u = N31_u.evaluate()
        #print(self.N31_u['g'])
        
        # A component -- correct with all-positive V2 definition. Checked 11/14/15
        N31_A_my1 = -1j*self.Q*(o1.A*o2.psi20_x) - 1j*self.Q*(o1.A*o2.psi20_star_x) + 1j*self.Q*(o1.A_star*o2.psi22_x) + 1j*2*self.Q*(o1.A_star_x*o2.psi22)
        N31_A_my2 = 1j*self.Q*(o1.psi*o2.A20_x) + 1j*self.Q*(o1.psi*o2.A20_star_x) - 1j*self.Q*(o1.psi_star*o2.A22_x) - 1j*2*self.Q*(o1.psi_star_x*o2.A22)
        
        #N31_A_my1 = -1j*self.Q*(o1.A*o2.psi20_x) + 1j*self.Q*(o1.A_star*o2.psi22_x) + 1j*2*self.Q*(o1.A_star_x*o2.psi22)
        #N31_A_my2 = 1j*self.Q*(o1.psi*o2.A20_x) - 1j*self.Q*(o1.psi_star*o2.A22_x) - 1j*2*self.Q*(o1.psi_star_x*o2.A22)
        
        N31_A = N31_A_my1 + N31_A_my2
        
        self.N31_A = N31_A.evaluate()
        #print(self.N31_A['g'])
        
        # B component -- correct with all-positive V2 definition. Checked 11/6/15
        N31_B_my1 = 1j*self.Q*(o1.psi*o2.B20_x) + 1j*self.Q*(o1.psi*o2.B20_star_x) - 1j*self.Q*(o1.psi_star*o2.B22_x) - 1j*2*self.Q*(o1.psi_star_x*o2.B22)
        N31_B_my2 = -1j*self.Q*(o1.B*o2.psi20_x) - 1j*self.Q*(o1.B*o2.psi20_star_x) + 1j*self.Q*(o1.B_star*o2.psi22_x) + 1j*2*self.Q*(o1.B_star_x*o2.psi22)
        N31_B_my3 = -1j*self.Q*(o1.A*o2.u20_x) - 1j*self.Q*(o1.A*o2.u20_star_x) + 1j*self.Q*(o1.A_star*o2.u22_x) + 1j*2*self.Q*(o1.A_star_x*o2.u22)
        N31_B_my4 = 1j*self.Q*(o1.u*o2.A20_x) + 1j*self.Q*(o1.u*o2.A20_star_x) - 1j*self.Q*(o1.u_star*o2.A22_x) - 1j*2*self.Q*(o1.u_star_x*o2.A22)
        
        # Umurhan+ has no 20_star elements. Try that?
        #N31_B_my1 = 1j*self.Q*(o1.psi*o2.B20_x) - 1j*self.Q*(o1.psi_star*o2.B22_x) - 1j*2*self.Q*(o1.psi_star_x*o2.B22)
        #N31_B_my2 = -1j*self.Q*(o1.B*o2.psi20_x) + 1j*self.Q*(o1.B_star*o2.psi22_x) + 1j*2*self.Q*(o1.B_star_x*o2.psi22)
        #N31_B_my3 = -1j*self.Q*(o1.A*o2.u20_x) + 1j*self.Q*(o1.A_star*o2.u22_x) + 1j*2*self.Q*(o1.A_star_x*o2.u22)
        #N31_B_my4 = 1j*self.Q*(o1.u*o2.A20_x)  - 1j*self.Q*(o1.u_star*o2.A22_x) - 1j*2*self.Q*(o1.u_star_x*o2.A22)
        
        
        N31_B = N31_B_my1 + N31_B_my2 + N31_B_my3 + N31_B_my4
        
        self.N31_B = N31_B.evaluate()
        #print(self.N31_B['g'])
        
        # Let's try normalizing N3 eigenvectors.
        #self.N31_psi['g'] = self.normalize_vector(self.N31_psi['g'])
        #self.N31_u['g'] = self.normalize_vector(self.N31_u['g'])
        #self.N31_A['g'] = self.normalize_vector(self.N31_A['g'])
        #self.N31_B['g'] = self.normalize_vector(self.N31_B['g'])


class AmplitudeAlpha(MRI):

    """
    Solves the coefficients of the first amplitude equation for alpha -- e^(iQz) terms.
    
    """
    
    def __init__(self, o1 = None, o2 = None, Q = 0.748, Rm = 4.879, Pm = 0.001, q = 1.5, beta = 25.0, norm = True):
        
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
        
        a_psi_rhs2 = o1.psi_star_xx - self.Q**2*o1.psi #test
        a_psi_rhs2 = a_psi_rhs2.evaluate()
        
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
        #l2twiddlel1twiddle_psi = 6*1j*(2/self.beta)*self.Q*o1.A - 3*(2/self.beta)*self.Q**2*o2.A21 + (2/self.beta)*o2.A21_xx - 12*self.Q**2*self.iR*o1.psi + 4*self.iR*o1.psi_xx - 4*1j*self.iR*self.Q**3*o2.psi21 + 4*self.iR*1j*self.Q*o2.psi21_xx + 2*o2.u21 # Umurhan+'s wrong definition of L2twiddle
        l2twiddlel1twiddle_psi = l2twiddlel1twiddle_psi.evaluate()
        
        #l2twiddlel1twiddle_u = (2/self.beta)*o1.B - 1j*self.Q*(2/self.beta)*o2.B21 - 1j*self.Q*(self.q - 2)*o2.psi21 + self.iR*o1.u #what?
        l2twiddlel1twiddle_u = (2/self.beta)*o2.B21 + 2*1j*self.iR*self.Q*o2.u21 + (self.q - 2)*o2.psi21 + self.iR*o1.u #correct
        #l2twiddlel1twiddle_u = (2/self.beta)*o2.B21 + 2*1j*self.iR*self.Q*o2.u21 + (self.q - 2)*o2.psi21 + 2*self.iR*o1.u #Umurhan+'s wrong def of L2twiddle
        l2twiddlel1twiddle_u = l2twiddlel1twiddle_u.evaluate()
        
        l2twiddlel1twiddle_A = self.iRm*o1.A + 2*1j*self.iRm*self.Q*o2.A21 + o2.psi21
        #l2twiddlel1twiddle_A = 2*self.iRm*o1.A + 2*1j*self.iRm*self.Q*o2.A21 + o2.psi21 #Umurhan+'s wrong def of L2twiddle
        l2twiddlel1twiddle_A = l2twiddlel1twiddle_A.evaluate()
        
        l2twiddlel1twiddle_B = -self.q*o2.A21 + self.iRm*o1.B + 2*1j*self.iRm*self.Q*o2.B21 + o2.u21
        #l2twiddlel1twiddle_B = -self.q*o2.A21 + 2*self.iRm*o1.B + 2*1j*self.iRm*self.Q*o2.B21 + o2.u21 #Umurhan+'s wrong def of L2twiddle
        l2twiddlel1twiddle_B = l2twiddlel1twiddle_B.evaluate()
        
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
        
        # a = <va . D V11*>
        self.a = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [a_psi_rhs, o1.u, o1.A, o1.B])
        self.a2 = self.take_inner_product2([ah.psi, ah.u, ah.A, ah.B], [a_psi_rhs2, o1.u_star, o1.A_star, o1.B_star]) #testing
        
        # c = <va . N31*>
        self.c = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [n3.N31_psi, n3.N31_u, n3.N31_A, n3.N31_B])
        
        # ctwiddle = < va . N31_twiddle_star >. Should be zero.
        self.ctwiddle = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [allzeros, c_twiddle_u_rhs, allzeros, c_twiddle_B_rhs])
        
        # b = < va . (X v11)* > :: in new terminology, b = < va . (Gtwiddle v11)* >
        #self.b = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [b_psi_rhs, o1.B, o1.psi, o1.u])
        self.b = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [b_psi_rhs, b_u_rhs, b_A_rhs, b_B_rhs])
  
        # h = < va . (L2twiddle v11 + L1twiddle v21)* >
        self.h = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [l2twiddlel1twiddle_psi, l2twiddlel1twiddle_u, l2twiddlel1twiddle_A, l2twiddlel1twiddle_B])
  
        # With new definition of b, no need for g
        # g = < va . (L3 v11) * >
        #self.g = self.take_inner_product([ah.psi, ah.u, ah.A, ah.B], [g_psi, allzeros, allzeros, allzeros])
    
        #self.linear_term = 1j*self.Q*self.b - 1j*self.Q**3*self.g
        self.linear_term = self.b
    
        print("sat_amp_coeffs = b/c")
        self.sat_amp_coeffs = np.sqrt(self.b/self.c) #np.sqrt((-1j*self.Q*self.b + 1j*self.Q**3*self.g)/self.c)
        print("a", self.a, "c", self.c, "ctwiddle", self.ctwiddle, "b", self.b, "h", self.h)#, "g", self.g)
        print("saturation amp", self.sat_amp_coeffs)
        
        # For interactive diagnostic purposes only
        self.o1 = o1
        self.o2 = o2
        self.n3 = n3
        self.ah = ah
        self.n2 = n2
        
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
        problem.expand(domain)
        
        solver = solvers.IVP(problem, domain, timesteppers.SBDF2)
        
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
    
        
