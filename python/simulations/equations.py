import numpy as np
import os
import subprocess
from mpi4py import MPI

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de

class Equations():
    def __init__(self):
        raw_id = subprocess.check_output(['hg','id','-i'])

        self.hg_version = raw_id.decode().strip()
        logger.info("equations version {}".format(self.hg_version))

        self.hg_diff = None
        if self.hg_version.endswith('+'):
            raw_diff = subprocess.check_output(['hg','diff'])
            self.hg_diff = raw_diff.decode()

    def set_IVP_problem(self, *args, **kwargs):
        self._set_domain()
        self.problem = de.IVP(self.domain, variables=self.variables)
        self.set_equations(*args, **kwargs)

    def set_eigenvalue_problem(self, *args, **kwargs):
        self._set_domain()
        self.problem = de.EVP(self.domain, variables=self.variables, eigenvalue='omega')
        self.problem.substitutions['dt(f)'] = "omega*f"
        self.set_equations(*args, **kwargs)

    def _apply_params(self):
        for k,v in self._eqn_params.items():
            self.problem.parameters[k] = v

    def set_equations(self, *args, **kwargs):
        self._apply_params()
        self._set_subs()
        self.set_aux()
        
    def initialize_output(self, solver ,data_dir, **kwargs):
        self.analysis_tasks = []
        analysis_slice = solver.evaluator.add_file_handler(os.path.join(data_dir,"slices"), max_writes=20, parallel=False, **kwargs)
        analysis_slice.add_task("psi", name="psi")
        analysis_slice.add_task("A", name="A")
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("b", name="b")
        
        self.analysis_tasks.append(analysis_slice)
        
        analysis_profile = solver.evaluator.add_file_handler(os.path.join(data_dir,"profiles"), max_writes=20, parallel=False, **kwargs)
        analysis_profile.add_task("plane_avg(KE)", name="KE")

        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(v_rms)", name="v_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        
        self.analysis_tasks.append(analysis_profile)

        analysis_scalar = solver.evaluator.add_file_handler(os.path.join(data_dir,"scalar"), max_writes=np.inf, parallel=False, **kwargs)
        analysis_scalar.add_task("vol_avg(KE)", name="KE")
        analysis_scalar.add_task("vol_avg(u_rms)", name="u_rms")
        analysis_scalar.add_task("vol_avg(v_rms)", name="v_rms")
        analysis_scalar.add_task("vol_avg(w_rms)", name="w_rms")

        self.analysis_tasks.append(analysis_scalar)

        # workaround for issue #29
        self.problem.namespace['u_rms'].store_last = True
        self.problem.namespace['v_rms'].store_last = True
        self.problem.namespace['w_rms'].store_last = True
        self.problem.namespace['KE'].store_last = True

        return self.analysis_tasks

    def set_BC(self):
        self.problem.add_bc("left(u) = 0")
        self.problem.add_bc("right(u) = 0")
        self.problem.add_bc("left(psi) = 0")
        self.problem.add_bc("right(psi) = 0")
        self.problem.add_bc("left(psi_x) = 0")
        self.problem.add_bc("right(psi_x) = 0")
        self.problem.add_bc("left(A) = 0")
        self.problem.add_bc("right(A) = 0")
        self.problem.add_bc("left(b_x) = 0")
        self.problem.add_bc("right(b_x) = 0")

    def _set_subs(self):
        """
        this implements the cylindrical del operators. 
        NB: ASSUMES THE EQUATION SET IS PREMULTIPLIED BY A POWER OF r (SEE BELOW)!!!

        Lap_s --> scalar laplacian
        Lap_r --> r component of vector laplacian
        Lap_t --> theta component of vector laplacian
        Lap_z --> z component of vector laplacian

        """
        #self.problem.substitutions['vel_sum_sq'] = 'u**2 + v**2 + w**2'

        # NB: this problem assumes delta = R2 - R1 = 1 
        self.problem.substitutions['plane_avg(A)'] = 'integ(A, "z")/Lz'
        self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lz'
        self.problem.substitutions['KE'] = '(psi_x**2 + dz(psi)**2)/2'
        self.problem.substitutions['u_rms'] = 'sqrt((dz(psi))**2)'
        self.problem.substitutions['v_rms'] = 'sqrt(u*u)'
        self.problem.substitutions['w_rms'] = 'sqrt(psi_x**2)'
        # self.problem.substitutions['Re_rms'] = 'sqrt(vel_sum_sq)*Lz/nu'
        # self.problem.substitutions['epicyclic_freq_sq']  = 'dr(r*r*v*v)/(r*r*r)'

class MRI_equations(Equations):
    """
    streamfunction/vector potential
    """

    def __init__(self, nx=32, nz=32, grid_dtype=np.float64, dealias=3/2, linear=False):
        super(MRI_equations,self).__init__()
        self.nx = nx 
        self.nz = nz
        self.grid_dtype = grid_dtype
        self.linear = linear
        logger.info("Grid_dtype = {}".format(self.grid_dtype))
        self.dealias = dealias

        self.equation_set = 'streamfunction/vector potential MRI'
        #self.variables = ['psi','u','A','b','psi_x','psi_xx','psi_xxx','u_x','A_x','A_xx', 'b_x']
        self.variables = ['psi','u','A','b','psi_x','psi_xx','psi_xxx','u_x','A_x','b_x']

    def _set_domain(self):
        """

        """
        #try:
        t_bases = self._set_transverse_bases()
        x_basis = self._set_x_basis()
        #except AttributeError:
        #    raise AttributeError("You must set parameters before constructing the domain.")

        bases = t_bases + x_basis
        
        self.domain = de.Domain(bases, grid_dtype=self.grid_dtype)        
        
    def _set_transverse_bases(self):
        z_basis = de.Fourier(  'z', self.nz, interval=[0., self.Lz], dealias=self.dealias)
        trans_bases = [z_basis,]

        return trans_bases

    def _set_x_basis(self):
        x_basis = de.Chebyshev('x', self.nx, interval=[-1., 1.], dealias=3/2)
        
        return [x_basis,]

    def set_equations(self, *args, **kwargs):
        super(MRI_equations,self).set_equations(*args, **kwargs)        

        self.set_streamfunction()
        self.set_vectorpotential()
        self.set_u()
        self.set_b()

    def set_parameters(self, Rm, Pm, eps, Omega0, qsh, beta, Q, Lz):
        """
        Lz is in units of 2*np.pi/Q
        """
        self.Rm = Rm
        self.Pm = Pm
        self.B0 = 1. - eps**2
        self.Omega0 = Omega0
        self.q = qsh
        self.beta = beta
        self.Lz = Lz * 2.*np.pi/Q 
            
        self._eqn_params = {}
        self._eqn_params['Re'] = self.Rm / self.Pm
        self._eqn_params['Rm'] = self.Rm
        self._eqn_params['B0'] = self.B0
        self._eqn_params['Omega0'] = self.Omega0
        self._eqn_params['q'] = self.q
        self._eqn_params['beta'] = self.beta
        self._eqn_params['Lz'] = self.Lz

    def set_streamfunction(self):
        if self.linear:
            RHS = "0"
        else:
            RHS = "2/beta*(dz(A)*(dx(dx(A_x)) + dz(dz(A_x))) - A_x*(dz(dx(A_x)) + dz(dz(dz(A))))) - ((psi_xxx + dz(dz(psi_x))) * dz(psi) - (dz(psi_xx) + dz(dz(dz(psi))))*psi_x)"

        self.problem.add_equation("dt(psi_xx) + dt(dz(dz(psi))) - 2*Omega0*dz(u) - (dx(psi_xxx) + dz(dz(dz(dz(psi)))))/Re - 2*(dz(dz(psi_xx)))/Re - 2*B0/beta*(dz(dx(A_x)) + dz(dz(dz(A)))) = " + RHS)

    def set_vectorpotential(self):
        if self.linear:
            RHS = "0"
        else:
            RHS = "dz(A) * psi_x - A_x * dz(psi)"

        self.problem.add_equation("dt(A) - B0 * dz(psi) - (dx(A_x) + dz(dz(A)))/Rm = "+RHS)

    def set_aux(self):
        self.problem.add_equation("psi_x - dx(psi) = 0")
        self.problem.add_equation("psi_xx - dx(psi_x) = 0")
        self.problem.add_equation("psi_xxx - dx(psi_xx) = 0")
        self.problem.add_equation("u_x - dx(u) = 0")
        self.problem.add_equation("A_x - dx(A) = 0")
        self.problem.add_equation("b_x - dx(b) = 0")

    def set_b(self):
        if self.linear:
            RHS = "0"
        else:
            RHS = "dz(A) * u_x - A_x * dz(u) - (dz(psi)*b_x - psi_x*dz(b))"
        self.problem.add_equation("dt(b) - B0*dz(u) + q*Omega0 * dz(A) - (dx(b_x) + dz(dz(b)))/Rm = " + RHS)

    def set_u(self):
        if self.linear:
            RHS = "0"
        else:
            RHS = "2./beta*(dz(A) * b_x - A_x * dz(b)) - (dz(psi)*u_x - psi_x*dz(u))"
        self.problem.add_equation("dt(u) + (2-q)*Omega0*dz(psi) - 2*B0/beta * dz(b) - (dx(u_x) + dz(dz(u)))/Re = "+RHS)
