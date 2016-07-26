import dedalus.public as de
import numpy as np
from mpi4py import MPI
from allorders2_widegap import AmplitudeAlpha
import h5py

# UMR07-like parameters

#Q = 0.01269
#Rm = 0.67355
#Pm = 1.0e-3
beta = 25.0
Omega1 = 313.55
Omega2 = 56.43
xi = 0
R1 = 5
R2 = 15

indx = 0
crit_params = h5py.File('../../data/widegap_crit_Rm_Q.h5', 'r')
Q = crit_params['Q_c'].value[indx]
Rm = crit_params['Rm_c'].value[indx]
Pm = crit_params['Pm'].value[indx]

# Hollerbach & Rudiger
"""
Pm = 1.0E-6
Re = 1521
Rm = Pm*Re
xi = 4
Q = 2.33
Ha = 16.3
beta = (2*Re*Rm)/(Ha**2)
R1 = 1.
R2 = 2.
mu_omega = 0.27
Omega1 = 313.55
Omega2 = Omega1*mu_omega
"""

gridnum = 128

r_basis = de.Chebyshev('r', gridnum, interval=(R1, R2))
domain = de.Domain([r_basis], np.complex128, comm=MPI.COMM_SELF)
print("running at gridnum", gridnum)

aa = AmplitudeAlpha(domain, Q = Q, Rm = Rm, Pm = Pm, xi = xi, Omega1 = Omega1, Omega2 = Omega2, beta = beta)
aa.print_coeffs()


#fn_root = "/Users/susanclark/weakly_nonlinear_mri/data/"
fn_root = "../../data/"
fn = fn_root + "widegap_amplitude_parameters_Q_{:03.2f}_Rm_{:04.4f}_Pm_{:.2e}_Omega1_{:05.2f}_Omega2_{:05.2f}_beta_{:.2f}_xi_{:.2f}_gridnum_{}_intnorm.h5".format(Q, Rm, Pm, Omega1, Omega2, beta, xi, gridnum)
with h5py.File(fn,'w') as f:
    r = f.create_dataset("r", data=aa.r)
    o1psi = f.create_dataset("psi11", data=aa.o1.psi['g'])
    o1psi_r = f.create_dataset("psi11_r", data=aa.o1.psi_r['g'])
    o1psi_rr = f.create_dataset("psi11_rr", data=aa.o1.psi_rr['g'])
    o1psi_rrr = f.create_dataset("psi11_rrr", data=aa.o1.psi_rrr['g'])
    o1psi_star = f.create_dataset("psi11_star", data=aa.o1.psi_star['g'])
    o1psi_star_r = f.create_dataset("psi11_star_r", data=aa.o1.psi_star_r['g'])
    o1psi_star_rr = f.create_dataset("psi11_star_rr", data=aa.o1.psi_star_rr['g'])
    o1psi_star_rrr = f.create_dataset("psi11_star_rrr", data=aa.o1.psi_star_rrr['g'])
    
    o1u = f.create_dataset("u11", data=aa.o1.u['g'])
    o1u_r = f.create_dataset("u11_r", data=aa.o1.u_r['g'])
    o1u_rr = f.create_dataset("u11_rr", data=aa.o1.u_rr['g'])
    o1u_star = f.create_dataset("u11_star", data=aa.o1.u_star['g'])
    o1u_star_r = f.create_dataset("u11_star_r", data=aa.o1.u_star_r['g'])
    
    o1A = f.create_dataset("A11", data=aa.o1.A['g'])
    o1A_r = f.create_dataset("A11_r", data=aa.o1.A_r['g'])
    o1A_rr = f.create_dataset("A11_rr", data=aa.o1.A_rr['g'])
    o1A_rrr = f.create_dataset("A11_rrr", data=aa.o1.A_rrr['g'])
    o1A_star = f.create_dataset("A11_star", data=aa.o1.A_star['g'])
    o1A_star_r = f.create_dataset("A11_star_r", data=aa.o1.A_star_r['g'])
    o1A_star_rr = f.create_dataset("A11_star_rr", data=aa.o1.A_star_rr['g'])
    o1A_star_rrr = f.create_dataset("A11_star_rrr", data=aa.o1.A_star_rrr['g'])
    
    o1B = f.create_dataset("B11", data=aa.o1.B['g'])
    o1B_r = f.create_dataset("B11_r", data=aa.o1.B_r['g'])
    o1B_rr = f.create_dataset("B11_rr", data=aa.o1.B_rr['g'])
    o1B_star = f.create_dataset("B11_star", data=aa.o1.B_star['g'])
    o1B_star_r = f.create_dataset("B11_star_r", data=aa.o1.B_star_r['g'])
    
    ahpsi = f.create_dataset("ah_psi", data=aa.ah.psi['g'])
    ahpsi_r = f.create_dataset("ah_psi_r", data=aa.ah.psi_r['g'])
    ahpsi_rr = f.create_dataset("ah_psi_rr", data=aa.ah.psi_rr['g'])
    ahpsi_rrr = f.create_dataset("ah_psi_rrr", data=aa.ah.psi_rrr['g'])
    ahu = f.create_dataset("ah_u", data=aa.ah.u['g'])
    ahu_r = f.create_dataset("ah_u_r", data=aa.ah.u_r['g'])
    ahA = f.create_dataset("ah_A", data=aa.ah.A['g'])
    ahA_r = f.create_dataset("ah_A_r", data=aa.ah.A_r['g'])
    ahA_rr = f.create_dataset("ah_A_rr", data=aa.ah.A_rr['g'])
    ahA_rrr = f.create_dataset("ah_A_rrr", data=aa.ah.A_rrr['g'])
    
    ahB = f.create_dataset("ah_B", data=aa.ah.B['g'])
    ahB_r = f.create_dataset("ah_B_r", data=aa.ah.B_r['g'])
    
    psi20 = f.create_dataset("psi20", data=aa.o2.psi20['g'])
    psi20_r = f.create_dataset("psi20_r", data=aa.o2.psi20_r['g'])
    psi20_rr = f.create_dataset("psi20_rr", data=aa.o2.psi20_rr['g'])
    psi20_rrr = f.create_dataset("psi20_rrr", data=aa.o2.psi20_rrr['g'])
    psi20_star = f.create_dataset("psi20_star", data=aa.o2.psi20_star['g'])
    psi20_star_r = f.create_dataset("psi20_star_r", data=aa.o2.psi20_star_r['g'])
    psi20_star_rr = f.create_dataset("psi20_star_rr", data=aa.o2.psi20_star_rr['g'])
    psi20_star_rrr = f.create_dataset("psi20_star_rrr", data=aa.o2.psi20_star_rrr['g'])
    
    u20 = f.create_dataset("u20", data=aa.o2.u20['g'])
    u20_r = f.create_dataset("u20_r", data=aa.o2.u20_r['g'])
    u20_star = f.create_dataset("u20_star", data=aa.o2.u20_star['g'])
    u20_star_r = f.create_dataset("u20_star_r", data=aa.o2.u20_star_r['g'])
    
    A20 = f.create_dataset("A20", data=aa.o2.A20['g'])
    A20_r = f.create_dataset("A20_r", data=aa.o2.A20_r['g'])
    A20_rr = f.create_dataset("A20_rr", data=aa.o2.A20_rr['g'])
    A20_rrr = f.create_dataset("A20_rrr", data=aa.o2.A20_rrr['g'])
    A20_star = f.create_dataset("A20_star", data=aa.o2.A20_star['g'])
    A20_star_r = f.create_dataset("A20_star_r", data=aa.o2.A20_star_r['g'])
    A20_star_rr = f.create_dataset("A20_star_rr", data=aa.o2.A20_star_rr['g'])
    A20_star_rrr = f.create_dataset("A20_star_rrr", data=aa.o2.A20_star_rrr['g'])
    
    B20 = f.create_dataset("B20", data=aa.o2.B20['g'])
    B20_r = f.create_dataset("B20_r", data=aa.o2.B20_r['g'])
    B20_star = f.create_dataset("B20_star", data=aa.o2.B20_star['g'])
    B20_star_r = f.create_dataset("B20_star_r", data=aa.o2.B20_star_r['g'])
    
    psi21 = f.create_dataset("psi21", data=aa.o2.psi21['g'])
    psi21_r = f.create_dataset("psi21_r", data=aa.o2.psi21_r['g'])
    psi21_rr = f.create_dataset("psi21_rr", data=aa.o2.psi21_rr['g'])
    
    u21 = f.create_dataset("u21", data=aa.o2.u21['g'])
    
    A21 = f.create_dataset("A21", data=aa.o2.A21['g'])
    A21_r = f.create_dataset("A21_r", data=aa.o2.A21_r['g'])
    A21_rr = f.create_dataset("A21_rr", data=aa.o2.A21_rr['g'])
    
    B21 = f.create_dataset("B21", data=aa.o2.B21['g'])
    
    psi22 = f.create_dataset("psi22", data=aa.o2.psi22['g'])
    psi22_r = f.create_dataset("psi22_r", data=aa.o2.psi22_r['g'])
    psi22_rr = f.create_dataset("psi22_rr", data=aa.o2.psi22_rr['g'])
    psi22_rrr = f.create_dataset("psi22_rrr", data=aa.o2.psi22_rrr['g'])
    
    u22 = f.create_dataset("u22", data=aa.o2.u22['g'])
    u22_star = f.create_dataset("u22_star", data=aa.o2.u22_star['g'])
    u22_star_r = f.create_dataset("u22_star_r", data=aa.o2.u22_star_r['g'])
    
    A22 = f.create_dataset("A22", data=aa.o2.A22['g'])
    A22_r = f.create_dataset("A22_r", data=aa.o2.A22_r['g'])
    A22_rr = f.create_dataset("A22_rr", data=aa.o2.A22_rr['g'])
    A22_rrr = f.create_dataset("A22_rrr", data=aa.o2.A22_rrr['g'])
    
    B22 = f.create_dataset("B22", data=aa.o2.B22['g'])
    B22_star = f.create_dataset("B22_star", data=aa.o2.B22_star['g'])
    B22_star_r = f.create_dataset("B22_star_r", data=aa.o2.B22_star_r['g'])
    
    N20_psi = f.create_dataset("N20_psi", data=aa.n2.N20_psi['g'])
    N20_u = f.create_dataset("N20_u", data=aa.n2.N20_u['g'])
    N20_A = f.create_dataset("N20_A", data=aa.n2.N20_A['g'])
    N20_B = f.create_dataset("N20_B", data=aa.n2.N20_B['g'])
    
    # additional diagnostics
    N20_psi_r4 = f.create_dataset("N20_psi_r4", data=aa.n2.N20_psi_r4_rederived['g'])
    N20_u_r2 = f.create_dataset("N20_u_r2", data=aa.n2.N20_u_r2_rederived['g'])
    N20_A_r = f.create_dataset("N20_A_r", data=aa.n2.N20_A_r_rederived['g'])
    N20_B_r2 = f.create_dataset("N20_B_r2", data=aa.n2.N20_B_r2_rederived['g'])
    
    N22_psi = f.create_dataset("N22_psi", data=aa.n2.N22_psi['g'])
    N22_u = f.create_dataset("N22_u", data=aa.n2.N22_u['g'])
    N22_A = f.create_dataset("N22_A", data=aa.n2.N22_A['g'])
    N22_B = f.create_dataset("N22_B", data=aa.n2.N22_B['g'])
    
    N31_psi = f.create_dataset("N31_psi", data=aa.n3.N31_psi['g'])
    N31_u = f.create_dataset("N31_u", data=aa.n3.N31_u['g'])
    N31_A = f.create_dataset("N31_A", data=aa.n3.N31_A['g'])
    N31_B = f.create_dataset("N31_B", data=aa.n3.N31_B['g'])
    
    f.attrs["Pm"] = aa.Pm
    f.attrs["Q"] = aa.Q
    f.attrs["Rm"] = aa.Rm
    f.attrs["xi"] = aa.xi
    f.attrs["Omega1"] = aa.Omega1
    f.attrs["Omega2"] = aa.Omega2
    f.attrs["beta"] = aa.beta
    f.attrs["R1"] = aa.R1
    f.attrs["R2"] = aa.R2
    f.attrs["gridnum"] = aa.gridnum
    f.attrs["q_R0"] = aa.q_R0
        
    f.attrs["a"] = aa.a
    f.attrs["b"] = aa.b
    f.attrs["c"] = aa.c
    f.attrs["h"] = aa.h


