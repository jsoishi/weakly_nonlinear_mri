import dedalus.public as de
import numpy as np
from mpi4py import MPI
from allorders_2 import AmplitudeAlpha
import h5py

# Q = 0.74955
# Rm = 4.898
# Pm = 5e-3
# Q = 0.7467
# Rm = 4.876
# Pm = 5e-4

#Q = 0.7470
#Rm = 4.879
#Pm = 1E-2#1e-3

# indx = 0 : Pm = 1E-4
tgc = h5py.File('../data/pm_sat_coeffs.h5', 'r')
for indx in range(len(tgc['Pm'].value)):
    Pm = tgc['Pm'].value[indx].real
    Rm = tgc['Rm_c'].value[indx].real
    Q = tgc['Q_c'].value[indx].real

    q = 1.5
    beta = 25.0

    gridnum = 128
    x_basis = de.Chebyshev('x',gridnum)
    domain = de.Domain([x_basis], np.complex128, comm=MPI.COMM_SELF)
    print("running at gridnum", gridnum)

    fn_root = "../data/"
    fn = fn_root + "thingap_amplitude_parameters_Q_{:03.2f}_Rm_{:04.4f}_Pm_{:.2e}_q_{:02.1f}_beta_{:.2f}_gridnum_{}.h5".format(Q, Rm, Pm, q, beta, gridnum)

    print(fn)

    aa = AmplitudeAlpha(domain,Q = Q,Rm = Rm, Pm = Pm, q=q, beta=beta)
    aa.print_coeffs()

    with h5py.File(fn,'w') as f:
        x = f.create_dataset("x", data=aa.x)
        o1psi = f.create_dataset("psi11", data=aa.o1.psi['g'])
        o1psi_x = f.create_dataset("psi11_x", data=aa.o1.psi_x['g'])
        o1psi_xx = f.create_dataset("psi11_xx", data=aa.o1.psi_xx['g'])
        o1psi_xxx = f.create_dataset("psi11_xxx", data=aa.o1.psi_xxx['g'])
        o1psi_star = f.create_dataset("psi11_star", data=aa.o1.psi_star['g'])
        o1psi_star_x = f.create_dataset("psi11_star_x", data=aa.o1.psi_star_x['g'])
        o1psi_star_xx = f.create_dataset("psi11_star_xx", data=aa.o1.psi_star_xx['g'])
        o1psi_star_xxx = f.create_dataset("psi11_star_xxx", data=aa.o1.psi_star_xxx['g'])
    
        o1u = f.create_dataset("u11", data=aa.o1.u['g'])
        o1u_x = f.create_dataset("u11_x", data=aa.o1.u_x['g'])
        o1u_xx = f.create_dataset("u11_xx", data=aa.o1.u_xx['g'])
        o1u_star = f.create_dataset("u11_star", data=aa.o1.u_star['g'])
        o1u_star_x = f.create_dataset("u11_star_x", data=aa.o1.u_star_x['g'])
    
        o1A = f.create_dataset("A11", data=aa.o1.A['g'])
        o1A_x = f.create_dataset("A11_x", data=aa.o1.A_x['g'])
        o1A_xx = f.create_dataset("A11_xx", data=aa.o1.A_xx['g'])
        o1A_xxx = f.create_dataset("A11_xxx", data=aa.o1.A_xxx['g'])
        o1A_star = f.create_dataset("A11_star", data=aa.o1.A_star['g'])
        o1A_star_x = f.create_dataset("A11_star_x", data=aa.o1.A_star_x['g'])
        o1A_star_xx = f.create_dataset("A11_star_xx", data=aa.o1.A_star_xx['g'])
        o1A_star_xxx = f.create_dataset("A11_star_xxx", data=aa.o1.A_star_xxx['g'])
    
        o1B = f.create_dataset("B11", data=aa.o1.B['g'])
        o1B_x = f.create_dataset("B11_x", data=aa.o1.B_x['g'])
        o1B_xx = f.create_dataset("B11_xx", data=aa.o1.B_xx['g'])
        o1B_star = f.create_dataset("B11_star", data=aa.o1.B_star['g'])
        o1B_star_x = f.create_dataset("B11_star_x", data=aa.o1.B_star_x['g'])
    
        ahpsi = f.create_dataset("ah_psi", data=aa.ah.psi['g'])
        ahpsi_x = f.create_dataset("ah_psi_x", data=aa.ah.psi_x['g'])
        ahpsi_xx = f.create_dataset("ah_psi_xx", data=aa.ah.psi_xx['g'])
        ahpsi_xxx = f.create_dataset("ah_psi_xxx", data=aa.ah.psi_xxx['g'])
        ahu = f.create_dataset("ah_u", data=aa.ah.u['g'])
        ahu_x = f.create_dataset("ah_u_x", data=aa.ah.u_x['g'])
        ahA = f.create_dataset("ah_A", data=aa.ah.A['g'])
        ahA_x = f.create_dataset("ah_A_x", data=aa.ah.A_x['g'])
        ahA_xx = f.create_dataset("ah_A_xx", data=aa.ah.A_xx['g'])
        ahA_xxx = f.create_dataset("ah_A_xxx", data=aa.ah.A_xxx['g'])
    
        ahB = f.create_dataset("ah_B", data=aa.ah.B['g'])
        ahB_x = f.create_dataset("ah_B_x", data=aa.ah.B_x['g'])
    
        psi20 = f.create_dataset("psi20", data=aa.o2.psi20['g'])
        psi20_x = f.create_dataset("psi20_x", data=aa.o2.psi20_x['g'])
        psi20_xx = f.create_dataset("psi20_xx", data=aa.o2.psi20_xx['g'])
        psi20_xxx = f.create_dataset("psi20_xxx", data=aa.o2.psi20_xxx['g'])
        psi20_star = f.create_dataset("psi20_star", data=aa.o2.psi20_star['g'])
        psi20_star_x = f.create_dataset("psi20_star_x", data=aa.o2.psi20_star_x['g'])
        psi20_star_xx = f.create_dataset("psi20_star_xx", data=aa.o2.psi20_star_xx['g'])
        psi20_star_xxx = f.create_dataset("psi20_star_xxx", data=aa.o2.psi20_star_xxx['g'])
    
        u20 = f.create_dataset("u20", data=aa.o2.u20['g'])
        u20_x = f.create_dataset("u20_x", data=aa.o2.u20_x['g'])
        u20_star = f.create_dataset("u20_star", data=aa.o2.u20_star['g'])
        u20_star_x = f.create_dataset("u20_star_x", data=aa.o2.u20_star_x['g'])
    
        A20 = f.create_dataset("A20", data=aa.o2.A20['g'])
        A20_x = f.create_dataset("A20_x", data=aa.o2.A20_x['g'])
        A20_xx = f.create_dataset("A20_xx", data=aa.o2.A20_xx['g'])
        A20_xxx = f.create_dataset("A20_xxx", data=aa.o2.A20_xxx['g'])
        A20_star = f.create_dataset("A20_star", data=aa.o2.A20_star['g'])
        A20_star_x = f.create_dataset("A20_star_x", data=aa.o2.A20_star_x['g'])
        A20_star_xx = f.create_dataset("A20_star_xx", data=aa.o2.A20_star_xx['g'])
        A20_star_xxx = f.create_dataset("A20_star_xxx", data=aa.o2.A20_star_xxx['g'])
    
        B20 = f.create_dataset("B20", data=aa.o2.B20['g'])
        B20_x = f.create_dataset("B20_x", data=aa.o2.B20_x['g'])
        B20_star = f.create_dataset("B20_star", data=aa.o2.B20_star['g'])
        B20_star_x = f.create_dataset("B20_star_x", data=aa.o2.B20_star_x['g'])
    
        psi21 = f.create_dataset("psi21", data=aa.o2.psi21['g'])
        psi21_x = f.create_dataset("psi21_x", data=aa.o2.psi21_x['g'])
        psi21_xx = f.create_dataset("psi21_xx", data=aa.o2.psi21_xx['g'])
    
        u21 = f.create_dataset("u21", data=aa.o2.u21['g'])
    
        A21 = f.create_dataset("A21", data=aa.o2.A21['g'])
        A21_x = f.create_dataset("A21_x", data=aa.o2.A21_x['g'])
        A21_xx = f.create_dataset("A21_xx", data=aa.o2.A21_xx['g'])
    
        B21 = f.create_dataset("B21", data=aa.o2.B21['g'])
    
        psi22 = f.create_dataset("psi22", data=aa.o2.psi22['g'])
        psi22_x = f.create_dataset("psi22_x", data=aa.o2.psi22_x['g'])
        psi22_xx = f.create_dataset("psi22_xx", data=aa.o2.psi22_xx['g'])
        psi22_xxx = f.create_dataset("psi22_xxx", data=aa.o2.psi22_xxx['g'])
    
        u22 = f.create_dataset("u22", data=aa.o2.u22['g'])
        u22_star = f.create_dataset("u22_star", data=aa.o2.u22_star['g'])
        u22_star_x = f.create_dataset("u22_star_x", data=aa.o2.u22_star_x['g'])
    
        A22 = f.create_dataset("A22", data=aa.o2.A22['g'])
        A22_x = f.create_dataset("A22_x", data=aa.o2.A22_x['g'])
        A22_xx = f.create_dataset("A22_xx", data=aa.o2.A22_xx['g'])
        A22_xxx = f.create_dataset("A22_xxx", data=aa.o2.A22_xxx['g'])
    
        B22 = f.create_dataset("B22", data=aa.o2.B22['g'])
        B22_star = f.create_dataset("B22_star", data=aa.o2.B22_star['g'])
        B22_star_x = f.create_dataset("B22_star_x", data=aa.o2.B22_star_x['g'])
    
        N20_psi = f.create_dataset("N20_psi", data=aa.n2.N20_psi['g'])
        N20_u = f.create_dataset("N20_u", data=aa.n2.N20_u['g'])
        N20_A = f.create_dataset("N20_A", data=aa.n2.N20_A['g'])
        N20_B = f.create_dataset("N20_B", data=aa.n2.N20_B['g'])
    
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
        f.attrs["beta"] = aa.beta
        f.attrs["gridnum"] = aa.gridnum
        f.attrs["q"] = aa.q
        
        f.attrs["a"] = aa.a
        f.attrs["b"] = aa.b
        f.attrs["c"] = aa.c
        f.attrs["h"] = aa.h