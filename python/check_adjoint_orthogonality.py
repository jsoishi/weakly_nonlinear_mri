import numpy as np
import dedalus.public as de

from allorders_2 import AdjointHomogenous

AH = AdjointHomogenous()
O1 = AH.o1
scale = (0.18+3/11.*0.02)/AH.B['g'][0].imag
Q = AH.Q
x = AH.EP.solver.domain.grid(0)

def adjoint_inner_product(bra, ket, x, Q, keys=None):
    """assume bra, ket are dedalus state vectors; x is a string giving the
    coordinate to integrate over.
    
    computes 
    <bra|ket> = integral(bra* dot ket dx)
    """
    
    d = bra.domain.new_field()
    if not keys:
        keys = bra.state.field_dict.keys()
        
    for k in keys:
        if k == 'psi':
            d['g'] += bra.state[k]['g'].conj() * (ket.state['psixx']['g'] - Q**2 * ket.state[k]['g'])
        else:
            d['g'] += bra.state[k]['g'].conj() * ket.state[k]['g']
    
    return d.integrate(x)['g'][0]


print("First, check fastest growing mode against adjoint")
ip = adjoint_inner_product(AH.EP.solver,O1.EP.solver,'x',Q,keys=['psi','u','A','B'])

print("Inner product = {}".format(ip))

print("Now, check another mode against adjoint")
mode2 = O1.largest_eval_indx - 1
O1.EP.solver.set_state(mode2)

ip2 = adjoint_inner_product(AH.EP.solver,O1.EP.solver,'x',Q, keys=['psi','u','A','B'])
print("ip2/ip = {}".format(ip2/ip))
