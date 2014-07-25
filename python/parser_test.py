import numpy as np
import matplotlib.pyplot as plt
from dedalus2.public import *
from dedalus2.pde.solvers import LinearEigenvalue, LinearBVP
import plot_tools

gridnum = 64
Q = 0.75

# possible shortened operators
MagSq = operators.MagSquared
Abs = operators.Absolute

problem = ParsedProblem(axis_names=['Z'],
                   field_names=['alpha', 'alphaZ'],
                   param_names=['a', 'b', 'c', 'h', 'g', 'Q'])#, 'Abs'])

#problem.add_equation("a*dt(alpha) + b*alphaZ - h*dZ(alphaZ) - g*1j*Q**3*alpha = -c*alpha*Absolute(alpha**2)")
#problem.add_equation("a*dt(alpha) + b*alphaZ - h*dZ(alphaZ) - g*1j*Q**3*alpha = -c*alpha*MagSq(alpha)")
problem.add_equation("a*dt(alpha) + b*alphaZ - h*dZ(alphaZ) - g*1j*Q**3*alpha = c*alpha*MagSquared(alpha)")
problem.add_equation("alphaZ - dZ(alpha) = 0")

#problem.add_equation("-(a/c)*dt(alpha) - (b/c)*alphaZ + (h/c)*dZ(alphaZ) + (g/c)*1j*Q**3*alpha = alpha*Absolute(alpha**2)")
        
problem.parameters['a'] = -0.43304107388011209+9.9502100165750874e-16j
problem.parameters['b'] = 1.5800469117209944e-15-0.0018542860461813176j
problem.parameters['c'] = 0.67590648491827332+5.0199606428696532e-13j
problem.parameters['h'] = -0.06803788569729452-0.00041479168022338464j
problem.parameters['g'] = 1.4691613554954249e-17+2.628877940581315e-05j
problem.parameters['Q'] = Q
#problem.parameters['MagSq'] = MagSq
#problem.parameters['Abs'] = Abs

lambda_crit = 2*np.pi/Q

Z_basis = Fourier(gridnum, interval=(-lambda_crit, lambda_crit), dealias=2/3)
domain = Domain([Z_basis], np.complex128)
problem.expand(domain)

solver = solvers.IVP(problem, domain, timesteppers.SBDF2)

# stopping criteria
solver.stop_sim_time = np.inf
solver.stop_wall_time = np.inf
solver.stop_iteration = 5000

# reference local grid and state fields
Z = domain.grid(0)
alpha = solver.state['alpha']
alphaZ = solver.state['alphaZ']

# alpha initial conditions
alpha['g'] = 1
alpha.differentiate(0, out=alphaZ)

# storage
alpha_list = [np.copy(alpha['g'])]
t_list = [solver.sim_time]

# Main loop
dt = 2e-3
while solver.ok:
    solver.step(dt)
    if solver.iteration % 20 == 0:
        alpha_list.append(np.copy(alpha['g']))
        t_list.append(solver.sim_time)
        
# Convert storage to arrays
alpha_array = np.array(alpha_list)
t_array = np.array(t_list)

# Plot
xmesh, ymesh = plot_tools.quad_mesh(x=Z, y=t_array)
plt.figure(figsize=(10,6))
plt.pcolormesh(xmesh, ymesh, alpha_array, cmap='RdBu_r')
plt.axis(plot_tools.pad_limits(xmesh, ymesh))
plt.colorbar()
plt.xlabel('Z')
plt.ylabel('t')

saturation_amplitude = alpha_array[-1, 0]
