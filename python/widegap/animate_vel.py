import matplotlib 
matplotlib.use('Agg')

import numpy as np
from scipy import integrate, interpolate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import h5py



N_trajectories = 20

"""
def lorentz_deriv((x, y, z), t0, sigma=10., beta=8./3, rho=28.0):
    #Compute the time-derivative of a Lorentz system.
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


# Choose random starting points, uniformly distributed from -15 to 15
np.random.seed(1)
x0 = -15 + 30 * np.random.random((N_trajectories, 3))

# Solve for the trajectories
t = np.linspace(0, 4, 1000)
x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t)
                  for x0i in x0])

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
#ax.axis('off')

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

# prepare the axes limits
ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((5, 55))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=500, interval=30, blit=True)

# Save as mp4. This requires mplayer or ffmpeg to be installed
#anim.save('lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

plt.show()
"""


N_trajectories = 15

widegap = True

if widegap is True:
    file_root = "/Users/susanclark/weakly_nonlinear_mri/data/"
    #fn = "widegap_amplitude_parameters_Q_0.01_Rm_0.6735_Pm_1.00e-03_Omega1_313.55_Omega2_56.43_beta_25.00_xi_0.00_gridnum_128.h5"
    fn = "widegap_amplitude_parameters_Q_0.01_Rm_0.6735_Pm_1.00e-03_Omega1_313.55_Omega2_56.43_beta_25.00_xi_0.00_gridnum_128_intnorm.h5"
    obj = h5py.File(file_root + fn, "r")

    Q = obj.attrs['Q']
    rgrid = obj['r'].value

    # epsilon (small parameter)
    eps = 0.5

    # saturation amplitude -- for now just constant, coefficient-determined
    satamp = 1#np.sqrt(obj.attrs['b']/obj.attrs['c']) #1

    # create z grid
    nz = obj.attrs['gridnum']
    Lz = 5*np.pi/Q # controls height 
    zgrid = np.linspace(0, Lz, nz, endpoint=False)
    zz = zgrid.reshape(nz, 1)

    dz = zgrid[1] - zgrid[0]

    # impart structure in the z direction
    eiqz = np.cos(Q*zz) + 1j*np.sin(Q*zz)
    eiqz_z = 1j*Q*np.cos(Q*zz) - Q*np.sin(Q*zz) # dz(e^{ikz})

    ei2qz = np.cos(2*Q*zz) + 1j*np.sin(2*Q*zz)
    ei0qz = np.cos(0*Q*zz) + 1j*np.sin(0*Q*zz)

    ei2qz_z = 2*1j*Q*np.cos(2*Q*zz) - 2*Q*np.sin(2*Q*zz)

    # two-dimensional u and Bstructure
    V1_u = eps*satamp*obj['u11'].value*eiqz
    V1_B = eps*satamp*obj['B11'].value*eiqz

    V2_u = eps**2*satamp**2*obj['u22'].value*ei2qz + eps**2*(np.abs(satamp))**2*obj['u20'].value*ei0qz
    V2_B = eps**2*satamp**2*obj['B22'].value*ei2qz + eps**2*(np.abs(satamp))**2*obj['B20'].value*ei0qz

    Vboth_B = V1_B + V2_B
    Vboth_u = V1_u + V2_u
    
    Omega1 = obj.attrs['Omega1']
    Omega2 = obj.attrs['Omega2']
    R1 = obj.attrs['R1']
    R2 = obj.attrs['R2']
    c1 = (Omega2*R2**2 - Omega1*R1**2)/(R2**2 - R1**2)
    c2 = (R1**2*R2**2*(Omega1 - Omega2))/(R2**2 - R1**2)
    
    base_flow_rdim = rgrid*c1 + c2/rgrid
    base_flow = (base_flow_rdim*ei0qz)/((R1 + R2)/2)
    
    norm_base_flow = base_flow/np.nanmax(base_flow)
    
    #Vboth_u = Vboth_u + norm_base_flow

    Vboth_uz1 = -eps*((1/rgrid)*satamp*obj['psi11_r'].value*eiqz) + eps**2*(-satamp**2*(1/rgrid)*ei2qz*obj['psi22_r'].value - (np.abs(satamp))**2*(1/rgrid)*obj['psi20_r'].value*ei0qz)
    Vboth_ur1 = eps*((1/rgrid)*satamp*obj['psi11'].value*eiqz_z) + eps**2*(satamp**2*ei2qz_z*obj['psi22'].value)

    #Vboth_Bz1 = eps*((1/r)*satamp*obj['A11_r'].value*eiqz) + eps**2*(-satamp**2*(1/r)*ei2qz*obj['A22_r'].value - (np.abs(satamp))**2*(1/r)*obj['A20_r'].value*ei0qz)
    #Vboth_Br1 = eps*((1/r)*satamp*obj['A11'].value*eiqz_z) + eps**2*(satamp**2*ei2qz_z*obj['A22'].value)


else:
    file_root = "/Users/susanclark/weakly_nonlinear_mri/data/"
    fn = "thingap_amplitude_parameters_Q_0.75_Rm_4.8790_Pm_1.00e-03_q_1.5_beta_25.00_gridnum_128.h5"
    obj = h5py.File(file_root + fn, "r")

    Q = obj.attrs['Q']
    rgrid = obj['x'].value + 1.1

    # epsilon (small parameter)
    eps = 0.5

    # saturation amplitude -- for now just constant, coefficient-determined
    satamp = np.sqrt(obj.attrs['b']/obj.attrs['c']) #1

    # create z grid
    nz = obj.attrs['gridnum']
    Lz = 2*np.pi/Q
    zgrid = np.linspace(0, Lz, nz, endpoint=False)
    zz = zgrid.reshape(nz, 1)

    dz = zgrid[1] - zgrid[0]

    # impart structure in the z direction
    eiqz = np.cos(Q*zz) + 1j*np.sin(Q*zz)
    eiqz_z = 1j*Q*np.cos(Q*zz) - Q*np.sin(Q*zz) # dz(e^{ikz})

    ei2qz = np.cos(2*Q*zz) + 1j*np.sin(2*Q*zz)
    ei0qz = np.cos(0*Q*zz) + 1j*np.sin(0*Q*zz)

    ei2qz_z = 2*1j*Q*np.cos(2*Q*zz) - 2*Q*np.sin(2*Q*zz)

    # two-dimensional u and Bstructure
    V1_u = eps*satamp*obj['u11'].value*eiqz
    V1_B = eps*satamp*obj['B11'].value*eiqz

    V2_u = eps**2*satamp**2*obj['u22'].value*ei2qz + eps**2*(np.abs(satamp))**2*obj['u20'].value*ei0qz
    V2_B = eps**2*satamp**2*obj['B22'].value*ei2qz + eps**2*(np.abs(satamp))**2*obj['B20'].value*ei0qz

    Vboth_B = V1_B + V2_B
    Vboth_u = V1_u + V2_u

    Vboth_uz1 = eps*(satamp*obj['psi11_x'].value*eiqz) + eps**2*(-satamp**2*ei2qz*obj['psi22_x'].value - (np.abs(satamp))**2*obj['psi20_x'].value*ei0qz)
    Vboth_ur1 = eps*(satamp*obj['psi11'].value*eiqz_z) + eps**2*(satamp**2*ei2qz_z*obj['psi22'].value)
        

phi_interpolator = interpolate.interp2d(rgrid, zgrid, Vboth_u)
z_interpolator = interpolate.interp2d(rgrid, zgrid, Vboth_uz1) 
r_interpolator = interpolate.interp2d(rgrid, zgrid, Vboth_ur1)   

#B_phi_interpolator = interpolate.interp2d(rgrid, zgrid, Vboth_B)
#B_z_interpolator = interpolate.interp2d(rgrid, zgrid, Vboth_Bz1) 
#B_r_interpolator = interpolate.interp2d(rgrid, zgrid, Vboth_Br1)        

def vel_deriv((r, phi, z), t0):
    #Compute the 3D velocities.
    
    # 2D interpolation on Vboth_u to get phi component of velocity
    vel_r = r_interpolator(r, z)[0]#[0]
    vel_phi = phi_interpolator(r, z)[0]#[0]
    vel_z = z_interpolator(r, z)[0]#[0]
    
    return [vel_r, vel_phi, vel_z]
    
# Choose random starting points, uniformly distributed from 5 to 15
x0 = np.zeros((N_trajectories, 3), np.float_)
np.random.seed(1)
x0[:, 0] = np.sort(5 + 10 * np.random.random(N_trajectories))
x0[:, 1] = 2*np.pi * np.random.random(N_trajectories)
x0[:, 2] = np.max(zgrid) * np.random.random(N_trajectories) #np.max(zgrid)/2.0 + 0.1 * np.random.random(N_trajectories)

# change trajectories to known values
#print("caution: non-random initial points for trajectories")
#for i in range(N_trajectories):
#    x0[i, 0] = 5.0 #+ i*(10.0/N_trajectories)
#    x0[i, 1] = 0.0 #+ i*(2*np.pi/N_trajectories)
#    x0[i, 2] = 0.0 + i*(np.max(zgrid)/N_trajectories)

# set one last starting point to something known
x0special = np.zeros((1, 3), np.float_)
x0special[0, 0] = 5
x0special[0, 1] = 0
x0special[0, 2] = 0

#x0[:, 1] = -1 + 2 * np.random.random(N_trajectories)

#np.random.seed(1)
#x0 = 5 + 10 * np.random.random((N_trajectories, 3))

#if widegap is False:
#    x0 = -1 + 2 * np.random.random((N_trajectories, 3))


# Solve for the trajectories
t = np.linspace(0, 500, 1000)#10000)
x_t = np.asarray([integrate.odeint(vel_deriv, x0i, t)
                  for x0i in x0])
                  
#tspecial = np.linspace(0, 1000000, 10000)
#x_tspecial = np.asarray([integrate.odeint(vel_deriv, x0i, t)
#                  for x0i in x0special])

#print("t:", t)
#print("x_t:", x_t)

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
#ax.axis('off')

# choose a different color for each trajectory
colors = plt.cm.spectral(np.linspace(0, 1, N_trajectories))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

# prepare the axes limits

ax.set_xlim((5, 15))
ax.set_ylim((-15, 15))
ax.set_zlim((-15, 15))

if widegap == False:
    ax.set_xlim((-1, 1))
    ax.set_ylim((-15, 15))
    ax.set_zlim((-15, 15))


# set point-of-view: specified by (altitude degrees, azimuth degrees)
#ax.view_init(30, 0)

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts


fig = plt.figure(facecolor="white", figsize=(8, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_rasterized(True)
#ax2 = fig.add_subplot(122, projection='3d')

# set point-of-view: specified by (altitude degrees, azimuth degrees)
#ax2.view_init(90, 0)

plotlastfirst = False
if plotlastfirst is True:
    for i in range(N_trajectories):
        # Convert to cartesian for plotting
        plot_r = x_t[i, :, 0]
        plot_phi = x_t[i, :, 1]
        plot_z = x_t[i, :, 2]

        plot_x = plot_r*np.cos(plot_phi)
        plot_y = plot_r*np.sin(plot_phi)
    
        #ax1.plot(plot_x, plot_y, plot_z, alpha=0.2, lw=2.0, c=colors[i])
    
        #xpair = (x for x in plot_x)
        #ypair = (x for x in plot_y)
        #zpair = (x for x in plot_z)
    
        for xx, yy, zz in zip(zip(plot_x, plot_x[1:]), zip(plot_y, plot_y[1:]), zip(plot_z, plot_z[1:])):
            ax1.plot(xx, yy, zz, alpha=0.2, c=colors[i], ls='solid')
    

    
#for i in xrange(len(plot_x) - 1):
#    ax1.plot(plot_x[i:i+1], plot_y[i:i+1], plot_z[i:i+1], alpha=0.2) 

    #for xx, yy, zz in zip(plot_x, plot_y, plot_z):
    #    print(xx, yy, zz)
    #    ax1.plot(xx, yy, zz, alpha=0.2)#, c=colors[i]);


#plot_x_special = x_tspecial[0, :, 0]*np.cos(x_tspecial[0, :, 1])
#plot_y_special = x_tspecial[0, :, 0]*np.sin(x_tspecial[0, :, 1])
#plot_z_special = x_tspecial[0, :, 2]

#ax2.plot(plot_x_special, plot_y_special, plot_z_special, alpha=0.5, lw=1.2, c="black")
    

#ax2.plot(plot_x, plot_y, plot_z, alpha=0.2, lw=1.2, c=colors[-1])

ax1.axis('off')
#ax2.axis('off')
ax1.set_rasterized(True)
plt.savefig("/Users/susanclark/weakly_nonlinear_mri/python/widegap/frames/3Dvel/final_widegapvel.png", bbox_inches='tight')


# build argument list
#call_list = []
#for xends,yends in zip(xpairs,ypairs):
#    call_list.append(xends)
#    call_list.append(yends)
#    call_list.append('-b')
#
#pp.plot(*call_list,alpha=0.1)



tstep = 2#10
#tol = 0.1

for tt in range(len(x_t[0, :, 0])):
    fig = plt.figure(facecolor="white", figsize=(6, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((-15, 15))
    ax.set_ylim((-15, 15))
    ax.set_zlim((0, Lz))
    ax.set_rasterized(True)
    ax.axis('off')

    for i in range(N_trajectories):
        # Convert to cartesian for plotting
        plot_r = x_t[i, :tt*tstep, 0]
        plot_phi = x_t[i, :tt*tstep, 1]
        plot_z = x_t[i, :tt*tstep, 2]

        plot_x = plot_r*np.cos(plot_phi)
        plot_y = plot_r*np.sin(plot_phi)
        
        if tt > 1:
            allspeed = np.sqrt((plot_x[1:] - plot_x[:-1])**2 + (plot_y[1:] - plot_y[:-1])**2 + (plot_z[1:] - plot_z[:-1])**2)
            maxspeed = np.nanmax(allspeed)
            minspeed = np.nanmin(allspeed)
            #print(maxspeed)    
        #ax.plot(plot_x, plot_y, plot_z, lw=2.0, c=colors[i])
        
        for xx, yy, zz in zip(zip(plot_x, plot_x[1:]), zip(plot_y, plot_y[1:]), zip(plot_z, plot_z[1:])):
            ax.plot(xx, yy, zz, alpha=0.2, ls='solid', c='black') #c=colors[i])
            
            #if i <10:
            #    print(xx, yy, zz)
        
        if tt > 1:
            #ax.scatter(plot_x[-1], plot_y[-1], plot_z[-1], '.', c='orangered', edgecolors='orangered', alpha=0.9)##c=colors[i])
            for i in np.arange(.1,1.01,.1):
                relspeed = np.sqrt((plot_x[-1] - plot_x[-2])**2 + (plot_y[-1] - plot_y[-2])**2 + (plot_z[-1] - plot_z[-2])**2)
                #print("relspeed/maxspeed", relspeed/maxspeed)
                ax.scatter(plot_x[-1], plot_y[-1], plot_z[-1], s=(50*i*(((relspeed - minspeed)/maxspeed)*.3+.1))**2, color=(1,0.307,0,.5/i/10))
        
        
    plt.savefig("/Users/susanclark/weakly_nonlinear_mri/python/widegap/frames/3Dvel/tallcolumn/widegapvel_{:03d}.png".format(tt))
    
    plt.close()


# instantiate the animator.
#anim = animation.FuncAnimation(fig, animate, init_func=init,
#                               frames=500, interval=30, blit=True)

# Save as mp4. This requires mplayer or ffmpeg to be installed
#anim.save('mri_anim_test.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

plt.show()

