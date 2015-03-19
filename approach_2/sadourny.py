#!/usr/bin/env python
#  SW_Sadourny.m
#
# Solve the 1-Layer Rotating Shallow Water (SW) Model
#
# Fields:
#   u : zonal velocity
#   v : meridional velocity
#   h : fluid depth
#
# Evolution Eqns:
#   B = g*h + 0.5*(u**2 + v**2)     Bernoulli function
#   Z = v_x - u_y + f               Total Vorticity
#   q = Z/h                         Potential Vorticity
#   [U,V] = h[u,v]                  Transport velocities
#
#	u_t =  (q*V^x)^y + d_x h
#	v_t = -(q*U^y)^x + d_y h
#	h_t = - div[U,V]
#
# Geometry: periodic in x and y
#           Arakawa C-grid
#
#      |           |          |         |
#      h --  u --  h  -- u -- h -- u -- h --
#      |           |          |         |
#      |           |          |         |
#      v     q     v     q    v    q    v
#      |           |          |         |
#      |           |          |         |
#      h --  u --  h  -- u -- h -- u -- h --
#      |           |          |         |
#      |           |          |         |
#      v     q     v     q    v    q    v
#      |           |          |         |
#      |           |          |         |
#      h --  u --  h  -- u -- h -- u -- h --
#      |           |          |         | 
#      |           |          |         |
#      v     q     v     q    v    q    |
#      |           |          |         |
#      |           |          |         |
#      h --  u --  h  -- u -- h -- u -- h --
#
#      Because of periodicity all fields are Nx by Ny
#      But we need to define different grids for u,v,h,q
#
# Numerical Method:
# 1) Sadourny's method 1 (energy conserving) and 2 (enstrophy conserving)
# 2) Adams-Bashforth for time stepping
#
# Requires scripts:
#        flux_sw_ener.py  - Sadourny's first method (energy conserving)
#        flux_sw_enst.py  - Sadourny's second method (enstrophy conserving)


from sadourny_setup import flux_sw_ener, Params, np, plt, animation, sys, time, \
                           ener_Euler, ener_AB2, ener_AB3, PLOTTO_649,   \
                           ener_Euler_f, ener_AB2_f, ener_AB3_f, periodic, odd, even


def main(Flux_Euler, Flux_AB2, Flux_AB3, sc=1):
    
    # Grid Parameters
    Lx,Ly  = 200e3, 200e3
    Nx,Ny  = 128*sc, 128*sc
    dx,dy  = Lx/Nx,Ly/Ny
    Nz     = 1

    # Define numerical method,geometry and grid
    #geometry = channel #periodic
    #grid = C
    
    # Physical parameters
    f0 = 1e-4
    gp = 9.81
    H0 = 500

    # Temporal Parameters
    t0, tf, dt  = 0.0, 3600, 5./sc
    Nt  = int((tf-t0)/dt)
    tt  = np.arange(Nt)*dt

    # FJP: if Cgrid vs Agrid
    # FJP: maybe define xs and ys, meshgrid,
    # FJP: then define xx.u, xx.v, xx.y, ...
    # FJP: only used for plotting and maybe forcing
    
    # Define Grid (staggered grid)
    x = np.linspace(-Lx/2+dx/2,Lx/2-dx/2,Nx)
    y = np.linspace(-Ly/2+dy/2,Ly/2-dy/2,Ny)
    xx,yy = np.meshgrid(x,y)
    xu = xx
    xv = xx - dx/2
    xh = xx - dx/2
    yu = yy - dy/2
    yv = yy
    yh = yy

    # Modify class 
    params = Params(dx, dy, f0, gp, H0, Nx, Ny, Nz, dt)

    # Initial Conditions with plot: u, v, h
    hmax = 1.0
    uvh = np.zeros([3,Ny+2,Nx+2])
    uvh[0,1:Ny+1,1:Nx+1] = 0*xu
    uvh[1,1:Ny+1,1:Nx+1] = 0*xv
    uvh[2,1:Ny+1,1:Nx+1] = hmax*np.exp(-(xh**2 + yh**2)/(Lx/6.0)**2)
    # uvh[2,1:Ny+1,1:Nx+1,0] = hmax*np.exp(-((yh-Ly/4)**2)/(Lx/20)**2)

    # Impose BCs
    for jj in range(3):
        uvh[jj, :, :] = periodic(uvh[jj, :, :], params)

    # Define arrays to store conserved quantitites: energy and enstrophy
    energy, enstr = np.zeros(Nt), np.zeros(Nt)
    # UVH = np.empty((3, Ny+2, Nx+2, Nt+1), dtype='d')
    # UVH[:, :, :, 0] = uvh

    # check to see if we're using Fortran
    if Flux_Euler is ener_Euler_f:
        params = np.array([params.dx, params.dy, params.gp, params.f0, params.H0, params.dt])

    # Euler step
    uvh, NLnm, energy[0], enstr[0] = Flux_Euler(uvh, params)
    # UVH[:, :, :, 1] = uvh

    # Impose BCs
    for jj in range(3):
        uvh[jj, :, :] = periodic(uvh[jj, :, :], params)

    # AB2 step
    uvh, NLn, energy[1], enstr[1] = Flux_AB2(uvh, NLnm, params)
    # UVH[:, :, :, 2] = uvh

    # Impose BCs
    for jj in range(3):
        uvh[jj, :, :] = periodic(uvh[jj, :, :], params)

    # step through time
    for ii in range(3, Nt+1):
        # AB3 step
        uvh, NL, energy[ii-1], enstr[ii-1] = Flux_AB3(uvh, NLn, NLnm, params)
        # UVH[:, :, :, ii] = uvh

        # Impose BCs
        for jj in range(3):
            uvh[jj, :, :] = periodic(uvh[jj, :, :], params)

        # Reset fluxes
        NLnm, NLn = NLn, NL
    
    # PLOTTING ==========================================================================
    # if Flux_Euler is ener_Euler_f:
    #     PLOTTO_649(UVH, x, y, Nt, './anims/sw_ener-FORTRAN.mp4')
    # else:
    #     PLOTTO_649(UVH, x, y, Nt, './anims/sw_ener-NUMPY.mp4')    

    print "Error in energy is ", np.amax(energy-energy[0])/energy[0]
    print "Error in enstrophy is ", np.amax(enstr-enstr[0])/enstr[0]

    plt.ioff()
    fig, axarr = plt.subplots(2, sharex=True)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.plot((energy-energy[0]), '-ob', linewidth=2, label='Energy')
    ax1.set_title('Energy')
    ax2.plot((enstr-enstr[0]), '-or', linewidth=2, label='Enstrophy')
    ax2.set_title('Enstrophy')
    plt.show()
    # if Flux_Euler is ener_Euler_f: plt.savefig('./anims/ener-enst-FORTRAN.png')
    # else:                          plt.savefig('./anims/ener-enst-NUMPY.png')

    return uvh[:, 1:-1, 1:-1]


uvh_N = main(ener_Euler, ener_AB2, ener_AB3)
uvh_F = main(ener_Euler_f, ener_AB2_f, ener_AB3_f)