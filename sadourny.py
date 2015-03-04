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
#   u_t =  (q*V^x)^y + d_x h
#   v_t = -(q*U^y)^x + d_y h
#   h_t = - div[U,V]
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


from sadourny_setup import flux_sw_ener, flux_sw_enst, wavenum, np, plt, animation, sys, \
                           time, PLOTTO_649, flux_sw_ener_F


def main(method, sc):

    # Number of grid points
    # sc  = 1
    Nx, Ny  = 128*sc, 128*sc

    # DEFINING SPATIAL, TEMPORAL AND PHYSICAL PARAMETERS ================================
    # Grid Parameters
    Lx, Ly  = 200e3, 200e3

    # x conditions
    dx = Lx/Nx
    x0, xf = -Lx/2, Lx/2
    x  = np.linspace(x0, xf-dx, Nx)
    xs = np.linspace(x0+dx/2, xf-dx/2, Nx)

    # y conditions
    dy = Ly/Ny
    y0, yf = -Ly/2, Ly/2
    y  = np.linspace(y0, yf-dy, Ny)
    ys = np.linspace(y0+dy/2, yf-dy/2, Ny)

    # Physical parameters
    f0, beta, gp, H0  = 1.e-4, 0e-11, 9.81, 500.

    # Temporal Parameters
    dt = 5./sc
    t0, tf = 0.0, 3600.0
    N  = int((tf - t0)/dt)

    # method = flux_sw_enst

    # Define Grid (staggered grid)
    xq, yq = np.meshgrid(xs, ys)
    xh, yh = np.meshgrid(x,  y)
    xu, yu = np.meshgrid(xs, y)
    xv, yv = np.meshgrid(x,  ys)

    # Modify class
    params = wavenum(dx, dy, f0, beta, gp, H0, Nx, Ny)
    # if method is flux_ener_F:
    #     params = np.array([params.dx, params.gp, params.f0, params.H0])

    # Initial Conditions with plot: u, v, h
    hmax = 1.e0
    uvh = np.vstack([0.*xu,
                     0.*xv,
                     hmax*np.exp(-(xh**2 + (1.0*yh)**2)/(Lx/6.0)**2)])

    # Define arrays to store conserved quantitites: energy and enstrophy
    energy, enstr = np.zeros(N), np.zeros(N)

    UVH = np.empty((3*Ny, Nx, N), dtype='d')
    UVH[:, :, 0] = uvh

    # BEGIN SOLVING =====================================================================
    t_start = time.time()

    # Euler step
    NLnm, energy[0], enstr[0] = method(uvh, params)
    uvh  = uvh + dt*NLnm
    UVH[:, :, 1] = uvh

    # AB2 step
    NLn, energy[1], enstr[1] = method(uvh, params)
    uvh  = uvh + 0.5*dt*(3*NLn - NLnm)
    UVH[:, :, 2] = uvh

    # loop through time
    for n in range(3, N):
        # AB3 step
        NL, energy[n-1], enstr[n-1] = method(uvh, params)
        uvh = uvh + dt/12*(23*NL - 16*NLn + 5*NLnm)
        UVH[:, :, n] = uvh

        # Reset fluxes
        NLnm, NLn = NLn, NL

    t_final = time.time()
    print t_final - t_start

    # PLOTTING ==========================================================================
    # PLOTTO_649(UVH, x, y, Ny, N, './anims/sadourny.mp4')

    print "Error in energy is ", np.amax(energy-energy[0])/energy[0]
    print "Error in enstrophy is ", np.amax(enstr-enstr[0])/enstr[0]

    fig, axarr = plt.subplots(2, sharex=True)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.plot((energy-energy[0])/energy[0], '-ob', linewidth=2, label='Energy')
    ax1.set_title('Energy')
    ax2.plot((enstr-enstr[0])/enstr[0], '-or',  linewidth=2, label='Enstrophy')
    ax2.set_title('Enstrophy')
    plt.savefig('./anims/ener-enst-FORTRAN.png')

main(flux_sw_ener_F, 1)