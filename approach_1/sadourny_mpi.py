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


# Import libraries
from sadourny_setup import flux_sw_ener, flux_sw_enst, gather_uvh, Params, set_mpi_bdr, \
                           create_global_objects, np, plt, animation, sys, time, MPI, comm, \
                           create_x_points, PLOTTO_649, Inds, ener_Euler, ener_AB2, ener_AB3, \
                           ener_Euler_f, ener_AB2_f, ener_AB3_f


def main(Flux_Euler, Flux_AB2, Flux_AB3, sc=1):
    # mpi stuff
    rank = comm.Get_rank()   # this process' ID
    p = comm.Get_size()      # number of processors
    tags = dict([(j, j+5) for j in xrange(p)])

    # Number of grid points
    Nx, Ny  = 128*sc, 128*sc
    nx = Nx / p  # x grid points per process

    # DEFINING SPATIAL, TEMPORAL AND PHYSICAL PARAMETERS ================================
    # Grid Parameters
    Lx, Ly  = 200e3, 200e3

    # x conditions
    x0, xf = -Lx/2, Lx/2
    dx  = Lx/Nx
    xG  = np.linspace(x0, xf-dx, Nx)             # global (non-staggered) x points
    xsG = np.linspace(x0 + dx/2, xf-dx/2, Nx)    # global (staggered) x points
    # create this process' x points
    x, xs  = create_x_points(rank, p, xG, xsG, nx)

    # y conditions
    y0, yf = -Ly/2, Ly/2
    dy = Ly/Ny
    y  = np.linspace(y0, yf-dy, Ny)
    ys = np.linspace(y0+dy/2, yf-dy/2, Ny)

    # Physical parameters
    f0, beta, gp, H0  = 1.e-4, 0e-11, 9.81, 500.

    # Temporal Parameters
    t0, tf = 0.0, 3600.0
    dt = 5./sc
    N  = int((tf - t0)/dt)

    # Define Grid (staggered grid)
    xq, yq = np.meshgrid(xs, ys)
    xh, yh = np.meshgrid(x,  y)
    xu, yu = np.meshgrid(xs, y)
    xv, yv = np.meshgrid(x,  ys)

    # construct the placeholder for parameters
    params = Params(dx, dy, f0, beta, gp, H0, nx, Ny, dt)
    inds   = Inds(Ny)
    if Flux_Euler is ener_Euler_f:
        params = np.array([params.dx, params.dy, params.gp, params.f0, params.H0, params.dt])
        inds   = np.array([inds.Iv_i, inds.Iv_f, inds.Ih_i, inds.Ih_f])

    # Initial Conditions with plot: u, v, h
    hmax = 1.e0
    uvh = np.vstack([0.*xu,
                     0.*xv,
                     hmax*np.exp(-(xh**2 + (1.0*yh)**2)/(Lx/6.0)**2)])

    # Define arrays to store conserved quantitites: energy and enstrophy
    energy, enstr = np.zeros(N), np.zeros(N)

    # create initial global solution and set of global solutions
    uvhG, UVHG = create_global_objects(rank, xG, y, xsG, ys, hmax, Lx, N)

    # allocate space to communicate the ghost columns
    col = np.empty(3*Ny, dtype='d')

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # BEGIN SOLVING =====================================================================
    # Euler step
    uvh, NLnm, energy[0], enstr[0] = Flux_Euler(uvh, params, inds)
    uvh  = set_mpi_bdr(uvh, rank, p, nx, col, tags)             # impose BCs on this iteration
    UVHG = gather_uvh(uvh, uvhG, UVHG, rank, p, nx, Ny, 1)      # add uvh to global soln

    # AB2 step
    uvh, NLn, energy[1], enstr[1]  = Flux_AB2(uvh, NLnm, params, inds)
    uvh  = set_mpi_bdr(uvh, rank, p, nx, col, tags)
    UVHG = gather_uvh(uvh, uvhG, UVHG, rank, p, nx, Ny, 2)

    # loop through time
    for n in range(3, N):
        # AB3 step
        uvh, NL, energy[n-1], enstr[n-1] = Flux_AB3(uvh, NLn, NLnm, params, inds)

        # Reset fluxes
        NLnm, NLn = NLn, NL

        uvh  = set_mpi_bdr(uvh, rank, p, nx, col, tags)
        UVHG = gather_uvh(uvh, uvhG, UVHG, rank, p, nx, Ny, n)

    comm.Barrier()
    t_final = (MPI.Wtime() - t_start)  # stop MPI timer

    print t_final

    # PLOTTING ==========================================================================
    if rank == 0:
        # PLOTTO_649(UVHG, xG, y, Ny, N, './anims/sadourny_mpi_%d.mp4' % p)

        print "Error in energy is ", np.amax(energy-energy[0])/energy[0]
        print "Error in enstrophy is ", np.amax(enstr-enstr[0])/enstr[0]

        """
        fig, axarr = plt.subplots(2, sharex=True)
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        ax1.plot((energy-energy[0])/energy[0], '-ob', linewidth=2, label='Energy')
        ax1.set_title('Energy')
        ax2.plot((enstr-enstr[0])/enstr[0], '-or',  linewidth=2, label='Enstrophy')
        ax2.set_title('Enstrophy')
        plt.show()
        """

main(ener_Euler, ener_AB2, ener_AB3, 1)
