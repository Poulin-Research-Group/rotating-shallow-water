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
#      Because of periodicity all fields are Mx by My
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
from sadourny_setup import flux_sw_ener, flux_sw_enst, gather_uvh, wavenum, set_mpi_bdr, \
                           create_global_objects, np, plt, animation, sys, time, MPI, comm, \
                           create_x_points, PLOTTO_649


def main():
    # mpi stuff
    rank = comm.Get_rank()   # this process' ID
    p = comm.Get_size()      # number of processors
    tags = dict([(j, j+5) for j in xrange(p)])

    # Number of grid points
    sc = 1
    Mx, My  = 128*sc, 128*sc
    mx = Mx / p  # x grid points per process

    # Grid Parameters
    Lx, Ly  = 200e3, 200e3

    # DEFINING SPATIAL, TEMPORAL AND PHYSICAL PARAMETERS ================================
    # x conditions
    dx = Lx/Mx
    x0, xf = -Lx/2, Lx/2

    # create global x points
    xG  = np.linspace(x0, xf-dx, Mx)            # all (non-staggered) x points
    xsG = np.linspace(x0 + dx/2, xf-dx/2, Mx)    # all (staggered) x points
    # create this process' x points
    x, xs  = create_x_points(rank, p, xG, xsG, mx)

    # y conditions
    dy = Ly/My
    y0, yf = -Ly/2, Ly/2
    y  = np.linspace(y0, yf-dy, My)
    ys = np.linspace(y0+dy/2, yf-dy/2, My)

    # Physical parameters
    f0, beta, gp, H0  = 1.e-4, 0e-11, 9.81, 500.

    # Temporal Parameters
    dt = 5./sc
    t0, tf = 0.0, 3600.0
    N  = int((tf - t0)/dt)

    # Define Grid (staggered grid)
    xq, yq = np.meshgrid(xs, ys)
    xh, yh = np.meshgrid(x,  y)
    xu, yu = np.meshgrid(xs, y)
    xv, yv = np.meshgrid(x,  ys)

    # construct the wave number
    params = wavenum(dx, dy, f0, beta, gp, H0, mx, My)

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
    col = np.empty(3*My, dtype='d')

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # BEGIN SOLVING =====================================================================
    method = flux_sw_enst

    # Euler step
    NLnm, energy[0], enstr[0] = method(uvh, params)
    uvh  = uvh + dt*NLnm
    # impose BCs on this iteration
    uvh  = set_mpi_bdr(uvh, rank, p, mx, col, tags)
    UVHG = gather_uvh(uvh, uvhG, UVHG, rank, p, mx, My, 1)      # add uvh to global soln

    # AB2 step
    NLn, energy[1], enstr[1] = method(uvh, params)
    uvh  = uvh + 0.5*dt*(3*NLn - NLnm)
    # impose BCs
    uvh  = set_mpi_bdr(uvh, rank, p, mx, col, tags)
    UVHG = gather_uvh(uvh, uvhG, UVHG, rank, p, mx, My, 2)      # add to global soln

    # loop through time
    for n in range(3, N):
        # AB3 step
        NL, energy[n-1], enstr[n-1] = method(uvh, params)
        uvh = uvh + dt/12*(23*NL - 16*NLn + 5*NLnm)

        # Reset fluxes
        NLnm, NLn = NLn, NL

        # impose BCs
        uvh  = set_mpi_bdr(uvh, rank, p, mx, col, tags)
        UVHG = gather_uvh(uvh, uvhG, UVHG, rank, p, mx, My, n)

    comm.Barrier()
    t_final = (MPI.Wtime() - t_start)  # stop MPI timer

    print t_final

    # PLOTTING ==========================================================================
    if rank == 0:
        PLOTTO_649(UVHG, xG, y, My, N, './anims/sadourny_mpi_%d.mp4' % p)

    """
    print "Error in energy is ", np.amax(energy-energy[0])/energy[0]
    print "Error in enstrophy is ", np.amax(enstr-enstr[0])/enstr[0]

    fig, axarr = plt.subplots(2, sharex=True)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.plot((energy-energy[0])/energy[0], '-ob', linewidth=2, label='Energy')
    ax1.set_title('Energy')
    ax2.plot((enstr-enstr[0])/enstr[0], '-or',  linewidth=2, label='Enstrophy')
    ax2.set_title('Enstrophy')
    plt.show()
    """

main()
