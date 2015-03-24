#!/usr/bin/env python
from sadourny_setup import flux_sw_ener, gather_uvh, Params, set_mpi_bdr,     \
                           create_global_objects, np, plt, animation, sys, time, MPI, comm,  \
                           create_x_points, PLOTTO_649, ener_Euler, ener_AB2, ener_AB3, \
                           ener_Euler_f, ener_AB2_f, ener_AB3_f, ener_Euler_f90, ener_AB2_f90, \
                           ener_AB3_f90, periodic


def main(Flux_Euler, Flux_AB2, Flux_AB3, sc=1):

    # mpi stuff
    p = comm.Get_size()      # number of processors
    rank = comm.Get_rank()   # this process' ID
    uTags = dict([(j, j+5) for j in xrange(p)])
    vTags = dict([(j, j+(p*10) *5) for j in xrange(p)])
    hTags = dict([(j, j+(p*100)*5) for j in xrange(p)])

    # DEFINING SPATIAL, TEMPORAL AND PHYSICAL PARAMETERS ================================
    # Grid Parameters
    Lx, Ly  = 200e3, 200e3
    Nx, Ny  = 128*sc, 128*sc
    dx, dy  = Lx/Nx, Ly/Ny
    Nz = 1
    nx = Nx / p

    # Define numerical method,geometry and grid
    # geometry = channel #periodic
    # grid = C

    # Physical parameters
    f0 = 1e-4
    gp = 9.81
    H0 = 500

    # Temporal Parameters
    t0, tf = 0.0, 3600.0
    dt = 5./sc
    N  = int((tf - t0)/dt)

    # x conditions
    x0, xf = -Lx/2, Lx/2
    dx  = Lx/Nx
    xG  = np.linspace(x0+dx/2, xf-dx/2, Nx)                # global (non-staggered) x points
    xsG = np.linspace(x0 + dx/2, xf-dx/2, Nx)       # global (staggered) x points
    x, xs = create_x_points(rank, p, xG, xsG, nx)  # create this process' x points

    # y conditions
    y0, yf = -Ly/2, Ly/2
    dy = Ly/Ny
    y  = np.linspace(y0 + dy/2, yf - dy/2, Ny)

    # Define Grid (staggered grid)
    xx, yy = np.meshgrid(x, y)
    xu = xx
    xv = xx - dx/2
    xh = xx - dx/2
    yh = yy

    # construct the placeholder for parameters
    params = Params(dx, dy, f0, gp, H0, nx, Ny, Nz, dt)

    # allocate space to communicate the ghost columns
    uCol, vCol, hCol = np.empty(Ny+2, dtype='d'), np.empty(Ny+2, dtype='d'), np.empty(Ny+2, dtype='d')

    # Initial Conditions with plot: u, v, h
    hmax = 1.0
    uvh = np.zeros([3, Ny+2, nx+2])
    uvh[0, 1:Ny+1, 1:nx+1] = 0*xu
    uvh[1, 1:Ny+1, 1:nx+1] = 0*xv
    uvh[2, 1:Ny+1, 1:nx+1] = hmax*np.exp(-(xh**2 + yh**2)/(Lx/6.0)**2)
    # uvh[2,1:Ny+1,1:Nx+1,0] = hmax*np.exp(-((yh-Ly/4)**2)/(Lx/20)**2)

    # Impose BCs
    # for jj in range(3):
    #     uvh[jj, :, :] = periodic(uvh[jj, :, :], params)
    uvh  = set_mpi_bdr(uvh, rank, p, nx, uCol, vCol, hCol, uTags, vTags, hTags)

    # Define arrays to store conserved quantitites: energy and enstrophy
    energy, enstr = np.zeros(N), np.zeros(N)

    # create initial global solution and set of global solutions
    uvhG, UVHG = create_global_objects(rank, xG, y, dx, hmax, Lx, N)

    # check to see if we're using Fortran
    if Flux_Euler is ener_Euler_f or Flux_Euler is ener_Euler_f90:
        params = np.array([params.dx, params.dy, params.gp, params.f0, params.H0, params.dt])

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # BEGIN SOLVING =====================================================================
    # Euler step
    uvh, NLnm, energy[0], enstr[0] = Flux_Euler(uvh, params)
    uvh  = set_mpi_bdr(uvh, rank, p, nx, uCol, vCol, hCol, uTags, vTags, hTags)  # impose BCs on this iteration
    UVHG = gather_uvh(uvh, uvhG, UVHG, rank, p, nx, Ny, 1)      # add uvh to global soln

    # AB2 step
    uvh, NLn, energy[1], enstr[1]  = Flux_AB2(uvh, NLnm, params)
    uvh  = set_mpi_bdr(uvh, rank, p, nx, uCol, vCol, hCol, uTags, vTags, hTags)
    UVHG = gather_uvh(uvh, uvhG, UVHG, rank, p, nx, Ny, 2)
    # if rank == 0:
    #     print UVHG[2, :, :, 2]
    # comm.Barrier()
    # sys.exit()

    # loop through time
    for n in range(3, N):
        # AB3 step
        uvh, NL, energy[n-1], enstr[n-1] = Flux_AB3(uvh, NLn, NLnm, params)

        # Reset fluxes
        NLnm, NLn = NLn, NL

        uvh  = set_mpi_bdr(uvh, rank, p, nx, uCol, vCol, hCol, uTags, vTags, hTags)
        UVHG = gather_uvh(uvh, uvhG, UVHG, rank, p, nx, Ny, n)

    comm.Barrier()
    t_final = (MPI.Wtime() - t_start)  # stop MPI timer

    if rank == 0:
        print t_final

        # PLOTTING ======================================================================
        PLOTTO_649(UVHG, xG, y, N, './anims/sadourny_mpi_%d.mp4' % p, True)

        # print "Error in energy is ", np.amax(energy-energy[0])/energy[0]
        # print "Error in enstrophy is ", np.amax(enstr-enstr[0])/enstr[0]

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
