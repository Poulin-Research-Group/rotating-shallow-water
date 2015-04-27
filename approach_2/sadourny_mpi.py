#!/usr/bin/env python
from __future__ import division
from setup_mpi import set_uvh_bdr_1D, create_global_objects, np, \
                      comm, ener_Euler_MPI,    \
                      ener_AB2_MPI, ener_AB3_MPI, solver_1D_helper
from fjp_helpers.mpi import create_tags, create_ranks, send_cols_periodic
from fjp_helpers.animator import mesh_animator
from fjp_helpers.bc import set_periodic_BC_y
from sadourny_setup import Params


def main(Flux_Euler, Flux_AB2, Flux_AB3, px, py, sc=1):

    # currently, this is only testing px = 4, py = 1.
    # if that works, then simple modifications to the script will allow
    # px = 1 and py = 4.
    MPI_Func = send_cols_periodic
    BC_Func  = set_periodic_BC_y

    # mpi stuff
    p = comm.Get_size()      # number of processors
    rank = comm.Get_rank()   # this process' ID
    tags = create_tags(p)[:2]
    ranks = create_ranks(rank, p, px, py)[:3]

    # DEFINING SPATIAL, TEMPORAL AND PHYSICAL PARAMETERS ================================
    # Grid Parameters
    Lx, Ly  = 200e3, 200e3
    Nx, Ny  = 128*sc, 128*sc
    dx, dy  = Lx/Nx, Ly/Ny
    nx = Nx / px
    ny = Ny / py

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
    Nt  = int((tf - t0)/dt)

    # x conditions
    x0, xf = -Lx/2, Lx/2
    dx = Lx/Nx
    xG = np.linspace(x0+dx/2, xf-dx/2, Nx)    # global x points
    x  = np.array_split(xG, p)[rank]          # create this process' x points

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
    params = np.array([dx, dy, f0, gp, H0, dt])
    params = Params()
    params.set_x_vars([x0, xf, dx, Nx, nx])
    params.set_y_vars([y0, yf, dy, Ny, ny])
    params.set_t_vars([t0, tf, dt, Nt])
    params.set_p_vars([p, px, py])
    params.set_consts([f0, gp, H0])
    params.euler, params.ab2, params.ab3 = Flux_Euler, Flux_AB2, Flux_AB3
    params.MPI_Func = MPI_Func
    params.BC_Func  = BC_Func

    # allocate space to communicate the ghost columns
    col = np.empty(Ny+2, dtype='d')

    # Initial Conditions with plot: u, v, h
    hmax = 1.0
    uvh = np.zeros([3, Ny+2, nx+2])
    uvh[0, 1:Ny+1, 1:nx+1] = 0*xu
    uvh[1, 1:Ny+1, 1:nx+1] = 0*xv
    uvh[2, 1:Ny+1, 1:nx+1] = hmax*np.exp(-(xh**2 + yh**2)/(Lx/6.0)**2)
    # uvh[2,1:Ny+1,1:Nx+1,0] = hmax*np.exp(-((yh-Ly/4)**2)/(Lx/20)**2)

    # create initial global solution and set of global solutions
    uvhG, UVHG = create_global_objects(rank, xG, y, Nx, Ny, Nt, dx, hmax, Lx)

    # Impose BCs
    uvh = set_uvh_bdr_1D(uvh, ranks, px, col, tags, MPI_Func, BC_Func)

    # Define arrays to store conserved quantitites: energy and enstrophy
    energy, enstr = np.zeros(Nt), np.zeros(Nt)

    # SOLVING =====================================================================
    t_total, UVHG = solver_1D_helper(uvh, energy, enstr, ranks, col, tags, params)

    if rank == 0:
        print t_total

        # PLOTTING ======================================================================
        H = np.empty((Ny*Nx, Nt), dtype='d')
        for i in xrange(Nt):
            temp = np.array_split(UVHG[:, i], p)
            temp = [np.array_split(part, 3)[2] for part in temp]
            print np.hstack(temp)
            H[:, i] = np.hstack(temp)

        xG = np.append(-1, np.append(xG, -1))
        y = np.append(-1, np.append(y, -1))

        mesh_animator(H, xG, y, nx, ny, Nt, p, px, py, './anims', 'PLEASE_WORK_MPI_4px_1py.mp4')

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

main(ener_Euler_MPI, ener_AB2_MPI, ener_AB3_MPI, 4, 1)
