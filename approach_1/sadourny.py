#!/usr/bin/env python
from sadourny_setup import np, sys, time, PLOTTO_649, writer, ener_Euler,       \
                           ener_AB2, ener_AB3, ener_Euler_f77, ener_AB2_f77,     \
                           ener_AB3_f77, ener_Euler_hybrid77, ener_AB2_hybrid77,  \
                           ener_AB3_hybrid77


def main(Flux_Euler, Flux_AB2, Flux_AB3, sc):

    # Number of grid points
    Nx, Ny  = 128*sc, 128*sc

    # DEFINING SPATIAL, TEMPORAL AND PHYSICAL PARAMETERS ================================
    # Grid Parameters
    Lx, Ly  = 200e3, 200e3

    # x conditions
    x0, xf = -Lx/2, Lx/2
    dx = Lx/Nx
    x  = np.linspace(x0, xf-dx, Nx)
    xs = np.linspace(x0+dx/2, xf-dx/2, Nx)

    # y conditions
    y0, yf = -Ly/2, Ly/2
    dy = Ly/Ny
    y  = np.linspace(y0, yf-dy, Ny)
    ys = np.linspace(y0+dy/2, yf-dy/2, Ny)

    # Physical parameters
    f0, gp, H0  = 1.e-4, 9.81, 500.

    # Temporal Parameters
    t0, tf = 0.0, 3600.0
    dt = 5./sc
    Nt  = int((tf - t0)/dt)

    # Define Grid (staggered grid)
    xq, yq = np.meshgrid(xs, ys)
    xh, yh = np.meshgrid(x,  y)
    xu, yu = np.meshgrid(xs, y)
    xv, yv = np.meshgrid(x,  ys)

    # Create parameters and indices
    params = np.array([dx, dy, f0, gp, H0, dt])
    inds   = np.array([0, Ny, Ny, 2*Ny, 2*Ny, 3*Ny])

    # check to see if we're using Fortran
    # if Flux_Euler is not ener_Euler:
    #     # Fortran 77?
    #     if Flux_Euler is ener_Euler_f:
    #         inds = np.array([inds.Iv_i, inds.Iv_f, inds.Ih_i])
    #     # or Fortran 90?
    #     else:
    #         inds = np.array([inds.Iv_i + 1, inds.Iv_f, inds.Ih_i + 1])

    # TODO: write Fortran 90 code ------------

    # Initial Conditions with plot: u, v, h
    hmax = 1.e0
    uvh = np.vstack([0.*xu,
                     0.*xv,
                     hmax*np.exp(-(xh**2 + (1.0*yh)**2)/(Lx/6.0)**2)])

    # Define arrays to store conserved quantitites: energy and enstrophy
    energy, enstr = np.zeros(Nt), np.zeros(Nt)
    UVH = np.empty((3*Ny, Nx, Nt+1), dtype='d')
    UVH[:, :, 0] = uvh

    # BEGIN SOLVING =====================================================================
    t_start = time.time()

    # Euler step
    uvh, NLnm, energy[0], enstr[0] = Flux_Euler(uvh, params, inds)
    UVH[:, :, 1] = uvh

    # AB2 step
    uvh, NLn, energy[1], enstr[1] = Flux_AB2(uvh, NLnm, params, inds)
    UVH[:, :, 2] = uvh

    # loop through time
    for n in range(3, Nt+1):
        # AB3 step
        uvh, NL, energy[n-1], enstr[n-1] = Flux_AB3(uvh, NLn, NLnm, params, inds)
        UVH[:, :, n] = uvh

        # Reset fluxes
        NLnm, NLn = NLn, NL

    t_final = time.time()
    t_total = t_final - t_start

    # PLOTTING ==========================================================================
    # PLOTTO_649(UVH, x, y, Ny, Nt, './anims/sadourny-flux_ener_sw.mp4')

    print "Error in energy is ", np.amax(energy-energy[0])/energy[0]
    print "Error in enstrophy is ", np.amax(enstr-enstr[0])/enstr[0]

    """
    fig, axarr = plt.subplots(2, sharex=False)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.plot((energy[:-1]-energy[0])/energy[0], '-ob', linewidth=2, label='Energy')
    ax1.set_title('Energy')
    ax2.plot((enstr[:-1]-enstr[0])/enstr[0], '-or',  linewidth=2, label='Enstrophy')
    ax2.set_title('Enstrophy')
    plt.show()
    """

    print t_total
    # return t_total


if len(sys.argv) > 1:
    argv = sys.argv[1:]
    method = argv[0]     # either Numpy, F77, F90
    sc = int(argv[1])

    if method == 'numpy':
        t = main(ener_Euler, ener_AB2, ener_AB3, sc)
    elif method == 'f77':
        t = main(ener_Euler_f77, ener_AB2_f77, ener_AB3_f77, sc)
    # elif method == 'f90':
    #     t = main(ener_Euler_f90, ener_AB2_f90, ener_AB3_f90, sc)
    elif method == 'hybrid77':
        t = main(ener_Euler_hybrid77, ener_AB2_hybrid77, ener_AB3_hybrid77)
    # elif method == 'hybrid90':
    #     t = main(ener_Euler_hybrid90, ener_AB2_hybrid90, ener_AB3_hybrid90)
    else:
        raise Exception("Invalid method specified.")

    print t
    writer(t, method, sc)


main(ener_Euler, ener_AB2, ener_AB3, 1)
main(ener_Euler_f77, ener_AB2_f77, ener_AB3_f77, 1)
main(ener_Euler_hybrid77, ener_AB2_hybrid77, ener_AB3_hybrid77, 1)
