#!/usr/bin/env python
from setup_serial import flux_sw_ener, np, ener_Euler, ener_AB2,     \
                           ener_AB3, ener_Euler_f77,      \
                           ener_AB2_f77, ener_AB3_f77,     \
                           ener_Euler_f90, ener_AB2_f90, ener_AB3_f90,               \
                           ener_Euler_hybrid77, ener_AB2_hybrid77, ener_AB3_hybrid77, \
                           ener_Euler_hybrid90, ener_AB2_hybrid90, ener_AB3_hybrid90,  \
                           METHODS, EULERS, AB2S, AB3S
from sadourny_setup import sys, time
from fjp_helpers.misc import write_time
from fjp_helpers.bc import set_periodic_BC as periodic
from fjp_helpers.animator import mesh_animator


def main(Flux_Euler, Flux_AB2, Flux_AB3, sc=1):

    # Grid Parameters
    Lx, Ly  = 200e3, 200e3
    Nx, Ny  = 128*sc, 128*sc
    dx, dy  = Lx/Nx, Ly/Ny

    # Define numerical method,geometry and grid
    # geometry = channel #periodic
    # grid = C

    # Physical parameters
    f0 = 1e-4
    gp = 9.81
    H0 = 500

    # Temporal Parameters
    t0, tf, dt  = 0.0, 3600, 5./sc
    Nt  = int((tf-t0)/dt)

    # FJP: if Cgrid vs Agrid
    # FJP: maybe define xs and ys, meshgrid,
    # FJP: then define xx.u, xx.v, xx.y, ...
    # FJP: only used for plotting and maybe forcing

    # Define Grid (staggered grid)
    x = np.linspace(-Lx/2+dx/2, Lx/2-dx/2, Nx)
    y = np.linspace(-Ly/2+dy/2, Ly/2-dy/2, Ny)
    xx, yy = np.meshgrid(x, y)
    xu = xx
    xv = xx - dx/2
    xh = xx - dx/2
    yh = yy

    # Modify class
    params = np.array([dx, dy, f0, gp, H0, dt])
    dims   = np.array([Nx, Ny])

    # Initial Conditions with plot: u, v, h
    hmax = 1.0
    uvh = np.zeros([3, Ny+2, Nx+2])
    uvh[0, 1:Ny+1, 1:Nx+1] = 0*xu
    uvh[1, 1:Ny+1, 1:Nx+1] = 0*xv
    uvh[2, 1:Ny+1, 1:Nx+1] = hmax*np.exp(-(xh**2 + yh**2)/(Lx/6.0)**2)
    # uvh[2,1:Ny+1,1:Nx+1,0] = hmax*np.exp(-((yh-Ly/4)**2)/(Lx/20)**2)

    # Impose BCs
    for jj in range(3):
        uvh[jj, :, :] = periodic(uvh[jj, :, :])

    # Define arrays to store conserved quantitites: energy and enstrophy
    energy, enstr = np.zeros(Nt), np.zeros(Nt)
    UVH = np.empty((3, Ny+2, Nx+2, Nt+1), dtype='d')
    UVH[:, :, :, 0] = uvh

    t_start = time.time()

    # Euler step
    uvh, NLnm, energy[0], enstr[0] = Flux_Euler(uvh, params, dims)
    for jj in range(3):
        uvh[jj, :, :] = periodic(uvh[jj, :, :])
    UVH[:, :, :, 1] = uvh

    # AB2 step
    uvh, NLn, energy[1], enstr[1] = Flux_AB2(uvh, NLnm, params, dims)
    for jj in range(3):
        uvh[jj, :, :] = periodic(uvh[jj, :, :])
    UVH[:, :, :, 2] = uvh

    # step through time
    for ii in range(3, Nt+1):
        # AB3 step
        uvh, NL, energy[ii-1], enstr[ii-1] = Flux_AB3(uvh, NLn, NLnm, params, dims)
        for jj in range(3):
            uvh[jj, :, :] = periodic(uvh[jj, :, :])
        UVH[:, :, :, ii] = uvh

        # Reset fluxes
        NLnm, NLn = NLn, NL

    t_final = time.time()

    # PLOTTING ==========================================================================
    # if Flux_Euler is ener_Euler_f:
    #     PLOTTO_649(UVH, x, y, Nt, './anims/sw_ener-FORTRAN.mp4')
    # else:
    # PLOTTO_649(UVH, x, y, Nt, './anims/sw_ener-HYBRID.mp4')
    H = np.empty((Ny*Nx, Nt+1), dtype='d')
    for i in xrange(Nt+1):
        H[:, i] = UVH[2, 1:-1, 1:-1, i].flatten()

    x = np.append(-1, np.append(x, -1))
    y = np.append(-1, np.append(y, -1))

    mesh_animator(H, x, y, Nx, Ny, Nt+1, 1, 1, 1, './anims', 'PLEASE_WORK.mp4')

    """
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
    """
    return t_final - t_start

"""
if len(sys.argv) > 1:
    argv   = sys.argv[1:]
    method = argv[0]
    sc = int(argv[1])
    opt = argv[2] if len(argv) == 3 else None

    if method in METHODS:
        i = METHODS.index(method)
        t = main(EULERS[i], AB2S[i], AB3S[i], sc)
    else:
        raise Exception("Invalid method specified. Pick one of: " + ", ".join(METHODS))

    print t
    writer(t, method, sc, opt)
"""

print main(ener_Euler, ener_AB2, ener_AB3)
# print main(ener_Euler_f77, ener_AB2_f77, ener_AB3_f77)
# print main(ener_Euler_f90, ener_AB2_f90, ener_AB3_f90)
# print main(ener_Euler_hybrid77, ener_AB2_hybrid77, ener_AB3_hybrid77, sc=1)
# print main(ener_Euler_hybrid90, ener_AB2_hybrid90, ener_AB3_hybrid90, sc=1)
