from __future__ import division
from sadourny_1D_setup import flux_sw_ener, flux_sw_enst, wavenum, np, sys, plt, \
                              time, ener_Euler, ener_AB2, ener_AB3, ener_Euler_f, \
                              ener_AB2_f, ener_AB3_f

np.seterr(all='warn')


def main(Flux_Euler, Flux_AB2, Flux_AB3, sc):

    # DEFINING SPATIAL, TEMPORAL AND PHYSICAL PARAMETERS ================================
    # Grid Parameters
    Lx  = 200e3
    Nx  = 128*sc
    dx  = Lx/Nx

    # Physical parameters
    f0, beta, gp, H0  = 1e-4, 0e-11, 9.81, 500

    # Temporal Parameters
    t0  = 0
    tf  = 3600
    dt  = 5/sc
    Nt  = int((tf-t0)/dt)

    # Define Grid (staggered grid)
    x  = np.linspace(-Lx/2, Lx/2-dx, Nx)
    xs = np.linspace(-Lx/2+dx/2, Lx/2-dx/2, Nx)

    # Define arrays to store conserved quantitites: energy and enstrophy
    energy, enstr  = np.zeros(Nt), np.zeros(Nt)

    # Modify class
    params = wavenum(dx, f0, beta, gp, H0, Nx, dt)
    if Flux_Euler is ener_Euler_f:
        params = np.array([params.dx, params.gp, params.f0, params.H0, params.dt])

    # Initial Conditions with plot: u, v, h
    hmax = 1.e0
    uvh = np.vstack([0*xs, 0*x, hmax*np.exp(-(x**2)/(Lx/20)**2)])

    # BEGIN SOLVING =====================================================================
    t_start = time.time()

    # Euler step
    uvh, NLnm, energy[0], enstr[0] = Flux_Euler(uvh, params)

    # AB2 step
    uvh, NLn, energy[1], enstr[1] = Flux_AB2(uvh, NLnm, params)

    # AB3 step over the remain timesteps
    for ii in range(3, Nt+1):
        # AB3 step
        uvh, NL, energy[ii-1], enstr[ii-1] = Flux_AB3(uvh, NLn, NLnm, params)

        # Reset fluxes
        NLnm, NLn = NLn, NL

    t_final = time.time()
    t_total = t_final - t_start

    print t_total
    return t_total


main(method, sc)
