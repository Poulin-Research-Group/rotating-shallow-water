from __future__ import division
from sadourny_1D_setup import flux_sw_ener, flux_sw_enst, wavenum, np, sys, plt, \
                           time, flux_ener_F

np.seterr(all='warn')


def main(method, sc):

    # Grid Parameters
    # sc  = 8
    Lx  = 200e3
    Nx  = 128*sc
    print Nx
    dx  = Lx/Nx
    print dx

    # Physical parameters
    f0, beta, gp, H0  = 1e-4, 0e-11, 9.81, 500

    # Temporal Parameters
    t0  = 0
    tf  = 3600
    dt  = 5/sc
    Nt  = int(tf/dt)
    tp  = 60
    print dt

    npt = int(tp/dt)
    tt  = np.arange(Nt)*dt

    # Define Grid (staggered grid)
    x  = np.linspace(-Lx/2, Lx/2-dx, Nx)
    xs = np.linspace(-Lx/2+dx/2, Lx/2-dx/2, Nx)

    # Define arrays to store conserved quantitites: energy and enstrophy
    energy, enstr  = np.zeros(Nt), np.zeros(Nt)
    energf, enstrf = np.zeros(Nt), np.zeros(Nt)

    # Modify class
    params = wavenum(dx, f0, beta, gp, H0, Nx)
    if method is flux_ener_F:
        params = np.array([params.dx, params.gp, params.f0, params.H0])

    # Initial Conditions with plot: u, v, h
    hmax = 1.e0
    uvh = np.vstack([0*xs, 0*x, hmax*np.exp(-(x**2)/(Lx/20)**2)])

    # Plot Initial Condition
    # plt.clf
    # plt.plot(x, uvh[2, :])
    # plt.title("h at t = %6.3f hours" % (0))
    # plt.show()

    t_start = time.time()

    NLnm, energy[0], enstr[0] = method(uvh, params)
    uvh  = uvh + dt*NLnm

    # AB2 step
    NLn, energy[1], enstr[1] = method(uvh, params)
    uvh  = uvh + 0.5*dt*(3*NLn - NLnm)

    # AB3 step over the remain timesteps
    for ii in range(3, Nt+1):
        # print ii

        # AB3 step
        NL, energy[ii-1], enstr[ii-1] = method(uvh, params)
        uvh  = uvh + dt/12*(23*NL - 16*NLn + 5*NLnm)

        # Reset fluxes
        NLnm, NLn = NLn, NL

    t_final = time.time()

    return t_final - t_start

if len(sys.argv) > 1:
    methodName, sc = sys.argv[1], int(sys.argv[2])
    if methodName == 'sw_enst':
        method = flux_sw_enst
    elif methodName == 'sw_ener':
        method = flux_sw_ener
    elif methodName == 'fortran':
        method = flux_ener_F
    else:
        raise Exception("Invalid method.")

else:
    method = flux_sw_ener
    sc = 8


print main(method, sc)
