from __future__ import division
import time
import numpy as np
from sadourny_helpers import I, I_XP, I_XN, I_YP, I_YN, I_XP_YP, I_XP_YN, I_XN_YP, \
                             euler, ab2, ab3, create_global_objects, animate_solution
from flux_ener_f2py77 import euler_f as ener_Euler_f77, \
                             ab2_f as ener_AB2_f77,      \
                             ab3_f as ener_AB3_f77,       \
                             flux_ener as flux_ener_F77
from flux_ener_f2py90 import euler_f as ener_Euler_f90, \
                             ab2_f as ener_AB2_f90,      \
                             ab3_f as ener_AB3_f90,       \
                             flux_ener as flux_ener_F90
from fjp_helpers.bc import set_periodic_BC as periodic


def solver_serial(uvh, params, save_time, animate, save_soln):
    """
    Solves the RSW equations in serial.

    Returns
    -------
    float64
        Total time taken for equation to be solved.
    """
    # update Params object with Euler, AB2 and AB3 functions
    method = params.method
    params.euler = EULERS[METHODS.index(method)]
    params.ab2   = AB2S[METHODS.index(method)]
    params.ab3   = AB3S[METHODS.index(method)]

    if animate:
        t_total, UVHG = solver_serial_helper_g(uvh, params)
        animate_solution(UVHG, 0, params)

    if save_soln:
        t_total = solver_serial_helper_w(uvh, params)

    if save_time:
        t_total = solver_serial_helper(uvh, params)

    return t_total


def solver_serial_helper(uvh, params):
    Flux_Euler, Flux_AB2, Flux_AB3 = params.euler, params.ab2, params.ab3
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    p, px, py = params.p_vars
    method    = params.method
    Nt = params.Nt

    params_obj = params
    if method == 'f2py77' or method == 'f2py90':
        params_obj = params.fortran_vars

    t_start = time.time()

    # Euler step
    uvh, NLnm, energy, enstr = Flux_Euler(uvh, params_obj)
    uvh = periodify(uvh)

    # AB2 step
    uvh, NLn, energy, enstr = Flux_AB2(uvh, NLnm, params_obj)
    uvh = periodify(uvh)

    # loop through time
    for n in range(3, Nt):
        # AB3 step
        uvh, NL, energy, enstr = Flux_AB3(uvh, NLn, NLnm, params_obj)
        uvh = periodify(uvh)

        # Reset fluxes
        NLnm, NLn = NLn, NL

    t_total = (time.time() - t_start)  # stop MPI timer

    return t_total


def solver_serial_helper_g(uvh, params):
    Flux_Euler, Flux_AB2, Flux_AB3 = params.euler, params.ab2, params.ab3
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    p, px, py = params.p_vars
    method    = params.method
    Nt = params.Nt

    params_obj = params
    if method == 'f2py77' or method == 'f2py90':
        params_obj = params.fortran_vars

    UVHG = create_global_objects(0, params)[1]

    t_start = time.time()

    # Euler step
    uvh, NLnm, energy, enstr = Flux_Euler(uvh, params_obj)
    uvh = periodify(uvh)
    UVHG[:, 1] = uvh[:, 1:-1, 1:-1].flatten()

    # AB2 step
    uvh, NLn, energy, enstr = Flux_AB2(uvh, NLnm, params_obj)
    uvh = periodify(uvh)
    UVHG[:, 2] = uvh[:, 1:-1, 1:-1].flatten()

    # loop through time
    for n in range(3, Nt):
        # AB3 step
        uvh, NL, energy, enstr = Flux_AB3(uvh, NLn, NLnm, params_obj)
        uvh = periodify(uvh)

        # Reset fluxes
        NLnm, NLn = NLn, NL

        UVHG[:, n] = uvh[:, 1:-1, 1:-1].flatten()

    t_total = (time.time() - t_start)  # stop MPI timer

    return t_total, UVHG


def flux_sw_ener(uvh, params):

    # Define parameters
    dx, dy     = params.dx, params.dy
    f0, gp, H0 = params.consts
    Nx, Ny     = params.Nx, params.Ny

    # Pull out primitive variables
    u, v, h = uvh[0, :, :], uvh[1, :, :],  H0 + uvh[2, :, :]

    # Initialize fields
    U, V = np.zeros((Ny+2, Nx+2)), np.zeros((Ny+2, Nx+2))
    B, q = np.zeros((Ny+2, Nx+2)), np.zeros((Ny+2, Nx+2))
    flux = np.zeros((3, Ny, Nx))

    # Compute U, V, B, q
    U[1:-1, 1:-1] = 0.5*(I(h) + I_XP(h)) * I(u)
    V[1:-1, 1:-1] = 0.5*(I(h) + I_YP(h)) * I(v)
    B[1:-1, 1:-1] = gp*I(h) + 0.25*(I(u)**2 + I_XN(u)**2 + I(v)**2 + I_YN(v)**2)
    q[1:-1, 1:-1] = 4*((I_XP(v) - I(v)) / dx - (I_YP(u) - I(u)) / dy + f0) /   \
                       (I(h) + I_YP(h) + I_XP(h) + I_XP_YP(h))

    # Enforce BCs
    U = periodic(U)
    V = periodic(V)
    B = periodic(B)
    q = periodic(q)
    # U = even(U)
    # V = odd(V)
    # B = even(B)
    # q = even(q)

    # Compute fluxes
    flux[0, :, :] =  0.25*(I(q) * (I_XP(V) + I(V)) + I_YN(q) * (I_XP_YN(V) + I_YN(V))) - \
                    (I_XP(B) - I(B))/dx
    flux[1, :, :] = -0.25*(I(q) * (I_YP(U) + I(U)) + I_XN(q) * (I_XN_YP(U) + I_XN(U))) - \
                    (I_YP(B) - I(B))/dy
    flux[2, :, :] = -(U[1:-1, 1:-1] - U[1:-1, 0:-2])/dx - (V[1:-1, 1:-1] - V[0:-2, 1:-1])/dy

    # compute energy and enstrophy
    energy = 0.5*np.mean(gp*I(h)**2 + 0.5*I(h)*(I(u)**2 + I_XN(u)**2 + I(v)**2 + I_YN(v)**2))
    enstrophy = 0.125*np.mean((I(h) + I_YP(h) + I_XP(h) + I_XP_YP(h)) * I(q)**2)

    return flux, energy, enstrophy


def ener_Euler(uvh, params):
    # pure Numpy
    NLnm, energy, enstr = flux_sw_ener(uvh, params)
    uvh[:, 1:-1, 1:-1]  = euler(uvh, params.dt, NLnm)
    return uvh, NLnm, energy, enstr


def ener_Euler_hybrid77(uvh, params):
    # calculating flux in Fortran 77, updating solution in Numpy
    NLnm, energy, enstr = flux_ener_F77(uvh, params.fortran_vars)
    uvh[:, 1:-1, 1:-1]  = euler(uvh, params.dt, NLnm)
    return uvh, NLnm, energy, enstr


def ener_Euler_hybrid90(uvh, params):
    # calculating flux in Fortran 90, updating solution in Numpy
    NLnm, energy, enstr = flux_ener_F90(uvh, params.fortran_vars)
    uvh[:, 1:-1, 1:-1]  = euler(uvh, params.dt, NLnm)
    return uvh, NLnm, energy, enstr


def ener_AB2(uvh, NLnm, params):
    NLn, energy, enstr = flux_sw_ener(uvh, params)
    uvh[:, 1:-1, 1:-1] = ab2(uvh, params.dt, NLn, NLnm)
    return uvh, NLn, energy, enstr


def ener_AB2_hybrid77(uvh, NLnm, params):
    NLn, energy, enstr = flux_ener_F77(uvh, params.fortran_vars)
    uvh[:, 1:-1, 1:-1] = ab2(uvh, params.dt, NLn, NLnm)
    return uvh, NLn, energy, enstr


def ener_AB2_hybrid90(uvh, NLnm, params):
    NLn, energy, enstr = flux_ener_F90(uvh, params.fortran_vars)
    uvh[:, 1:-1, 1:-1] = ab2(uvh, params.dt, NLn, NLnm)
    return uvh, NLn, energy, enstr


def ener_AB3(uvh, NLn, NLnm, params):
    NL, energy, enstr  = flux_sw_ener(uvh, params)
    uvh[:, 1:-1, 1:-1] = ab3(uvh, params.dt, NL, NLn, NLnm)
    return uvh, NL, energy, enstr


def ener_AB3_hybrid77(uvh, NLn, NLnm, params):
    NL, energy, enstr  = flux_ener_F77(uvh, params.fortran_vars)
    uvh[:, 1:-1, 1:-1] = ab3(uvh, params.dt, NL, NLn, NLnm)
    return uvh, NL, energy, enstr


def ener_AB3_hybrid90(uvh, NLn, NLnm, params):
    NL, energy, enstr  = flux_ener_F90(uvh, params.fortran_vars)
    uvh[:, 1:-1, 1:-1] = ab3(uvh, params.dt, NL, NLn, NLnm)
    return uvh, NL, energy, enstr


def periodify(uvh):
    # sets periodic BCs for each part of uvh
    for i in xrange(3):
        uvh[i, :, :] = periodic(uvh[i, :, :])
    return uvh


METHODS = ['numpy', 'f2py77', 'f2py90', 'hybrid77', 'hybrid90']
EULERS  = [ener_Euler, ener_Euler_f77, ener_Euler_f90, ener_Euler_hybrid77, ener_Euler_hybrid90]
AB2S    = [ener_AB2, ener_AB2_f77, ener_AB2_f90, ener_AB2_hybrid77, ener_AB2_hybrid90]
AB3S    = [ener_AB3, ener_AB3_f77, ener_AB3_f90, ener_AB3_hybrid77, ener_AB3_hybrid90]
