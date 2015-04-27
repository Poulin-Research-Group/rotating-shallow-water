from __future__ import division
import numpy as np
from mpi4py import MPI
from sadourny_setup import I, I_XP, I_XN, I_YP, I_YN, I_XP_YP, I_XP_YN, I_XN_YP, \
                           euler, ab2, ab3
from flux_ener_f2py77 import euler_f as ener_Euler_f77, \
                             ab2_f as ener_AB2_f77,      \
                             ab3_f as ener_AB3_f77,       \
                             flux_ener as flux_ener_F77
from flux_ener_f2py90 import euler_f as ener_Euler_f90, \
                             ab2_f as ener_AB2_f90,      \
                             ab3_f as ener_AB3_f90,       \
                             flux_ener as flux_ener_F90
comm = MPI.COMM_WORLD
np.seterr(all='raise')


def solver_MPI():
    pass


def solver_1D_helper(uvh, energy, enstr, ranks, ghost_arr, tags, params):
    Flux_Euler, Flux_AB2, Flux_AB3 = params.euler, params.ab2, params.ab3
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    p, px, py = params.p_vars
    MPI_Func  = params.MPI_Func
    BC_Func   = params.BC_Func
    Nt = params.Nt

    rank = ranks[0]

    Lx = 200e3
    hmax = 1.0

    xG = np.linspace(x0+dx/2, xf-dx/2, Nx)    # global x points
    y  = np.linspace(y0 + dy/2, yf - dy/2, Ny)

    uvhG, UVHG = create_global_objects(rank, xG, y, Nx, Ny, Nt, dx, hmax, Lx)

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # Euler step
    uvh, NLnm, energy[0], enstr[0] = Flux_Euler(uvh, ranks, px, ghost_arr, tags, params)
    uvh = set_uvh_bdr_1D(uvh, ranks, px, ghost_arr, tags, MPI_Func, BC_Func)
    comm.Gather(uvh[:, 1:-1, 1:-1].flatten(), uvhG, root=0)
    if rank == 0:
        UVHG[:, 1] = uvhG

    # AB2 step
    uvh, NLn, energy[1], enstr[1]  = Flux_AB2(uvh, NLnm, ranks, px, ghost_arr, tags, params)
    uvh = set_uvh_bdr_1D(uvh, ranks, px, ghost_arr, tags, MPI_Func, BC_Func)
    comm.Gather(uvh[:, 1:-1, 1:-1].flatten(), uvhG, root=0)
    if rank == 0:
        UVHG[:, 2] = uvhG

    # loop through time
    for n in range(3, Nt):
        # AB3 step
        uvh, NL, energy[n-1], enstr[n-1] = Flux_AB3(uvh, NLn, NLnm, ranks, px, ghost_arr, tags, params)

        # Reset fluxes
        NLnm, NLn = NLn, NL

        uvh  = set_uvh_bdr_1D(uvh, ranks, px, ghost_arr, tags, MPI_Func, BC_Func)
        comm.Gather(uvh[:, 1:-1, 1:-1].flatten(), uvhG, root=0)
        if rank == 0:
            UVHG[:, n] = uvhG

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer

    return t_total, UVHG


def flux_sw_ener_MPI_1D(uvh, ranks, px, ghost_arr, tags, params):
    # All terms (h, U, V, B, etc...) are calculated in numpy.
    rank, rankLU, rankRD = ranks
    tagsLU, tagsRD = tags

    # define what kind of MPI function we're using (i.e. sending rows or cols)
    MPI_Func = params.MPI_Func

    # define what kind of BC function we're using (i.e. rows or cols)
    BC_Func  = params.BC_Func

    # Define parameters
    dx, dy     = params.dx, params.dy
    f0, gp, H0 = params.consts
    Nx, Ny     = params.nx, params.ny

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

    # Enforce BCs using MPI
    U = BC_Func(MPI_Func(U, rank, px, ghost_arr, tagsLU, tagsRD, rankLU, rankRD))
    V = BC_Func(MPI_Func(V, rank, px, ghost_arr, tagsLU, tagsRD, rankLU, rankRD))
    B = BC_Func(MPI_Func(B, rank, px, ghost_arr, tagsLU, tagsRD, rankLU, rankRD))
    q = BC_Func(MPI_Func(q, rank, px, ghost_arr, tagsLU, tagsRD, rankLU, rankRD))

    # Compute fluxes
    flux[0, :, :] =  0.25*(I(q) * (I_XP(V) + I(V)) + I_YN(q) * (I_XP_YN(V) + I_YN(V))) - \
                    (I_XP(B) - I(B))/dx
    flux[1, :, :] = -0.25*(I(q) * (I_YP(U) + I(U)) + I_XN(q) * (I_XN_YP(U) + I_XN(U))) - \
                    (I_YP(B) - I(B))/dy
    flux[2, :, :] = -(U[1:-1, 1:-1] - U[1:-1, 0:-2])/dx - (V[1:-1, 1:-1] - V[0:-2, 1:-1])/dy

    # compute energy and enstrophy
    # energy = 0.5*np.mean(gp*I(h)**2 + 0.5*I(h)*(I(u)**2 + I_XN(u)**2 + I(v)**2 + I_YN(v)**2))
    # enstrophy = 0.125*np.mean((I(h) + I_YP(h) + I_XP(h) + I_XP_YP(h)) * I(q)**2)
    energy, enstrophy = 0, 0

    return flux, energy, enstrophy

"""
def flux_sw_ener_f90_MPI(uvh, params, dims, rank, p, col, tags):
    # All terms (h, U, V, B, etc...) are calculated in F90.

    # Define parameters
    dx, dy     = params[0], params[1]
    f0, gp, H0 = params[2], params[3], params[4]
    Nx, Ny     = dims

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

    # Enforce BCs using MPI
    U = set_mpi_bc(U, rank, p, col, tags)
    V = set_mpi_bc(V, rank, p, col, tags)
    B = set_mpi_bc(B, rank, p, col, tags)
    q = set_mpi_bc(q, rank, p, col, tags)
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
"""


def ener_Euler_MPI(uvh, ranks, px, ghost_arr, tags, params):
    # MPI'd pure Numpy
    NLnm, energy, enstr = flux_sw_ener_MPI_1D(uvh, ranks, px, ghost_arr, tags, params)
    uvh[:, 1:-1, 1:-1]  = euler(uvh, params.dt, NLnm)
    return uvh, NLnm, energy, enstr


def ener_AB2_MPI(uvh, NLnm, ranks, px, ghost_arr, tags, params):
    NLn, energy, enstr = flux_sw_ener_MPI_1D(uvh, ranks, px, ghost_arr, tags, params)
    uvh[:, 1:-1, 1:-1] = ab2(uvh, params.dt, NLn, NLnm)
    return uvh, NLn, energy, enstr


def ener_AB3_MPI(uvh, NLn, NLnm, ranks, px, ghost_arr, tags, params):
    NL, energy, enstr  = flux_sw_ener_MPI_1D(uvh, ranks, px, ghost_arr, tags, params)
    uvh[:, 1:-1, 1:-1] = ab3(uvh, params.dt, NL, NLn, NLnm)
    return uvh, NL, energy, enstr


def set_uvh_bdr_1D(uvh, ranks, px, ghost_arr, tags, MPI_Func, BC_Func):
    rank, rankLU, rankRD = ranks
    tagsLU, tagsRD = tags

    u, v, h = uvh[0, :, :], uvh[1, :, :], uvh[2, :, :]

    u = BC_Func(MPI_Func(u, rank, px, ghost_arr, tagsLU, tagsRD, rankLU, rankRD))
    v = BC_Func(MPI_Func(v, rank, px, ghost_arr, tagsLU, tagsRD, rankLU, rankRD))
    h = BC_Func(MPI_Func(h, rank, px, ghost_arr, tagsLU, tagsRD, rankLU, rankRD))

    uvh[0, :, :], uvh[1, :, :], uvh[2, :, :] = u, v, h

    return uvh


def create_global_objects(rank, xG, yG, Nx, Ny, Nt, dx, hmax, Lx):
    if rank == 0:
        xx, yy = np.meshgrid(xG, yG)
        xuG = xx
        xvG = xx - dx/2
        xhG = xx - dx/2
        yhG = yy

        uvhG = np.zeros([3, Ny, Nx])                # global initial solution
        UVHG = np.empty((3*Ny*Nx, Nt), dtype='d')  # set of ALL global solutions

        uvhG[0, :, :] = 0*xuG
        uvhG[1, :, :] = 0*xvG
        uvhG[2, :, :] = hmax*np.exp(-(xhG**2 + yhG**2)/(Lx/6.0)**2)
        uvhG = uvhG.flatten()
        UVHG[:, 0] = uvhG
    else:
        uvhG = None
        UVHG = None

    return (uvhG, UVHG)
