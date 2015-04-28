from __future__ import division
import os
import numpy as np
from mpi4py import MPI
from sadourny_helpers import I, I_XP, I_XN, I_YP, I_YN, I_XP_YP, I_XP_YN, I_XN_YP, \
                             euler, ab2, ab3, animate_solution, create_global_objects
from fjp_helpers.mpi import send_periodic
comm = MPI.COMM_WORLD


def solver_mpi_1D(u, ranks, ghost_arr, tags, params, save_time, animate, save_soln):
    params.euler = ener_Euler_MPI
    params.ab2   = ener_AB2_MPI
    params.ab3   = ener_AB3_MPI

    if animate:
        t_total, UVHG = solver_1D_helper_g(u, ranks, ghost_arr, tags, params)
        if ranks[0] == 0:
            animate_solution(UVHG, 0, params)

    if save_soln:
        t_total = solver_1D_helper_w(u, ranks, ghost_arr, tags, params)

    if save_time:
        t_total = solver_1D_helper(u, ranks, ghost_arr, tags, params)

    return t_total


def solver_1D_helper(uvh, ranks, ghost_arr, tags, params):
    Flux_Euler, Flux_AB2, Flux_AB3 = params.euler, params.ab2, params.ab3
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    p, px, py = params.p_vars
    MPI_Func  = params.mpi_func
    BC_Func   = params.bc_func
    Nt = params.Nt

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # Euler step
    uvh, NLnm, energy, enstr = Flux_Euler(uvh, ranks, px, ghost_arr, tags, params)
    uvh = set_uvh_bdr_1D(uvh, ranks, px, ghost_arr, tags, MPI_Func, BC_Func)

    # AB2 step
    uvh, NLn, energy, enstr = Flux_AB2(uvh, NLnm, ranks, px, ghost_arr, tags, params)
    uvh = set_uvh_bdr_1D(uvh, ranks, px, ghost_arr, tags, MPI_Func, BC_Func)

    # loop through time
    for n in range(3, Nt):
        # AB3 step
        uvh, NL, energy, enstr = Flux_AB3(uvh, NLn, NLnm, ranks, px, ghost_arr, tags, params)

        # Reset fluxes
        NLnm, NLn = NLn, NL

        uvh = set_uvh_bdr_1D(uvh, ranks, px, ghost_arr, tags, MPI_Func, BC_Func)

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer

    return t_total


def solver_1D_helper_g(uvh, ranks, ghost_arr, tags, params):
    Flux_Euler, Flux_AB2, Flux_AB3 = params.euler, params.ab2, params.ab3
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    p, px, py = params.p_vars
    MPI_Func  = params.mpi_func
    BC_Func   = params.bc_func
    Nt = params.Nt

    rank = ranks[0]
    uvhG, UVHG = create_global_objects(rank, params)

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # Euler step
    uvh, NLnm, energy, enstr = Flux_Euler(uvh, ranks, px, ghost_arr, tags, params)
    uvh = set_uvh_bdr_1D(uvh, ranks, px, ghost_arr, tags, MPI_Func, BC_Func)
    comm.Gather(uvh[:, 1:-1, 1:-1].flatten(), uvhG, root=0)
    if rank == 0:
        UVHG[:, 1] = uvhG

    # AB2 step
    uvh, NLn, energy, enstr = Flux_AB2(uvh, NLnm, ranks, px, ghost_arr, tags, params)
    uvh = set_uvh_bdr_1D(uvh, ranks, px, ghost_arr, tags, MPI_Func, BC_Func)
    comm.Gather(uvh[:, 1:-1, 1:-1].flatten(), uvhG, root=0)
    if rank == 0:
        UVHG[:, 2] = uvhG

    # loop through time
    for n in range(3, Nt):
        # AB3 step
        uvh, NL, energy, enstr = Flux_AB3(uvh, NLn, NLnm, ranks, px, ghost_arr, tags, params)

        # Reset fluxes
        NLnm, NLn = NLn, NL

        uvh  = set_uvh_bdr_1D(uvh, ranks, px, ghost_arr, tags, MPI_Func, BC_Func)
        comm.Gather(uvh[:, 1:-1, 1:-1].flatten(), uvhG, root=0)
        if rank == 0:
            UVHG[:, n] = uvhG

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer

    return t_total, UVHG


def solver_mpi_2D(u, ranks, col, row, tags, params, save_time, animate, save_soln):
    params.euler = ener_Euler_MPI_2D
    params.ab2   = ener_AB2_MPI_2D
    params.ab3   = ener_AB3_MPI_2D

    if animate:
        t_total, UVHG = solver_2D_helper_g(u, ranks, col, row, tags, params)
        if ranks[0] == 0:
            animate_solution(UVHG, 0, params)

    if save_soln:
        t_total = solver_2D_helper_w(u, ranks, col, row, tags, params)

    if save_time:
        t_total = solver_2D_helper(u, ranks, col, row, tags, params)

    return t_total


def solver_2D_helper(uvh, ranks, col, row, tags, params):
    Flux_Euler, Flux_AB2, Flux_AB3 = params.euler, params.ab2, params.ab3
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    p, px, py = params.p_vars
    Nt = params.Nt

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # Euler step
    uvh, NLnm, energy, enstr = Flux_Euler(uvh, ranks, px, col, row, tags, params)
    uvh = set_uvh_bdr_2D(uvh, ranks, px, col, row, tags)

    # AB2 step
    uvh, NLn, energy, enstr = Flux_AB2(uvh, NLnm, ranks, px, col, row, tags, params)
    uvh = set_uvh_bdr_2D(uvh, ranks, px, col, row, tags)

    # loop through time
    for n in range(3, Nt):
        # AB3 step
        uvh, NL, energy, enstr = Flux_AB3(uvh, NLn, NLnm, ranks, px, col, row, tags, params)

        # Reset fluxes
        NLnm, NLn = NLn, NL

        uvh = set_uvh_bdr_2D(uvh, ranks, px, col, row, tags)

    comm.Barrier()
    t_total = (MPI.Wtime() - t_start)  # stop MPI timer

    return t_total


def solver_2D_helper_g(uvh, ranks, col, row, tags, params):
    Flux_Euler, Flux_AB2, Flux_AB3 = params.euler, params.ab2, params.ab3
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    p, px, py = params.p_vars
    Nt = params.Nt

    rank = ranks[0]
    uvhG, UVHG = create_global_objects(rank, params)

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()

    # Euler step
    uvh, NLnm, energy, enstr = Flux_Euler(uvh, ranks, px, col, row, tags, params)
    uvh = set_uvh_bdr_2D(uvh, ranks, px, col, row, tags)
    comm.Gather(uvh[:, 1:-1, 1:-1].flatten(), uvhG, root=0)
    if rank == 0:
        UVHG[:, 1] = uvhG

    # AB2 step
    uvh, NLn, energy, enstr = Flux_AB2(uvh, NLnm, ranks, px, col, row, tags, params)
    uvh = set_uvh_bdr_2D(uvh, ranks, px, col, row, tags)
    comm.Gather(uvh[:, 1:-1, 1:-1].flatten(), uvhG, root=0)
    if rank == 0:
        UVHG[:, 2] = uvhG

    # loop through time
    for n in range(3, Nt):
        # AB3 step
        uvh, NL, energy, enstr = Flux_AB3(uvh, NLn, NLnm, ranks, px, col, row, tags, params)

        # Reset fluxes
        NLnm, NLn = NLn, NL

        uvh  = set_uvh_bdr_2D(uvh, ranks, px, col, row, tags)
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
    MPI_Func = params.mpi_func

    # define what kind of BC function we're using (i.e. rows or cols)
    BC_Func  = params.bc_func

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


def flux_sw_ener_MPI_2D(uvh, ranks, px, col, row, tags, params):
    # All terms (h, U, V, B, etc...) are calculated in numpy.
    rank, rankL, rankR, rankU, rankD = ranks
    tagsL, tagsR, tagsU, tagsD = tags

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
    U = send_periodic(U, rank, px, col, row, tagsL, tagsR, tagsU, tagsD,
                      rankL, rankR, rankU, rankD)
    V = send_periodic(V, rank, px, col, row, tagsL, tagsR, tagsU, tagsD,
                      rankL, rankR, rankU, rankD)
    B = send_periodic(B, rank, px, col, row, tagsL, tagsR, tagsU, tagsD,
                      rankL, rankR, rankU, rankD)
    q = send_periodic(q, rank, px, col, row, tagsL, tagsR, tagsU, tagsD,
                      rankL, rankR, rankU, rankD)

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


def ener_Euler_MPI_2D(uvh, ranks, px, col, row, tags, params):
    # MPI'd pure Numpy
    NLnm, energy, enstr = flux_sw_ener_MPI_2D(uvh, ranks, px, col, row, tags, params)
    uvh[:, 1:-1, 1:-1]  = euler(uvh, params.dt, NLnm)
    return uvh, NLnm, energy, enstr


def ener_AB2_MPI_2D(uvh, NLnm, ranks, px, col, row, tags, params):
    NLn, energy, enstr = flux_sw_ener_MPI_2D(uvh, ranks, px, col, row, tags, params)
    uvh[:, 1:-1, 1:-1] = ab2(uvh, params.dt, NLn, NLnm)
    return uvh, NLn, energy, enstr


def ener_AB3_MPI_2D(uvh, NLn, NLnm, ranks, px, col, row, tags, params):
    NL, energy, enstr  = flux_sw_ener_MPI_2D(uvh, ranks, px, col, row, tags, params)
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


def set_uvh_bdr_2D(uvh, ranks, px, col, row, tags):
    rank, rankL, rankR, rankU, rankD = ranks
    tagsL, tagsR, tagsU, tagsD = tags

    u, v, h = uvh[0, :, :], uvh[1, :, :], uvh[2, :, :]

    u = send_periodic(u, rank, px, col, row, tagsL, tagsR, tagsU, tagsD,
                      rankL, rankR, rankU, rankD)
    v = send_periodic(v, rank, px, col, row, tagsL, tagsR, tagsU, tagsD,
                      rankL, rankR, rankU, rankD)
    h = send_periodic(h, rank, px, col, row, tagsL, tagsR, tagsU, tagsD,
                      rankL, rankR, rankU, rankD)

    uvh[0, :, :], uvh[1, :, :], uvh[2, :, :] = u, v, h
    return uvh
