from __future__ import division
import numpy as np
import sys
import time
import os
from fjp_helpers.mpi import create_x, create_y, create_ranks, create_tags, \
                            send_periodic, send_cols_periodic, send_rows_periodic
from fjp_helpers.bc import set_periodic_BC, set_periodic_BC_x, set_periodic_BC_y
from fjp_helpers.misc import write_time
from sadourny_helpers import set_periodic_BC_placeholder
from setup_serial import solver_serial
from setup_mpi import solver_mpi_1D
from mpi4py import MPI
comm = MPI.COMM_WORLD


def solver(params, px, py, SAVE_TIME=False, ANIMATE=False, SAVE_SOLN=False):
    p = px * py
    if p != comm.Get_size():
        if comm.Get_rank() == 0:
            raise Exception("Incorrect number of cores used; MPI is being run with %d, "
                            "but %d was inputted." % (comm.Get_size(), p))

    # update Params object with p value and updater
    params.set_p_vars([p, px, py])

    # create ranks and tags in all directions
    rank = comm.Get_rank()
    rankL, rankR, rankU, rankD = create_ranks(rank, p, px, py)[1:]
    tagsL, tagsR, tagsU, tagsD = create_tags(p)

    # get variables and type of BCs
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    t0, tf, dt, Nt = params.t_vars
    C, Kx, Ky = params.consts
    bcs_type  = params.bcs_type
    method    = params.method

    # split x and y values along each process
    x = create_x(px, rank, x0, xf, dx, nx, Nx)
    y = create_y(px, py, rank, y0, yf, dy, ny, Ny)
    t = np.linspace(t0, tf, Nt)

    # BUILD ZE GRID (of initial conditions)
    f = params.ics
    uvh = f(x, y, params)

    # create ghost column and row
    col = np.empty(ny+2, dtype='d')
    row = np.empty(nx+2, dtype='d')

    # update Params object with BC functions
    # if the BCs are periodic...
    if bcs_type == 'P':
        # if any of the BC functions passed were None, use default periodic BC functions
        if any([bc is None for bc in params.bcs]):
            params.set_bc_funcs([set_periodic_BC, set_periodic_BC_x, set_periodic_BC_y,
                                 set_periodic_BC_placeholder])

    # if we have one process per direction, we're solving it in serial
    if px == 1 and py == 1:
        params.mpi_func = None
        params.bc_func  = params.bc_s
        t_total = solver_serial(uvh, params, SAVE_TIME, ANIMATE, SAVE_SOLN)

    # if we have one process in y, we're parallelizing the solution in x
    elif py == 1:
        ranks = (rank,  rankL, rankR)
        tags  = (tagsL, tagsR)
        params.mpi_func = send_cols_periodic
        params.bc_func  = params.bc_y
        t_total = solver_mpi_1D(uvh, ranks, col, tags, params, SAVE_TIME, ANIMATE, SAVE_SOLN)

    # if we have one process in x, we're parallelizing the solution in y
    elif px == 1:
        ranks = (rank,  rankU, rankD)
        tags  = (tagsU, tagsD)
        params.mpi_func = send_rows_periodic
        params.bc_func  = params.bc_x
        t_total = solver_mpi_1D(uvh, ranks, row, tags, params, SAVE_TIME, ANIMATE, SAVE_SOLN)

    # otherwise we're parallelizing the solution in both directions
    else:
        ranks = (rank,  rankL, rankR, rankU, rankD)
        tags  = (tagsL, tagsR, tagsU, tagsD)
        params.mpi_func = send_periodic
        params.bc_func  = params.bc_xy
        t_total = solver_mpi_2D(uvh, ranks, col, row, tags, params, ANIMATE, SAVE_SOLN)

    if rank == 0:
        # save the time to a file
        if SAVE_TIME:
            filename = params.filename_time.split(os.sep)
            direc, filename = os.sep.join(filename[:-1]), filename[-1]
            if not os.path.isdir(direc):
                os.makedirs(direc)
            write_time(t_total, direc, filename)

        print t_total

    return t_total


def odd(f):
    f[0,  :] = -f[1,  :]
    f[-1, :] = -f[-2, :]
    f[:,  0] =  f[:, -2]
    f[:, -1] =  f[:,  1]

    return f


def even(f):
    f[0,  :] = f[1,  :]
    f[-1, :] = f[-2, :]
    f[:,  0] = f[:, -2]
    f[:, -1] = f[:,  1]

    return f


class Params(object):
    """Placeholder for several constants and what not."""
    def __init__(self):
        super(Params, self).__init__()
        self.x_vars, self.y_vars, self.t_vars, self.p_vars, self.consts = 5*[None]
        self.funcs = {}

    def __str__(self):
        return "x-vars: %s\ny-vars: %s\nt-vars: %s\np-vars: %s\nconsts: %s\n""" % (
            str(self.x_vars), str(self.y_vars), str(self.t_vars), str(self.p_vars),
            str(self.consts)
        )

    def set_x_vars(self, x_vars):
        self.x_vars = x_vars
        self.x0, self.xf, self.dx, self.Nx, self.nx = x_vars

    def set_y_vars(self, y_vars):
        self.y_vars = y_vars
        self.y0, self.yf, self.dy, self.Ny, self.ny = y_vars

    def set_t_vars(self, t_vars):
        self.t_vars = t_vars
        self.t0, self.tf, self.dt, self.Nt = t_vars

    def set_p_vars(self, p_vars):
        self.p_vars = p_vars
        self.p, self.px, self.py = p_vars

    def set_consts(self, consts):
        self.consts = consts
        self.f0, self.gp, self.H0 = consts

    def set_funcs(self, funcs):
        # funcs is a dictionary
        self.funcs = funcs
        self.ics = funcs['ic']
        self.bc_s, self.bc_x  = funcs['bc_s'], funcs['bc_x']
        self.bc_y, self.bc_xy = funcs['bc_y'], funcs['bc_xy']
        self.updater  = funcs['updater']
        self.mpi_func = funcs['mpi']

    def set_bc_funcs(self, bc_funcs):
        # bc_funcs is an array, [serial, x, y, xy]
        self.bc_s, self.bc_x, self.bc_y, self.bc_xy = bc_funcs
        self.funcs['bc_s'], self.funcs['bc_x']  = self.bc_s, self.bc_x
        self.funcs['bc_y'], self.funcs['bc_xy'] = self.bc_y, self.bc_xy
        self.bcs = bc_funcs
