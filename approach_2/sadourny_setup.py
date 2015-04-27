from __future__ import division
import numpy as np
import sys
import time
import os


def I(A):
    # index A normally...
    return A[1:-1, 1:-1]


def I_XP(A):
    # index A with x being pushed in positive direction (+1)
    return A[1:-1, 2:]


def I_XN(A):
    # index A with x being pushed in negative direction (-1)
    return A[1:-1, 0:-2]


def I_YP(A):
    return A[2:, 1:-1]


def I_YN(A):
    return A[0:-2, 1:-1]


def I_XP_YP(A):
    # index A with x,y being pushed in pos. dir.
    return A[2:, 2:]


def I_XP_YN(A):
    # index A with x pushed in pos, y pushed in neg.
    return A[0:-2, 2:]


def I_XN_YP(A):
    # index A with x pushed in neg, y pushed in pos
    return A[2:, 0:-2]


def I_XN_YN(A):
    return A[0:-2, 0:-2]


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


def euler(uvh, dt, NLnm):
    return uvh[:, 1:-1, 1:-1] + dt*NLnm


def ab2(uvh, dt, NLn, NLnm):
    return uvh[:, 1:-1, 1:-1] + 0.5*dt*(3*NLn - NLnm)


def ab3(uvh, dt, NL, NLn, NLnm):
    return uvh[:, 1:-1, 1:-1] + dt/12*(23*NL - 16*NLn + 5*NLnm)


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
