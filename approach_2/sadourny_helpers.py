import numpy as np
import os
from fjp_helpers.animator import mesh_animator


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


def euler(uvh, dt, NLnm):
    return uvh[:, 1:-1, 1:-1] + dt*NLnm


def ab2(uvh, dt, NLn, NLnm):
    return uvh[:, 1:-1, 1:-1] + 0.5*dt*(3*NLn - NLnm)


def ab3(uvh, dt, NL, NLn, NLnm):
    return uvh[:, 1:-1, 1:-1] + dt/12*(23*NL - 16*NLn + 5*NLnm)


def create_global_objects(rank, params):
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    hmax, Lx = params.hmax, params.Lx
    Nt = params.Nt
    xG = np.linspace(x0 + dx/2, xf - dx/2, Nx)    # global x points
    yG = np.linspace(y0 + dy/2, yf - dy/2, Ny)

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


def set_periodic_BC_placeholder(uvh):
    return uvh


def animate_solution(UVHG, rank, params):
    x0, xf, dx, Nx, nx = params.x_vars
    y0, yf, dy, Ny, ny = params.y_vars
    p, px, py = params.p_vars
    Nt = params.Nt

    xG = np.linspace(x0 - dx/2, xf + dx/2, Nx+2)    # global x points (padded)
    yG = np.linspace(y0 - dy/2, yf + dy/2, Ny+2)

    H = np.empty((Ny*Nx, Nt), dtype='d')
    for i in xrange(Nt):
        temp = np.array_split(UVHG[:, i], p)
        temp = [np.array_split(part, 3)[2] for part in temp]
        H[:, i] = np.hstack(temp)

    filename = params.filename_anim.split(os.sep)    # split filename according to OS' separator
    direc, filename = os.sep.join(filename[:-1]), filename[-1]
    mesh_animator(H, xG, yG, nx, ny, Nt, p, px, py, direc, filename)
