import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import time
from mpi4py import MPI
from flux_ener_components import calc_h, calc_u, calc_v, calc_b, calc_q, calc_flux_1, \
                                 calc_flux_2, calc_flux_3, calc_energy, calc_enstrophy
from flux_sw_ener import euler_f as ener_Euler_f, ab2_f as ener_AB2_f, ab3_f as ener_AB3_f
comm = MPI.COMM_WORLD


def dxp(f, dx):
    fx = (np.roll(f, -1, 1) - f)/dx
    return fx


def dyp(f, dy):
    fy = (np.roll(f, -1, 0) - f)/dy
    return fy


def dxm(f, dx):
    fx = (f - np.roll(f, 1, 1))/dx
    return fx


def dym(f, dy):
    fy = (f - np.roll(f, 1, 0))/dy
    return fy


def axp(f):
    afx = 0.5*(np.roll(f, -1, 1) + f)
    return afx


def ayp(f):
    afy = 0.5*(np.roll(f, -1, 0) + f)
    return afy


def axm(f):
    afx = 0.5*(f + np.roll(f, 1, 1))
    return afx


def aym(f):
    afy = 0.5*(f + np.roll(f, 1, 0))
    return afy


def euler(uvh, dt, NLnm):
    """
    Updates the solution, uvh, using an Euler time stepping method.
    """
    return uvh + dt*NLnm


def ab2(uvh, dt, NLn, NLnm):
    """
    Updates the solution, uvh, using a second order Adams-Bashforth time
    stepping method.
    """
    return uvh + 0.5*dt*(3*NLn - NLnm)


def ab3(uvh, dt, NL, NLn, NLnm):
    """
    Updates the solution, uvh, using a third order Adams-Bashforth time
    stepping method.
    """
    return uvh + dt/12*(23*NL - 16*NLn + 5*NLnm)


def flux_sw_ener(uvh, params, inds):

    # Define parameters
    dx = params.dx
    dy = params.dy
    gp = params.gp
    f0 = params.f0
    H0 = params.H0
    Iu_i = inds.Iu_i
    Iu_f = inds.Iu_f
    Iv_i = inds.Iv_i
    Iv_f = inds.Iv_f
    Ih_i = inds.Ih_i
    Ih_f = inds.Ih_f

    # Turn off nonlinear terms
    h = H0 + uvh[Ih_i:Ih_f]
    U = axp(h)*uvh[Iu_i:Iu_f, :]
    V = ayp(h)*uvh[Iv_i:Iv_f, :]
    B = gp*h + 0.5*(axm(uvh[Iu_i:Iu_f, :]**2) + aym(uvh[Iv_i:Iv_f, :]**2))
    q = (dxp(uvh[Iv_i:Iv_f, :], dx) - dyp(uvh[Iu_i:Iu_f, :], dy) + f0)/ayp(axp(h))

    # Compute fluxes
    flux = np.vstack([aym(q*axp(V)) - dxp(B, dx),
                     -axm(q*ayp(U)) - dyp(B, dy),
                     -dxm(U, dx) - dym(V, dy)])

    # compute energy and enstrophy
    energy = 0.5*np.mean(gp*h**2 + h*(axm(uvh[Iu_i:Iu_f, :]**2) + aym(uvh[Iv_i:Iv_f, :]**2)))
    enstrophy = 0.5*np.mean(q**2*ayp(axp(h)))

    return flux, energy, enstrophy


def flux_sw_enst(uvh, params, inds):

    # Define parameters
    dx = params.dx
    dy = params.dy
    gp = params.gp
    f0 = params.f0
    H0 = params.H0
    Iu_i = inds.Iu_i
    Iu_f = inds.Iu_f
    Iv_i = inds.Iv_i
    Iv_f = inds.Iv_f
    Ih_i = inds.Ih_i
    Ih_f = inds.Ih_f

    h = H0 + uvh[Ih_i:Ih_f]
    U = axp(h)*uvh[Iu_i:Iu_f, :]
    V = ayp(h)*uvh[Iv_i:Iv_f, :]
    B = gp*h + 0.5*(axm(uvh[Iu_i:Iu_f, :]**2) + aym(uvh[Iv_i:Iv_f, :]**2))
    q = (dxp(uvh[Iv_i:Iv_f, :], dx) - dyp(uvh[Iu_i:Iu_f, :], dy) + f0)/ayp(axp(h))

    # Compute fluxe (use np.vstack)
    flux1 =  aym(q)*aym(axp(V)) - dxp(B, dx)
    flux2 = -axm(q)*axm(ayp(U)) - dyp(B, dy)
    flux3 = -dxm(U, dx) - dym(V, dy)

    flux = np.vstack([flux1, flux2, flux3])

    # compute energy and enstrophy
    energy = 0.5*np.mean(gp*h**2 + h*(axm(uvh[Iu_i:Iu_f, :]**2) + aym(uvh[Iv_i:Iv_f, :]**2)))
    enstrophy = 0.5*np.mean(q**2*ayp(axp(h)))

    return flux, energy, enstrophy


def flux_sw_ener_Fcomp(uvh, params, inds):

    # Define parameters
    dx = params.dx
    dy = params.dy
    gp = params.gp
    f0 = params.f0
    H0 = params.H0
    Iu_i = inds.Iu_i
    Iu_f = inds.Iu_f
    Iv_i = inds.Iv_i
    Iv_f = inds.Iv_f
    Ih_i = inds.Ih_i
    Ih_f = inds.Ih_f

    # Turn off nonlinear terms
    h = calc_h(uvh, H0, Ih_i)
    U = calc_u(uvh, h)
    V = calc_v(uvh, h, Iv_i, Iv_f)
    B = calc_b(uvh, h, gp, Iv_i, Iv_f)
    q = calc_q(uvh, h, dx, dy, f0, Iv_i, Iv_f)

    # Compute fluxes
    flux = np.vstack([calc_flux_1(q, V, B, dx),
                      calc_flux_2(q, U, B, dy),
                      calc_flux_3(U, V, dx, dy)])

    # compute energy and enstrophy
    energy = calc_energy(uvh, h, gp, Iv_i, Iv_f)
    enstrophy = calc_enstrophy(h, q)

    return flux, energy, enstrophy


def ener_Euler(uvh, params, inds):
    NLnm, energy, enstr = flux_sw_ener(uvh, params, inds)
    uvh = euler(uvh, params.dt, NLnm)
    return uvh, NLnm, energy, enstr


def ener_AB2(uvh, NLnm, params, inds):
    NLn, energy, enstr = flux_sw_ener(uvh, params, inds)
    uvh = ab2(uvh, params.dt, NLn, NLnm)
    return uvh, NLn, energy, enstr


def ener_AB3(uvh, NLn, NLnm, params, inds):
    NL, energy, enstr = flux_sw_ener(uvh, params, inds)
    uvh = ab3(uvh, params.dt, NL, NLn, NLnm)
    return uvh, NL, energy, enstr


def set_mpi_bdr(uvh, rank, p, mx, col, tags):
    """
    Given a process' slice of the solution (uvh), the second column is
    communicated to the left process and the second-last column is communicated
    to the right process, placing them in the appropriate columns (the "ghost
    columns").

    Note: process 0 is to the right of process (p-1) due to the periodic grid.

    Parameters
    ----------
    uvh : ndarray
        the process' slice of the solution
    rank : int
        rank of this process
    p : int
        the total number of processes
    mx : int
        the number of real (non-ghost) x points per process
    col : ndarray
        an array the same size as a column of uvh, used as a placeholder for
        communicating the columns.
    tags : dict
        a dictionary that associates a tag number with each rank

    Returns
    -------
    ndarray
        modified version of uvh, where the ghost columns have been updated via
        MPI communication.
    """
    # send uvh[:, 1] (second column) ...
    if 0 < rank:
        # ...from rank to rank-1 (left)
        comm.Send(uvh[:, 1].flatten(), dest=rank-1, tag=tags[rank])
    else:
        # ...from 0 to p-1
        comm.Send(uvh[:, 1].flatten(), dest=p-1,    tag=tags[rank])

    # receive uvh[:, 1], place in uvh[:, mx+1] (last column) ...
    if rank < p-1:
        # ... from rank+1
        comm.Recv(col, source=rank+1, tag=tags[rank+1])
        uvh[:, mx+1] = col
    else:
        # ... from rank 0
        comm.Recv(col, source=0,      tag=tags[0])
        uvh[:, mx+1] = col

    # send uvh[:, mx] (second-last column)...
    if rank < p-1:
        # ...from rank to rank+1
        comm.Send(uvh[:, mx].flatten(), dest=rank+1, tag=tags[rank])
    else:
        # ...from p-1 to 0
        comm.Send(uvh[:, mx].flatten(), dest=0,      tag=tags[rank])

    # receive uvh[:, mx], place in uvh[:, 0] (first column) ...
    if 0 < rank:
        # ...from rank-1
        comm.Recv(col, source=rank-1, tag=tags[rank-1])
        uvh[:, 0] = col
    else:
        # ...from p-1
        comm.Recv(col, source=p-1,    tag=tags[p-1])
        uvh[:, 0] = col

    return uvh


def gather_uvh(uvh, uvhG, UVHG, rank, p, mx, Ny, n):
    """
    Gather each process' slice of uvh and store it in the global set of
    solutions. Gathering is done on process 0.

    Parameters
    ----------
    uvh : ndarray
        process' slice of the solution (dimensions: (Ny, mx+2))
    uvhG : ndarray
        the global solution, excluding ghost points (dimensions: (Ny, p*mx))
    UVHG : ndarray
        the set of global solutions (dimensions: (Ny, p*mx, N)) where N is the
        number of time steps defined elsewhere.
    rank : int
        rank of this process
    mx : int
        the number of real (non-ghost) x points per process
    Ny : int
        the number of y points
    n : int
        time-step number

    Returns
    -------
    ndarray or None
        If this process' rank is 0, then the newest solution is added to UVHG
        and UVHG is returned. Otherwise, None is returned.
    """
    comm.Gather(uvh[:, 1:mx+1].flatten(), uvhG, root=0)
    if rank == 0:
        # evenly split ug into a list of p parts
        temp = np.array_split(uvhG, p)
        # reshape each part
        temp = [a.reshape(3*Ny, mx) for a in temp]
        UVHG[:, :, n] = np.hstack(temp)
        return UVHG
    else:
        return None


class Params(object):
    """
    A placeholder for the parameters of the solution.

    Parameters
    ----------
    dx : float64
        the spacing between two x points of the same kind (staggered vs
        non-staggered)
    dy : float64
        the spacing between two y points of the same kind
    f0 : float64
        TODO
    beta : float64
        TODO
    gp : float64
        TODO
    H0 : float64
        TODO
    Nx : int
        The number of x points. If running in parallel, Nx is the number of x
        points per process, excluding ghost points.
    Ny : int
        The number of y points.

    Attributes
    ----------
    All parameters are also defined as attributes with the same name.
    """
    def __init__(self, dx, dy, f0, beta, gp, H0, Nx, Ny, dt):
        super(Params, self).__init__()
        self.dx   = dx
        self.dy   = dy
        self.f0   = f0
        self.beta = beta
        self.gp   = gp
        self.H0   = H0
        self.Nx   = Nx
        self.Ny   = Ny
        self.dt   = dt


class Inds(object):
    """
    A placeholder for the indices of the solution matrix.

    Parameters
    ----------
    Ny : int
        The number of y points.

    Attributes
    ----------
    Iu_i : int
        The row number that the matrix of u values begins at.
    Iu_f : int
        The row number that the matrix of u values end at, plus one for the
        sake of indexing.
    Iv_i : int
        The row number that the matrix of v values begins at.
    Iv_f : int
        The row number that the matrix of v values end at, plus one for the
        sake of indexing.
    Ih_i : int
        The row number that the matrix of h values begins at.
    Ih_f : int
        The row number that the matrix of h values end at, plus one for the
        sake of indexing.
    """
    def __init__(self, Ny):
        super(Inds, self).__init__()
        self.Iu_i = 0    # first row of u
        self.Iu_f = Ny   # last row of u; not inclusive!
        self.Iv_i = Ny   # first row of v...
        self.Iv_f = 2*Ny
        self.Ih_i = 2*Ny
        self.Ih_f = 3*Ny


def create_global_objects(rank, xG, yG, xsG, ysG, hmax, Lx, N):
    """
    If process is 0:
        - construct global staggered & non-staggered meshgrids
        - construct initial global solution, uvhG; flatten it for use in MPI.
        - construct set of global solutions, UVHG
        - return tuple (uvhG, UVHG)

    Otherwise: do nothing!

    Parameters
    ----------
    rank : int
        rank of process
    xG : ndarray
        global non-staggered x points (entire set of non-staggered points in x)
    yG : ndarray
        global non-staggered y points (entire set of non-staggered points in y)
    xsG : ndarray
        global staggered x points (entire set of staggered points in x)
    ysG : ndarray
        global staggered y points (entire set of staggered points in y)
    hmax : float64
        maximum water height
    Lx : float64
        length of x-axis
    N : int
        number of time steps

    Returns
    -------
    tuple
        (uvhG, UVHG) if process' rank is 0; (None, None) otherwise.
    """

    if rank == 0:
        xqG, yqG = np.meshgrid(xsG, ysG)
        xhG, yhG = np.meshgrid(xG,  yG)
        xuG, yuG = np.meshgrid(xsG, yG)
        xvG, yvG = np.meshgrid(xG,  ysG)

        Nx = len(xG)
        Ny = len(yG)

        # global initial solution
        uvhG = np.vstack([0.*xuG,
                          0.*xvG,
                          hmax*np.exp(-(xhG**2 + (1.0*yhG)**2)/(Lx/6.0)**2)]).flatten()

        # set of ALL global solutions
        UVHG = np.empty((3*Ny, Nx, N), dtype='d')
        UVHG[:, :, 0] = uvhG.reshape(3*Ny, Nx)

    else:
        uvhG = None
        UVHG = None

    return (uvhG, UVHG)


def create_x_points(rank, p, xG, xsG, mx):
    """
    Create the non-staggered and staggered x grid points for each process,
    including the appropriate ghost points (one on the left, one on the right).

    Parameters
    ----------
    rank : int
        rank of process
    p : int
        total number of processes
    xG : ndarray
        global non-staggered x points (entire set of non-staggered points in x)
    xsG : ndarray
        global staggered x points (entire set of staggered points in x)
    mx : int
        number of x points per process


    Returns
    -------
    tuple
        the x points for the given process, both non-staggered and staggered,
        including ghost points, with dimensions (1, mx+2).
    """

    if rank == 0:
        x  = np.hstack([[xG[-1]],  xG[:mx+1]])
        xs = np.hstack([[xsG[-1]], xsG[:mx+1]])

    elif rank == p-1:
        x  = np.hstack([xG[rank*mx - 1:],  [xG[0]]])
        xs = np.hstack([xsG[rank*mx - 1:], [xsG[0]]])

    else:
        x  = xG[rank*mx - 1: (rank+1)*mx + 1]
        xs = xsG[rank*mx - 1: (rank+1)*mx + 1]

    return (x, xs)


def PLOTTO_649(UVHG, xG, yG, Ny, N, output_name):
    """
    Creates an animation of the system evolving through time by using the
    set of global solutions.

    Parameters
    ----------
    UVHG : ndarray
        set of global solutions
    xG : ndarray
        global non-staggered x points
    yG : ndarray
        global non-staggered y points
    Ny : int
        number of y points
    N : int
        number of time steps
    output_name : str
        name of the animation file
    """
    xhG, yhG = np.meshgrid(xG,  yG)

    fig = plt.figure()
    ims = []
    for n in xrange(N):
        ims.append((plt.pcolormesh(xhG/1e3, yhG/1e3, UVHG[2*Ny:3*Ny, :, n], norm=plt.Normalize(0, 1)), ))

    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)
    im_ani.save(output_name)
