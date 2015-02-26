#!/usr/bin/env python
#  SW_Sadourny.m
#
# Solve the 1-Layer Rotating Shallow Water (SW) Model
#
# Fields:
#   u : zonal velocity
#   v : meridional velocity
#   h : fluid depth
#
# Evolution Eqns:
#   B = g*h + 0.5*(u**2 + v**2)     Bernoulli function
#   Z = v_x - u_y + f               Total Vorticity
#   q = Z/h                         Potential Vorticity
#   [U,V] = h[u,v]                  Transport velocities
#
#   u_t =  (q*V^x)^y + d_x h
#   v_t = -(q*U^y)^x + d_y h
#   h_t = - div[U,V]
#
# Geometry: periodic in x and y
#           Arakawa C-grid
#
#      |           |          |         |
#      h --  u --  h  -- u -- h -- u -- h --
#      |           |          |         |
#      |           |          |         |
#      v     q     v     q    v    q    v
#      |           |          |         |
#      |           |          |         |
#      h --  u --  h  -- u -- h -- u -- h --
#      |           |          |         |
#      |           |          |         |
#      v     q     v     q    v    q    v
#      |           |          |         |
#      |           |          |         |
#      h --  u --  h  -- u -- h -- u -- h --
#      |           |          |         |
#      |           |          |         |
#      v     q     v     q    v    q    |
#      |           |          |         |
#      |           |          |         |
#      h --  u --  h  -- u -- h -- u -- h --
#
#      Because of periodicity all fields are Mx by My
#      But we need to define different grids for u,v,h,q
#
# Numerical Method:
# 1) Sadourny's method 1 (energy conserving) and 2 (enstrophy conserving)
# 2) Adams-Bashforth for time stepping
#
# Requires scripts:
#        flux_sw_ener.py  - Sadourny's first method (energy conserving)
#        flux_sw_enst.py  - Sadourny's second method (enstrophy conserving)


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD

class wavenum:
    pass


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


def flux_sw_ener(uvh, params):

    # Define parameters
    dx = params.dx
    dy = params.dy
    gp = params.gp
    f0 = params.f0
    H0 = params.H0
    Mx = params.Mx
    My = params.My

    # Separate fields
    Iu, Iv, Ih = range(0, My), range(My, 2*My), range(2*My, 3*My)

    # Turn off nonlinear terms
    h = H0  +  uvh[Ih, :]
    U = axp(h)*uvh[Iu, :]
    V = ayp(h)*uvh[Iv, :]
    B = gp*h + 0.5*(axm(uvh[Iu, :]**2) + aym(uvh[Iv, :]**2))
    q = (dxp(uvh[Iv, :], dx) - dyp(uvh[Iu, :], dy) + f0)/ayp(axp(h))

    # Compute fluxes
    flux = np.vstack([aym(q*axp(V)) - dxp(B, dx),
                     -axm(q*ayp(U)) - dyp(B, dy),
                     -dxm(U, dx) - dym(V, dy)])

    # compute energy and enstrophy
    energy = 0.5*np.mean(gp*h**2 + h*(axm(uvh[Iu, :]**2) + aym(uvh[Iv, :]**2)))
    enstrophy = 0.5*np.mean(q**2*ayp(axp(h)))

    return flux, energy, enstrophy


def flux_sw_enst(uvh, params):

    # Define parameters
    dx = params.dx
    dy = params.dy
    gp = params.gp
    f0 = params.f0
    H0 = params.H0
    Mx = params.Mx
    My = params.My

    # Separate fields
    Iu, Iv, Ih = range(0, My), range(My, 2*My), range(2*My, 3*My)

    h = H0 + uvh[Ih, :]
    U = axp(h)*uvh[Iu, :]
    V = ayp(h)*uvh[Iv, :]
    B = gp*h + 0.5*(axm(uvh[Iu, :]**2) + aym(uvh[Iv, :]**2))
    q = (dxp(uvh[Iv, :], dx) - dyp(uvh[Iu, :], dy) + f0)/ayp(axp(h))

    # Compute fluxe (use np.vstack)
    flux1 =  aym(q)*aym(axp(V)) - dxp(B, dx)
    flux2 = -axm(q)*axm(ayp(U)) - dyp(B, dy)
    flux3 = -dxm(U, dx) - dym(V, dy)

    flux = np.vstack([flux1, flux2, flux3])

    # compute energy and enstrophy
    energy = 0.5*np.mean(gp*h**2 + h*(axm(uvh[Iu, :]**2) + aym(uvh[Iv, :]**2)))
    enstrophy = 0.5*np.mean(q**2*ayp(axp(h)))

    return flux, energy, enstrophy


def set_mpi_bdr(uvh, rank, p, mx, col, tags):
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


def gather_uvh(uvh, uvhG, UVHG, mx, My, rank, p, ii):
    comm.Gather(uvh[:, 1:mx+1].flatten(), uvhG, root=0)
    if rank == 0:
        # evenly split ug into a list of p parts
        temp = np.array_split(uvhG, p)
        # reshape each part
        temp = [a.reshape(3*My, mx) for a in temp]
        UVHG[:, :, ii] = np.hstack(temp)
        return UVHG
    else:
        return None


def main():
    # mpi stuff
    rank = comm.Get_rank()   # this process' ID
    p = comm.Get_size()      # number of processors
    tags = dict([(j, j+5) for j in xrange(p)])

    # Number of grid points
    sc = 1
    Mx, My  = 128*sc, 128*sc
    mx = Mx / p  # x grid points per process

    # Grid Parameters
    Lx, Ly  = 200e3, 200e3
    Iu, Iv, Ih = range(0, My), range(My, 2*My), range(2*My, 3*My)

    # x conditions
    dx = Lx/Mx
    x0, xf = -Lx/2, Lx/2

    # handling ghost points
    x_all  = np.linspace(x0, xf-dx, Mx)            # all (non-staggered) x points
    xs_all = np.linspace(x0 +dx/2, xf-dx/2, Mx)    # all (staggered) x points
    if rank == 0:
        x  = np.hstack([ [x_all[-1]],  x_all[:mx+1] ])
        xs = np.hstack([ [xs_all[-1]], xs_all[:mx+1]])

    elif rank == p-1:
        x  = np.hstack([ x_all[rank*mx - 1:],  [x_all[0]]  ])
        xs = np.hstack([ xs_all[rank*mx - 1:], [xs_all[0]] ])

    else:
        x  = x_all [rank*mx - 1 : (rank+1)*mx + 1]
        xs = xs_all[rank*mx - 1 : (rank+1)*mx + 1]

    # y conditions
    dy = Ly/My
    y0, yf = -Ly/2, Ly/2
    y  = np.linspace(y0, yf-dy, My)
    ys = np.linspace(y0+dy/2, yf-dy/2, My)


    # Physical parameters
    f0, beta, gp, H0  = 1.e-4, 0e-11, 9.81, 500.

    # Temporal Parameters
    dt = 5./sc
    t0, tf = 0.0, 3600.0
    N  = int((tf - t0)/dt)
    t  = np.arange(N)*dt

    method = flux_sw_enst

    # Define Grid (staggered grid)
    xq, yq = np.meshgrid(xs, ys)
    xh, yh = np.meshgrid(x,  y)
    xu, yu = np.meshgrid(xs, y)
    xv, yv = np.meshgrid(x,  ys)

    # Modify class
    params = wavenum()
    params.dx   = dx
    params.dy   = dy
    params.f0   = f0
    params.beta = beta
    params.gp   = gp
    params.H0   = H0
    params.Mx   = mx
    params.My   = My

    # Initial Conditions with plot: u, v, h
    hmax = 1.e0
    uvh = np.vstack([0.*xu,
                     0.*xv,
                     hmax*np.exp(-(xh**2 + (1.0*yh)**2)/(Lx/6.0)**2)])

    # Define arrays to store conserved quantitites: energy and enstrophy
    energy, enstr = np.zeros(N), np.zeros(N)


    # GLOBAL STUFF
    if rank == 0:
        xG,  yG  = x_all, y
        xsG, ysG = xs_all, ys
        xqG, yqG = np.meshgrid(xsG, ysG)
        xhG, yhG = np.meshgrid(xG,  yG)
        xuG, yuG = np.meshgrid(xsG, yG)
        xvG, yvG = np.meshgrid(xG,  ysG)

        uvhG = np.vstack([0.*xuG,
                          0.*xvG,
                          hmax*np.exp(-(xhG**2 + (1.0*yhG)**2)/(Lx/6.0)**2)]).flatten()

        UVHG = np.empty((3*My, Mx, N), dtype='d')
        UVHG[:, :, 0] = uvhG.reshape(3*My, Mx)
    else:
        uvhG = None
        UVHG = None

        
    # allocate space to communicate the ghost columns
    col = np.empty(3*My, dtype='d')

    comm.Barrier()         # start MPI timer
    t_start = MPI.Wtime()


    # Euler step
    NLnm, energy[0], enstr[0] = method(uvh, params)
    uvh  = uvh + dt*NLnm
    # impose BCs on this iteration
    uvh  = set_mpi_bdr(uvh, rank, p, mx, col, tags)
    UVHG = gather_uvh(uvh, uvhG, UVHG, mx, My, rank, p, 1)      # add uvh to global soln


    # AB2 step
    NLn, energy[1], enstr[1] = method(uvh, params)
    uvh  = uvh + 0.5*dt*(3*NLn - NLnm)
    # impose BCs
    uvh  = set_mpi_bdr(uvh, rank, p, mx, col, tags)
    UVHG = gather_uvh(uvh, uvhG, UVHG, mx, My, rank, p, 2)      # add to global soln


    # loop through time
    for ii in range(3, N):
        print ii

        # AB3 step
        NL, energy[ii-1], enstr[ii-1] = method(uvh, params)
        uvh = uvh + dt/12*(23*NL - 16*NLn + 5*NLnm)

        # Reset fluxes
        NLnm, NLn = NLn, NL


        # impose BCs
        uvh  = set_mpi_bdr(uvh, rank, p, mx, col, tags)
        UVHG = gather_uvh(uvh, uvhG, UVHG, mx, My, rank, p, ii)


    comm.Barrier()
    t_final = (MPI.Wtime() - t_start)  # stop MPI timer

    print t_final


    # PLOTTING
    if rank == 0:
        fig = plt.figure()
        ims = []
        for j in xrange(N):
            ims.append((plt.pcolormesh(xhG/1e3, yhG/1e3, UVHG[Ih, :, j], norm=plt.Normalize(0, 1)), ))

        print 'done creating meshes, attempting to put them together...'
        im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)

        print 'saving...'
        im_ani.save('sadourny_mpi_%d.mp4' % p)
        print 'saved.'


    """
    print "Error in energy is ", np.amax(energy-energy[0])/energy[0]
    print "Error in enstrophy is ", np.amax(enstr-enstr[0])/enstr[0]

    fig, axarr = plt.subplots(2, sharex=True)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.plot((energy-energy[0])/energy[0], '-ob', linewidth=2, label='Energy')
    ax1.set_title('Energy')
    ax2.plot((enstr-enstr[0])/enstr[0], '-or',  linewidth=2, label='Enstrophy')
    ax2.set_title('Enstrophy')
    plt.show()
    """

main()
