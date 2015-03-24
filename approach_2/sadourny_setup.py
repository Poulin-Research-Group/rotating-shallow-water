import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import time
import os
from mpi4py import MPI
from flux_sw_ener import euler_f as ener_Euler_f, ab2_f as ener_AB2_f, ab3_f as ener_AB3_f
from flux_sw_ener90 import euler_f as ener_Euler_f90, ab2_f as ener_AB2_f90, ab3_f as ener_AB3_f90
comm = MPI.COMM_WORLD


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


def periodic(f, params):
    # Nx = params.Nx
    # Ny = params.Ny
    Ny, Nx = np.array(f.shape) - 2

    # Impose periodic BCs
    f[0,:]    = f[Ny,:]
    f[Ny+1,:] = f[1,:]
    f[:,0]    = f[:,Nx]
    f[:,Nx+1] = f[:,1]

    return f


def odd(f, params):
    Nx = params.Nx
    Ny = params.Ny

    # Impose periodic BCs
    f[0,:]    = -f[1,:]
    f[Ny+1,:] = -f[Ny,:]
    f[:,0]    =  f[:,Nx]
    f[:,Nx+1] =  f[:,1]

    return f


def even(f, params):
    Nx = params.Nx
    Ny = params.Ny

    # Impose periodic BCs
    f[0,:]    = f[1,:]
    f[Ny+1,:] = f[Ny,:]
    f[:,0]    = f[:,Nx]
    f[:,Nx+1] = f[:,1]

    return f


def euler(uvh, dt, NLnm):
    return uvh[:, 1:-1, 1:-1] + dt*NLnm


def ab2(uvh, dt, NLn, NLnm):
    return uvh[:, 1:-1, 1:-1] + 0.5*dt*(3*NLn - NLnm)


def ab3(uvh, dt, NL, NLn, NLnm):
    return uvh[:, 1:-1, 1:-1] + dt/12*(23*NL - 16*NLn + 5*NLnm)


def flux_sw_ener(uvh, params):

    # Define parameters
    dx,dy     = params.dx, params.dy
    gp,f0,H0  = params.gp, params.f0, params.H0
    Nx,Ny,Nz  = params.Nx, params.Ny, params.Nz

    # Pull out primitive variables
    u,v,h = uvh[0,:,:], uvh[1,:,:],  H0 + uvh[2,:,:]

    # Initialize fields
    U,V  = np.zeros((Ny+2,Nx+2),float), np.zeros((Ny+2,Nx+2),float)
    B,q  = np.zeros((Ny+2,Nx+2),float), np.zeros((Ny+2,Nx+2),float)
    flux = np.zeros((3,Ny,Nx),float)
    
    # Compute U, V, B, q
    U[1:-1, 1:-1] = 0.5*(I(h) + I_XP(h)) * I(u)   
    V[1:-1, 1:-1] = 0.5*(I(h) + I_YP(h)) * I(v)
    B[1:-1, 1:-1] = gp*I(h) + 0.25* (I(u)**2 + I_XN(u)**2 + I(v)**2 + I_YN(v)**2)
    q[1:-1, 1:-1] = 4*((I_XP(v) - I(v)) / dx - (I_YP(u) - I(u)) / dy + f0) /   \
                       (I(h) + I_YP(h) + I_XP(h) + I_XP_YP(h))

    # Enforce BCs
    U = periodic(U,params)
    V = periodic(V,params)
    B = periodic(B,params)
    q = periodic(q,params)
    #U = even(U,params)
    #V = odd(V,params)
    #B = even(B,params)
    #q = even(q,params)

    # Compute fluxes
    flux[0, :, :] = 0.25* (I(q) * (I_XP(V) + I(V)) + I_YN(q) * (I_XP_YN(V) + I_YN(V))) - \
                    (I_XP(B) - I(B))/dx
    flux[1, :, :] = -0.25*(I(q) * (I_YP(U) + I(U)) + I_XN(q) * (I_XN_YP(U) + I_XN(U))) - \
                    (I_YP(B) - I(B))/dy
    flux[2,:,:] = -(U[1:-1,1:-1] - U[1:-1,0:-2])/dx - (V[1:-1,1:-1] - V[0:-2,1:-1])/dy

    #compute energy and enstrophy
    energy = 0.5*np.mean( gp*I(h)**2 + 0.5*I(h)*(I(u)**2 + I_XN(u)**2 + I(v)**2 + I_YN(v)**2))
    enstrophy = 0.125*np.mean((I(h) + I_YP(h) + I_XP(h) + I_XP_YP(h)) * I(q)**2)

    return flux, energy, enstrophy


def ener_Euler(uvh, params):
    NLnm, energy, enstr = flux_sw_ener(uvh, params)
    uvh[:, 1:-1, 1:-1]  = euler(uvh, params.dt, NLnm)
    return uvh, NLnm, energy, enstr


def ener_AB2(uvh, NLnm, params):
    NLn, energy, enstr = flux_sw_ener(uvh, params)
    uvh[:, 1:-1, 1:-1] = ab2(uvh, params.dt, NLn, NLnm)
    return uvh, NLn, energy, enstr


def ener_AB3(uvh, NLn, NLnm, params):
    NL, energy, enstr  = flux_sw_ener(uvh, params)
    uvh[:, 1:-1, 1:-1] = ab3(uvh, params.dt, NL, NLn, NLnm)
    return uvh, NL, energy, enstr


def set_mpi_bdr(uvh, rank, p, nx, uCol, vCol, hCol, uTags, vTags, hTags):
    # send uvh[j, :, 1] (second column) ... j = 0,1,2
    if 0 < rank:
        # ...from rank to rank-1 (left)
        comm.Send(uvh[0, :, 1].flatten(), dest=rank-1, tag=uTags[rank])
        comm.Send(uvh[1, :, 1].flatten(), dest=rank-1, tag=vTags[rank])
        comm.Send(uvh[2, :, 1].flatten(), dest=rank-1, tag=hTags[rank])
    else:
        # ...from 0 to p-1
        comm.Send(uvh[0, :, 1].flatten(), dest=p-1,    tag=uTags[rank])
        comm.Send(uvh[1, :, 1].flatten(), dest=p-1,    tag=vTags[rank])
        comm.Send(uvh[2, :, 1].flatten(), dest=p-1,    tag=hTags[rank])

    # receive uvh[:, 1], place in uvh[:, nx+1] (last column) ...
    if rank < p-1:
        # ... from rank+1
        comm.Recv(uCol, source=rank+1, tag=uTags[rank+1])
        comm.Recv(vCol, source=rank+1, tag=vTags[rank+1])
        comm.Recv(hCol, source=rank+1, tag=hTags[rank+1])
        uvh[0, :, -1] = uCol
        uvh[1, :, -1] = vCol
        uvh[2, :, -1] = hCol
    else:
        # ... from rank 0
        comm.Recv(uCol, source=0, tag=uTags[0])
        comm.Recv(vCol, source=0, tag=vTags[0])
        comm.Recv(hCol, source=0, tag=hTags[0])
        uvh[0, :, -1] = uCol
        uvh[1, :, -1] = vCol
        uvh[2, :, -1] = hCol

    # send uvh[:, nx] (second-last column)...
    if rank < p-1:
        # ...from rank to rank+1
        comm.Send(uvh[0, :, -2].flatten(), dest=rank+1, tag=uTags[rank])
        comm.Send(uvh[1, :, -2].flatten(), dest=rank+1, tag=vTags[rank])
        comm.Send(uvh[2, :, -2].flatten(), dest=rank+1, tag=hTags[rank])
    else:
        # ...from p-1 to 0
        comm.Send(uvh[0, :, -2].flatten(), dest=0, tag=uTags[rank])
        comm.Send(uvh[1, :, -2].flatten(), dest=0, tag=vTags[rank])
        comm.Send(uvh[2, :, -2].flatten(), dest=0, tag=hTags[rank])

    # receive uvh[:, nx], place in uvh[:, 0] (first column) ...
    if 0 < rank:
        # ...from rank-1
        comm.Recv(uCol, source=rank-1, tag=uTags[rank-1])
        comm.Recv(vCol, source=rank-1, tag=vTags[rank-1])
        comm.Recv(hCol, source=rank-1, tag=hTags[rank-1])
        uvh[0, :, 0] = uCol
        uvh[1, :, 0] = vCol
        uvh[2, :, 0] = hCol
    else:
        # ...from p-1
        comm.Recv(uCol, source=p-1, tag=uTags[p-1])
        comm.Recv(vCol, source=p-1, tag=vTags[p-1])
        comm.Recv(hCol, source=p-1, tag=hTags[p-1])
        uvh[0, :, 0] = uCol
        uvh[1, :, 0] = vCol
        uvh[2, :, 0] = hCol

    # Impose periodic BCs

    for jj in range(3):
        uvh[jj, 0,  :] = uvh[jj, -2, :]
        uvh[jj, -1, :] = uvh[jj,  1, :]
    # f[:,0]    = f[:, nx]
    # f[:,nx+1] = f[:,  1]

    return uvh


def gather_uvh(uvh, uvhG, UVHG, rank, p, nx, Ny, n):
    """
    Gather each process' slice of uvh and store it in the global set of
    solutions. Gathering is done on process 0.

    Parameters
    ----------
    uvh : ndarray
        process' slice of the solution (dimensions: (Ny, nx+2))
    uvhG : ndarray
        the global solution, excluding ghost points (dimensions: (Ny, p*nx))
    UVHG : ndarray
        the set of global solutions (dimensions: (Ny, p*nx, N)) where N is the
        number of time steps defined elsewhere.
    rank : int
        rank of this process
    p : int
        number of processes
    nx : int
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
    comm.Gather(uvh[:, 1:-1, 1:-1].flatten(), uvhG, root=0)
    if rank == 0:
        # evenly split ug into a list of p parts
        temp = np.array_split(uvhG, p)
        # reshape each part
        temp = [a.reshape(3, Ny, nx) for a in temp]

        UVHG[:, :, :, n] = np.dstack(temp)
        return UVHG
    else:
        return None


class Params(object):
    """
    See approach_1/sadoury_setup.py for doc.
    """
    def __init__(self, dx, dy, f0, gp, H0, Nx, Ny, Nz, dt):
        super(Params, self).__init__()
        self.dx   = dx
        self.dy   = dy
        self.f0   = f0
        self.gp   = gp
        self.H0   = H0
        self.Nx   = Nx
        self.Ny   = Ny
        self.Nz   = Nz
        self.dt   = dt


def create_global_objects(rank, xG, yG, dx, hmax, Lx, N):
    if rank == 0:
        xx, yy = np.meshgrid(xG, yG)
        xuG = xx
        xvG = xx - dx/2
        xhG = xx - dx/2
        yhG = yy

        Nx = len(xG)
        Ny = len(yG)

        uvhG = np.zeros([3, Ny, Nx])                # global initial solution
        UVHG = np.empty((3, Ny, Nx, N), dtype='d')  # set of ALL global solutions

        uvhG[0, :, :] = 0*xuG
        uvhG[1, :, :] = 0*xvG
        uvhG[2, :, :] = hmax*np.exp(-(xhG**2 + yhG**2)/(Lx/6.0)**2)
        UVHG[:, :, :, 0] = uvhG
        
        uvhG = uvhG.flatten()

    else:
        uvhG = None
        UVHG = None

    return (uvhG, UVHG)


def create_x_points(rank, p, xG, xsG, nx):
    # TODO: update this.
    x = np.array_split(xG, p)[rank]
    xs = np.array([])
    return (x, xs)


def PLOTTO_649(UVHG, xG, yG, Nt, output_name, MPI):
    xhG, yhG = np.meshgrid(xG,  yG)

    fig = plt.figure()
    ims = []
    if not MPI:
        for n in xrange(Nt):
            ims.append((plt.pcolormesh(xhG/1e3, yhG/1e3, UVHG[2, 1:-1, 1:-1, n], 
                norm=plt.Normalize(0, 1)), ))
    else:
        for n in xrange(Nt):
            ims.append((plt.pcolormesh(xhG/1e3, yhG/1e3, UVHG[2, :, :, n], 
                norm=plt.Normalize(0, 1)), ))

    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)
    im_ani.save(output_name)


def writer(t_final, method, sc):
    filename = './tests/%s/sc-%d.txt' % (method, sc)

    # check to see if file exists; if it doesn't, create it.
    if not os.path.exists(filename):
        open(filename, 'a').close()

    # write time to the file
    F = open(filename, 'a')
    F.write('%f\n' % t_final)
    F.close()
