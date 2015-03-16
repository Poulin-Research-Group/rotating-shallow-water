import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import time
from mpi4py import MPI
from flux_sw_ener import euler_f as ener_Euler_f, ab2_f as ener_AB2_f, ab3_f as ener_AB3_f
comm = MPI.COMM_WORLD


def periodic(f, params):
    # Nx = params.Nx
    # Ny = params.Ny
    Nx, Ny = np.array(f.shape) - 2

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
    U[1:-1,1:-1] = 0.5*(h[1:-1,1:-1]+h[1:-1,2:])*u[1:-1,1:-1]   
    V[1:-1,1:-1] = 0.5*(h[1:-1,1:-1]+h[2:,1:-1])*v[1:-1,1:-1]
    B[1:-1,1:-1] = gp*h[1:-1,1:-1] + 0.25*(u[1:-1,1:-1]**2 + u[1:-1,0:-2]**2 + v[1:-1,1:-1]**2 + v[0:-2,1:-1]**2)
    q[1:-1,1:-1] = 4*((v[1:-1,2:]-v[1:-1,1:-1])/dx - (u[2:,1:-1]-u[1:-1,1:-1])/dy + f0)/(h[1:-1,1:-1]+h[2:,1:-1]+h[1:-1,2:]+h[2:,2:])

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
    flux[0,:,:] =  0.25*(q[1:-1,1:-1]*(V[1:-1,2:] + V[1:-1,1:-1]) + q[0:-2,1:-1]*(V[0:-2,2:] + V[0:-2,1:-1])) - (B[1:-1,2:]-B[1:-1,1:-1])/dx
    flux[1,:,:] = -0.25*(q[1:-1,1:-1]*(U[2:,1:-1] + U[1:-1,1:-1]) + q[1:-1,0:-2]*(U[2:,0:-2] + U[1:-1,0:-2])) - (B[2:,1:-1]-B[1:-1,1:-1])/dy
    flux[2,:,:] = -(U[1:-1,1:-1] - U[1:-1,0:-2])/dx - (V[1:-1,1:-1] - V[0:-2,1:-1])/dy

    #compute energy and enstrophy
    energy = 0.5*np.mean( gp*h[1:-1,1:-1]**2 + 0.5*h[1:-1,1:-1]*(u[1:-1,1:-1]**2 + u[1:-1,0:-2]**2 + v[1:-1,1:-1]**2 + v[0:-2,1:-1]**2))
    enstrophy = 0.125*np.mean((h[1:-1,1:-1] + h[2:,1:-1] + h[1:-1,2:] + h[2:,2:])*q[1:-1,1:-1]**2)

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


def PLOTTO_649(UVHG, xG, yG, Nt, output_name):
    xhG, yhG = np.meshgrid(xG,  yG)

    fig = plt.figure()
    ims = []
    for n in xrange(Nt):
        ims.append((plt.pcolormesh(xhG/1e3, yhG/1e3, UVHG[2, 1:-1, 1:-1, n], norm=plt.Normalize(0, 1)), ))

    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=False)
    im_ani.save(output_name)
