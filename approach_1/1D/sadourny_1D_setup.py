from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import time
from mpi4py import MPI
from flux_ener import euler_f as ener_Euler_f, ab2_f as ener_AB2_f, ab3_f as ener_AB3_f
comm = MPI.COMM_WORLD


# delta_x (positive direction)
def dxp(f, dx):
    fx = (np.roll(f, -1, 0) - f)/dx
    return fx


# delta_x (negative direction)
def dxm(f, dx):
    fx = (f - np.roll(f, 1, 0))/dx
    return fx


# Define averaging functions
# average in x (positive direction)
def axp(f):
    afx = 0.5*(np.roll(f, -1, 0) + f)
    return afx


# average in x (negative direction)
def axm(f):
    afx = 0.5*(f + np.roll(f, 1, 0))
    return afx


def euler(uvh, dt, NLnm):
    # euler stepper
    return uvh + dt*NLnm


def ab2(uvh, dt, NLn, NLnm):
    # ab2 stepper
    return uvh + 0.5*dt*(3*NLn - NLnm)


def ab3(uvh, dt, NL, NLn, NLnm):
    # ab3 stepper
    return uvh + dt/12*(23*NL - 16*NLn + 5*NLnm)


# Define flux for Sadourny's energy conserving scheme
def flux_sw_ener(uvh, parms):

    # Define parameters
    dx = parms.dx
    gp = parms.gp
    f0 = parms.f0
    H0 = parms.H0

    # Compute Fields
    h = H0  +  uvh[2, :]
    U = axp(h)*uvh[0, :]
    V = h*uvh[1, :]
    B = gp*h + 0.5*(axm(uvh[0, :]**2) + uvh[1, :]**2)
    q = (dxp(uvh[1, :], dx) + f0)/axp(h)

    # Compute fluxes
    flux = np.vstack([q*axp(V) - dxp(B, dx), -axm(q*U), -dxm(U, dx)])

    # compute energy and enstrophy
    energy = 0.5*np.mean(gp*h**2 + h*(axm(uvh[0, :]**2) + uvh[1, :]**2))
    enstrophy = 0.5*np.mean(q**2*axp(h))

    return flux, energy, enstrophy


# Define flux for Sadourny's enstrophy conserving scheme
def flux_sw_enst(uvh, parms):

    # Define parameters
    dx = parms.dx
    gp = parms.gp
    f0 = parms.f0
    H0 = parms.H0

    h = H0 + uvh[2, :]
    U = axp(h)*uvh[0, :]
    V = h*uvh[1, :]
    B = gp*h + 0.5*(axm(uvh[0, :]**2) + uvh[1, :]**2)
    q = (dxp(uvh[1, :], dx) + f0)/axp(h)

    # Compute fluxes
    flux = np.vstack([q*axp(V) - dxp(B, dx), -axm(q)*axm(U), -dxm(U, dx)])

    # compute energy and enstrophy
    energy = 0.5*np.mean(gp*h**2 + h*(axm(uvh[0, :]**2) + uvh[1, :]**2))
    enstrophy = 0.5*np.mean(q**2*axp(h))

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


class wavenum(object):
    """
    See the 2D sadourny setup for the documentation.
    """
    def __init__(self, dx, f0, beta, gp, H0, Nx, dt):
        super(wavenum, self).__init__()
        self.dx   = dx
        self.f0   = f0
        self.beta = beta
        self.gp   = gp
        self.H0   = H0
        self.Nx   = Nx
        self.dt   = dt
