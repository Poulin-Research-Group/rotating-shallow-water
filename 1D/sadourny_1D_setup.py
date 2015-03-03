from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import time
from mpi4py import MPI
from flux_ener import flux_ener as flux_ener_F
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


class wavenum(object):
    """
    See the 2D sadourny setup for the documentation.
    """
    def __init__(self, dx, f0, beta, gp, H0, Mx):
        super(wavenum, self).__init__()
        self.dx   = dx
        self.f0   = f0
        self.beta = beta
        self.gp   = gp
        self.H0   = H0
        self.Mx   = Mx
