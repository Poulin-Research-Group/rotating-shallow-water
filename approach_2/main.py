from __future__ import division
import sys
import numpy as np
from sadourny_setup import solver, Params


# TODO ==============
# find a way to make an initial condition function that can take an arbitrary
# number of arguments given by the user.
def f(x, y, params):
    dx = params.dx
    nx, ny = params.nx, params.ny

    xx, yy = np.meshgrid(x, y)
    xu = xx
    xv = xx - dx/2
    xh = xx - dx/2
    yh = yy

    hmax = params.hmax
    Lx   = params.Lx

    uvh = np.zeros([3, ny+2, nx+2])
    uvh[0, :, :] = 0*xu
    uvh[1, :, :] = 0*xv
    uvh[2, :, :] = hmax*np.exp(-(xh**2 + yh**2)/(Lx/6.0)**2)
    return uvh


# boundary condition (BC) functions; four functions imposing BCs must be
# defined: one for serial solutions, and then one each for solutions
# parallelized in only x, only y, and both x and y.
# TODO ==============
# add some BC functions...

# handling command line arguments, e.g.
#
#   python stepping_mpi_2D.py numpy 1 1 2 2
#
# will run this script using numpy to calculate the next solution, px = 1,
# py = 1, sc_x = 2, sc_y = 2

if len(sys.argv) > 1:
    argv = sys.argv[1:]
    method = argv[0]
    px   = int(argv[1])
    py   = int(argv[2])
    sc_x = int(argv[3])
    sc_y = int(argv[4])

# if there are no command line arguments...
else:
    # number of processors to use in each direction
    px = 1
    py = 1

    # scaling parameters
    sc_x = 2
    sc_y = 2

    # method to use; options are 'numpy', 'f2py77', 'f2py90'
    method = 'numpy'


# lengths of axes
Lx = 200e3
Ly = 200e3

# number of spatial points
Nx = 128*sc_x
Ny = 128*sc_y

# number of (x, y) spatial points per processes dedicated to respective directions
nx = Nx/px
ny = Ny/py

# x conditions
x0 = -Lx/2        # start
xf = Lx/2         # end
dx = Lx/Nx        # spatial step size

# y conditions
y0 = -Ly/2
yf = Ly/2
dy = Lx/Ny

# temporal conditions
t0, tf = 0.0, 3600.0
dt = 5./sc_x               # TODO: change depency on sc_x ???
Nt  = int((tf - t0)/dt)

# some constants
f0 = 1e-4
gp = 9.81
H0 = 500

# used for the global objects / ICs...
hmax = 1.0
Lx   = 200e3

# Define type of BCs ('P' = periodic)
BC_type = 'P'
BC_s  = None
BC_x  = None
BC_y  = None
BC_xy = None

# if SAVE_TIME is True, then the total time to solve the problem will be saved
# to a file named filename_time
SAVE_TIME = False
filename_time = './tests/%s/%dscx_%dscy_%dpx_%dpy.txt' % (method, sc_x, sc_y, px, py)

# if ANIMATE is True, then an animation of the solution will be saved to a file
# named filename_anim
ANIMATE = True
filename_anim = './anims/anim_%dpx_%dpy.mp4' % (px, py)

# if SAVE_SOLN is True, then the solution at every time step will be saved to a
# file named filename_soln
SAVE_SOLN = False
filename_soln = './solns/soln_%dpx_%dpy.txt' % (px, py)

# DO NOT ALTER =======================================
params = Params()
params.set_x_vars([x0, xf, dx, Nx, nx])
params.set_y_vars([y0, yf, dy, Ny, ny])
params.set_t_vars([t0, tf, dt, Nt])
params.set_consts([f0, gp, H0])
params.set_bc_funcs([BC_s, BC_x, BC_y, BC_xy])
params.ics = f
params.hmax, params.Lx = hmax, Lx
params.bcs_type = BC_type
params.filename_time = filename_time
params.filename_anim = filename_anim
params.filename_soln = filename_soln
params.method = method

solver(params, px, py, SAVE_TIME, ANIMATE, SAVE_SOLN)
