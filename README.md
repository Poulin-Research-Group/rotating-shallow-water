# Rotating Shallow Water equations
The code in this repository solves the rotating shallow water (RSW) equations using pure Numpy methods and a combination of Numpy and Fortran (using f2py). There are currently two approaches to solving the equations:

## Approach 1
"Directly"; in Numpy using the `roll` method to add terms at differing grid points, and writing this manually in Fortran (which leads to very awful Fortran 77 code).

**NOTE** - the code for this method is deprecated and it is unlikely that any more work will be done on it. See approach 2 instead.

## Approach 2
Padding the solution matrix with two extra rows (one on top, one on bottom) and two extra columns (one on left, one on right), which removes the need for `roll`.

## Compiling
Before running the Python code, the Fortran code must be compiled using f2py. This is done using the following commands:

```
f2py -c -m flux_ener_f2py77 flux_ener_f2py.f
f2py --f90flags=-ffixed-line-length-0 -c -m flux_ener_f2py90 flux_ener_f2py.f90
```

The Fortran 90 code MUST be compiled with the `--f90flags=-ffixed-line-length-0` option, as the line length tends to go over gfortran's default limit.

To optimize the code, you can tell f2py to use optimization flags that are available in the gfortran compiler, such as `O3`:

```
f2py --opt=-O3 -c -m flux_ener_f2py77 flux_ener_f2py.f
f2py --opt=-O3 --f90flags=-ffixed-line-length-0 -c -m flux_ener_f2py90 flux_ener_f2py.f90
```

To optimize using `Ofast`, your system's stack limit must be changed. The exact value needed is unknown, so setting it to unlimited works. To see what your current stack limit is, run

```
ulimit -s
```

To change it to unlimited, run

```
ulimit -s unlimited
```

Now you can compile using the `Ofast` flag like so:

```
f2py --opt=-Ofast -c -m flux_ener_f2py77 flux_ener_f2py.f
f2py --opt=-Ofast --f90flags=-ffixed-line-length-0 -c -m flux_ener_f2py90 flux_ener_f2py.f90
```

## Running the code
**NOTE**: This only applies to approach 2.

The script to run is `main.py`. Inside this script, several parameters can be changed, although there is still some work to be done. (See the TODO below.) Command line arguments can be accepted. Currently, the MPI'd version of this approach can only run the Numpy solver.

For the serial case, command line arguments can be passed like `python main.py method px py sc_x sc_y`, where:

- `method` = one of: 'numpy', 'f2py77', 'f2py90', 'hybrid77', 'hybrid90'  (without quotes)
- `px` = the number of processes to use in the x-direction
- `py` = the number of processes to use in the y-direction
- `sc_x` = the scaling factor in the x-direction
- `sc_y` = the scaling factor in the y-direction

For the parallel case, command line arguments can be passed like `mpirun -np p method px py sc_x sc_y`, where `p` is the number of processes used overall (i.e. `px*py`). The only method available is 'numpy' for the parallel case.

## Running the tests
**NOTE**: This only applies to approach 2.

Running the command

```
bash run_tests.sh
```

will create a `tests` directory and other directories in the approach's directory, and then Numpy, (f2py) F77, (f2py) F90, and the two hybrid implementations of the solver will be run in serial for square grid sizes of 128x128 and 256x256, 10 times. 

Also, the Numpy implementation of the solver will be run in parallel for core layouts 2x1, 1x2, 2x2, 4x1, and 1x4, for the same grid sizes, 10 times.


## TODO
- Write Fortran 77 and Fortran 90 code for approach 2 - this is a bit different from the serial code, as the components `U, V, B, q` must make use of MPI, so the components will have to be calculated in Fortran, passed back to Python and then MPI'd in Python
- Write the solver functions that save the solution at every time step for approach 2, in both serial and parallel
- Write PURE Fortran 77 and Fortran 90 code for approach 1 (this probably won't be done)
- Add test script to Approach 1 dir (this probably won't be done)
- Write Fortran 90 code for Approach 1 (this probably won't be done)
