# Rotating Shallow Water equations
The code in this repository solves the rotating shallow water (RSW) equations using pure Numpy methods and a combination of Numpy and Fortran (using f2py). There are currently two approaches to solving the equations:

## Approach 1
"Directly"; in Numpy using the `roll` method to add terms at differing grid points, and writing this manually in Fortran (which leads to very awful Fortran 77 code).

## Approach 2
Padding the solution matrix with two extra rows (one on top, one on bottom) and two extra columns (one on left, one on right), which removes the need for `roll`.

## Compiling
Before running the Python code, the Fortran code must be compiled using f2py. This is done using the following commands:

```
f2py -c -m flux_sw_ener flux_sw_ener.f
f2py --f90flags=-ffixed-line-length-0 -c -m  flux_sw_ener90 flux_sw_ener.f90
```

The Fortran 90 code MUST be compiled with the `--f90flags=-ffixed-line-length-0` option, as the line length tends to go over gfortran's default limit.

To optimize the code, you can tell f2py to use optimization flags that are available in the gfortran compiler, such as `O3`:

```
f2py --opt=-O3 -c -m flux_sw_ener flux_sw_ener.f
f2py --opt=-O3 --f90flags=-ffixed-line-length-0 -c -m  flux_sw_ener90 flux_sw_ener.f90
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
f2py --opt=-Ofast -c -m flux_sw_ener flux_sw_ener.f
f2py --opt=-Ofast --f90flags=-ffixed-line-length-0 -c -m  flux_sw_ener90 flux_sw_ener.f90
```

## Running the code
To test out the different methods in either `sadourny.py` or `sadourny_mpi.py`, call the `main` function using the different flux calculator & time steppers, and `sc` (scale) value. Some of the methods that can be passed are:

- `ener_Euler`, `ener_Euler_f`, `ener_Euler_f90`: Numpy, (f2py) Fortran 77 and (f2py) Fortran 90 implementations of Sadourny's first method (energy conserving) using an Euler time stepping method
- `ener_AB2`, `ener_AB2_f`, `ener_AB2_f90`: Numpy, (f2py) Fortran 77 and (f2py) Fortran 90 implementations of Sadourny's first method using a second-order Adams-Bashforth time stepping method
- `ener_AB3`, `ener_AB3_f`, `ener_AB3_f90`: Numpy, (f2py) Fortran 77 and (f2py) Fortran 90 implementations of Sadourny's first method using a third-order Adams-Bashforth time stepping method

Others are:

- `ener_Euler_hybrid77`, `ener_Euler_hybrid90`: (f2py) Fortran 77 and (f2py) Fortran 90 implementations of Sadourny's first method, using a Numpy implementation of the Euler time stepping method
- `ener_AB2_hybrid77`, `ener_AB2_hybrid90`: (f2py) Fortran 77 and (f2py) Fortran 90 implementations of Sadourny's first method, using a Numpy implementation of the second-order Adams-Bashforth time stepping method
- `ener_AB3_hybrid77`, `ener_AB3_hybrid90`: (f2py) Fortran 77 and (f2py) Fortran 90 implementations of Sadourny's first method, using a Numpy implementation of the third-order Adams-Bashforth time stepping method

**NOTE**: (f2py) Fortran 90 and the hybrid methods are not available for Approach 1 yet.

For example, in `approach_2`, add the following lines of code to the bottom of either `sadourny.py` or `sadourny_mpi.py` to solve the RSW equations with Numpy and (f2py) Fortran 77 with a grid size of 128-by-128:

```python
# using Numpy
main(ener_Euler, ener_AB2, ener_AB3, 1)

# using (f2py) Fortran 77
main(ener_Euler_f, ener_AB2_f, ener_AB3_f, 1)
```

To run the MPI code, `sadourny_mpi.py`, the terminal must be used, e.g. running with two cores:

```
mpirun -np 2 python sadourny_mpi.py
```

## Running the tests
**NOTE**: This is currently only available for approach 2.

Running the command

```
bash run_tests.sh
```

will create a `tests` directory and other directories in the approach's directory, and then Numpy, (f2py) F77 and (f2py) F90 implementations of the solver will be run. This will run 10 trials for grid sizes 128-by-128, 256-by-256, and 512-by-512, and 2 trials for grid sizes 1024-by-1024 and 2048-by-2048.


## TODO
- Write Fortran 90 code for Approach 1
- Write PURE Fortran 77 and Fortran 90 code for both approaches
- Add test script to Approach 1 dir
