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
To test out the different methods in either `sadourny.py` or `sadourny_mpi.py`, call the `main` function using the different flux calculator & time steppers, and `sc` (scale) value. The methods that can be passed are:

- `ener_Euler`, `ener_Euler_f`, `ener_Euler_f90`: Numpy, Fortran 77 and Fortran 90 implementations of Sadourny's first method (energy conserving) using an Euler time stepping method
- `ener_AB2`, `ener_AB2_f`, `ener_AB2_f90`: Numpy, Fortran 77 and Fortran 90 implementations of Sadourny's first method using a second-order Adams-Bashforth time stepping method
- `ener_AB3`, `ener_AB3_f`, `ener_AB3_f90`: Numpy, Fortran 77 and Fortran 90 implementations of Sadourny's first method using a third-order Adams-Bashforth time stepping method

**NOTE**: Fortran 90 is not available for Approach 1 yet.

Running `sadourny.py` as is will run the following:

```python
# using Numpy
main(ener_Euler, ener_AB2, ener_AB3, 1)

# using Fortran 77
main(ener_Euler_f, ener_AB2_f, ener_AB3_f, 1)
```

As with most (all?) MPI code, `sadourny_mpi.py` must be run from the terminal, e.g. running with two cores:

```
mpirun -np 2 python sadourny_mpi.py
```

This will run the Fortran 77 implementation.


## TODO
- Write Fortran 90 code for Approach 1
- MPI-ify Approach 2
