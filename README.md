# Rotating Shallow Water equations
The code in this repository solves the rotating shallow water (RSW) equations using pure Numpy methods and a combination of Numpy and Fortran (using f2py). All solution methods work in both serial and parallel, using mpi4py. That being said, mpi4py must be installed to run the parallel code, and f2py must be installed to compile the Fortran code.

Before running the Python code, the Fortran code must be compiled using f2py. This is done using the following commands:

```
f2py -c -m flux_ener_components flux_ener_components.f
f2py -c -m flux_sw_ener flux_sw_ener.f
```

## Running the code
To test out the different methods in either `sadourny.py` or `sadourny_mpi.py` (**TODO**: MPI code is not working yet for updated version), call the `main` function using the different flux calculator & time steppers, and `sc` (scale) value. The methods that can be passed are:

- `ener_Euler`, `ener_Euler_f`: Numpy and Fortran implementations of Sadourny's first method (energy conserving) using an Euler time stepping method
- `ener_AB2`, `ener_AB2_f`: Numpy and Fortran implementations of Sadourny's first method using a second-order Adams-Bashforth time stepping method
- `ener_AB3`, `ener_AB3_f`: Numpy and Fortran implementations of Sadourny's first method using a third-order Adams-Bashforth time stepping method

Running `sadourny.py` as is will run the following:

```python
# using Numpy
main(ener_Euler, ener_AB2, ener_AB3, 1)

# using Fortran
main(ener_Euler_f, ener_AB2_f, ener_AB3_f, 1)
```

Alter whatever you want.

**Note**: these methods cannot be mixed and matched yet, but that will be hopefully made available.