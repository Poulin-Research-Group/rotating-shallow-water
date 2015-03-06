# Rotating Shallow Water equations
The code in this repository solves the rotating shallow water (RSW) equations using pure Numpy methods and a combination of Numpy and Fortran (using f2py). All solution methods work in both serial and parallel, using mpi4py. That being said, mpi4py must be installed to run the parallel code, and f2py must be installed to compile the Fortran code.

Before running the Python code, the Fortran code must be compiled using f2py. This is done using the following commands:

```
f2py -c -m flux_ener_components flux_ener_components.f
f2py -c -m flux_sw_ener flux_sw_ener.f
```

## Running the code
To test out the different methods in either `sadourny.py` or `sadourny_mpi.py`, call the `main` function using the different method names and `sc` (scale) value. The method names that can be passed to `main` are:

- `flux_sw_ener`: Sadourny's first method (energy conserving)
- `flux_sw_enst`: Sadourny's second method (enstrophy conserving)
- `flux_sw_ener_Fcomp`: the first version of the Fortran implementation of Sadourny's first method, where each variable `h, U, V, ...` are calculated using different subroutines (`calc_h, calc_U, calc_V, ...`) imported from the Fortran file
- `flux_ener_f`: the second version of the Fortran implementation of Sadourny's first method, where everything is calculated in one subroutine (that calls upon other subroutines)