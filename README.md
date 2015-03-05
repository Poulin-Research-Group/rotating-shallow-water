# Rotating Shallow Water equations
The code in this repository solves the rotating shallow water (RSW) equations using pure Numpy methods and a combination of Numpy and Fortran (using f2py). All solution methods work in both serial and parallel, using mpi4py. That being said, mpi4py must be installed to run the parallel code, and f2py must be installed to compile the Fortran code.

Before running the Python code, the Fortran code must be compiled using f2py. This is done using the following commands:

```
f2py -c -m flux_ener_components flux_ener_components.f
f2py -c -m flux_sw_ener flux_sw_ener.f
```