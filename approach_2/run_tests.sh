# make test directory if it doesn't exist
if [ ! -d "tests" ]; then
  echo Creating test directories...
  mkdir -p tests/numpy tests/f2py77 tests/f2py90
fi

# retrieve old stack limit, set new stack limit to unlimited
old_stack_lim=$(ulimit -s)
echo Current stack limit is $old_stack_lim. Setting it to unlimited.
ulimit -s unlimited


function test_python () {
  px="$1"
  py="$2"
  sc_x="$3"
  sc_y="$4"
  T="$5"
  p=$(expr $px \* $py)

  echo px = $px, py = $py
  echo ------------------
  echo numpy
  for ((i=0; i<$T; i++)) do
    mpirun -np $p python main.py numpy $px $py $sc_x $sc_y
  done
  echo
}


function test_f2py77 () {
  px="$1"
  py="$2"
  sc_x="$3"
  sc_y="$4"
  T="$5"
  p=$(expr $px \* $py)

  echo f2py77
  for ((i=0; i<$T; i++)) do
    mpirun -np $p python main.py f2py77 $px $py $sc_x $sc_y
  done
  echo
}


function test_f2py90 () {
  px="$1"
  py="$2"
  sc_x="$3"
  sc_y="$4"
  T="$5"
  p=$(expr $px \* $py)

  echo f2py90
  for ((i=0; i<$T; i++)) do
    mpirun -np $p python main.py f2py90 $px $py $sc_x $sc_y
  done
  echo
}


function test_hybrid77 () {
  px="$1"
  py="$2"
  sc_x="$3"
  sc_y="$4"
  T="$5"
  p=$(expr $px \* $py)

  echo hybrid77
  for ((i=0; i<$T; i++)) do
    mpirun -np $p python main.py hybrid77 $px $py $sc_x $sc_y
  done
  echo
}


function test_hybrid90 () {
  px="$1"
  py="$2"
  sc_x="$3"
  sc_y="$4"
  T="$5"
  p=$(expr $px \* $py)

  echo hybrid90
  for ((i=0; i<$T; i++)) do
    mpirun -np $p python main.py hybrid90 $px $py $sc_x $sc_y
  done
  echo
}

function test_all () {
  px="$1"
  py="$2"
  sc_x="$3"
  sc_y="$4"
  T="$5"

  test_python $px $py $sc_x $sc_y $T
  test_f2py77 $px $py $sc_x $sc_y $T
  test_f2py90 $px $py $sc_x $sc_y $T
  test_hybrid77 $px $py $sc_x $sc_y $T
  test_hybrid90 $px $py $sc_x $sc_y $T
  echo
}


# number of trials; change as need be.
T=1

echo Compiling Fortran code with Ofast optimization...
f2py --opt=-Ofast -c -m flux_ener_f2py77 flux_ener_f2py.f 2>/dev/null 1>&2
f2py --opt=-Ofast --f90flags=-ffixed-line-length-0 -c -m  flux_ener_f2py90 flux_ener_f2py.f90 2>/dev/null 1>&2
echo Compiled.

# this is assuming that sc_x = sc_y = sc, because of laziness from me.
for sc in 1 2; do
  echo sc = $sc, $T trials
  echo

  test_all 1 1 $sc $sc $T   # serial
  test_python 2 1 $sc $sc $T   # px = 2, py = 1
  test_python 1 2 $sc $sc $T   # px = 1, py = 2
  test_python 2 2 $sc $sc $T   # px = 2, py = 2
  test_python 4 1 $sc $sc $T
  test_python 1 4 $sc $sc $T
done

ulimit -s $old_stack_lim
echo Changed stack limit back to $old_stack_lim.
