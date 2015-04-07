# make test directory if it doesn't exist
if [ ! -d "tests" ]; then
  echo Creating test directories...
  mkdir -p tests/numpy
  mkdir -p tests/f2py-f77/O0 tests/f2py-f77/O3 tests/f2py-f77/Ofast 
  mkdir -p tests/f2py-f90/O0 tests/f2py-f90/O3 tests/f2py-f90/Ofast
  mkdir -p tests/hybrid77/O0 tests/hybrid77/O3 tests/hybrid77/Ofast 
  mkdir -p tests/hybrid90/O0 tests/hybrid90/O3 tests/hybrid90/Ofast 
  mkdir -p tests/f77/O0 tests/f77/O3 tests/f77/Ofast
fi

function test_python () {
  sc="$1"
  T="$2"

  echo numpy
  for ((i=0; i<$T; i++)) do
    python sadourny.py numpy $sc
  done
  echo
}

function test_f2py () {
  # Tests f2py-f77, f2py-f90, hybrid77, hybrid90
  # with a specified sc and number of trials, T.
  sc="$1"
  T="$2"
  opt="$3"

  echo f2py-f77
  for ((i=0; i<$T; i++)) do
    python sadourny.py f2py-f77 $sc $opt
  done
  echo

  echo f2py-f90
  for ((i=0; i<$T; i++)) do
    python sadourny.py f2py-f90 $sc $opt
  done
  echo

  echo hybrid77
  for ((i=0; i<$T; i++)) do
    python sadourny.py hybrid77 $sc $opt
  done
  echo

  echo hybrid90
  for ((i=0; i<$T; i++)) do
    python sadourny.py hybrid90 $sc $opt
  done
  echo
}

function test_fortran () {
  # Tests F77 with a specified sc and a number of
  # trials, T.
  sc="$1"
  T="$2"
  opt="$3"

  file_f77=tests/f77/$opt/sc-$sc.txt
  file_f90=tests/f90/$opt/sc-$sc.txt
  echo $file_f77

  echo f77
  for ((i=0; i<$T; i++)) do
    /usr/bin/time -f "%U" -a -o $file_f77 ./flux_ener_F    # should it be "%e" instead??
  done
  echo
}


function opt_tests () {
  sc="$1"
  T="$2"
  opt="$3"

  echo "$opt optimization ------"

  f2py --opt=-$opt -c -m flux_sw_ener77 flux_sw_ener.f 2>/dev/null 1>&2
  f2py --opt=-$opt --f90flags=-ffixed-line-length-0 -c -m  flux_sw_ener90 flux_sw_ener.f90 2>/dev/null 1>&2
  gfortran -$opt flux_sw_ener.f -o flux_ener_F

  test_f2py $sc $T $opt
  test_fortran $sc $T $opt
}


# THIS SHOULDN'T BE CHANGED
old_sc=1

# number of trials; change as need be.
T=10

for sc in 1 2 4; do
  echo sc = $sc, $T trials
  echo

  python modify_fortran_code.py $old_sc $sc    # modify the fortran script's sc value.

  test_python $sc $T
  opt_tests $sc $T O0
  opt_tests $sc $T O3
  opt_tests $sc $T Ofast
  old_sc=$sc
done

T=2
for sc in 8 16; do
  echo sc = $sc, $T trials
  echo

  python modify_fortran_code.py $old_sc $sc    # modify the fortran script's sc value.

  test_python $sc $T
  opt_tests $sc $T O0
  opt_tests $sc $T O3
  opt_tests $sc $T Ofast
  old_sc=$sc
done

# change value of sc back to 1 in Fortran script
python modify_fortran_code.py $sc 1
