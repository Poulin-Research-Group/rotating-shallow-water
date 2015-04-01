# make test directory if it doesn't exist
if [ ! -d "tests" ]; then
  mkdir tests
  mkdir tests/numpy tests/f2py-f77 tests/f2py-f90 tests/hybrid77 tests/hybrid90
fi

function test_sc () {
    sc="$1"
    T="$2"

    echo numpy
    for ((i=0; i<$T; i++)) do
      python sadourny.py numpy $sc
    done

    echo f2py-f77
    for ((i=0; i<$T; i++)) do
      python sadourny.py f2py-f77 $sc
    done

    echo f2py-f90
    for ((i=0; i<$T; i++)) do
      python sadourny.py f2py-f90 $sc
    done

    echo hybrid77
    for ((i=0; i<$T; i++)) do
      python sadourny.py hybrid77 $sc
    done

    echo hybrid90
    for ((i=0; i<$T; i++)) do
      python sadourny.py hybrid90 $sc
    done
}


for sc in 1 2 4; do
  test_sc $sc 10
done

for sc in 8; do
  test_sc $sc 1
done