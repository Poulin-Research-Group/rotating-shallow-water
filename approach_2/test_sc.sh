sc="$1"
T="$2"
echo sc: $sc, $T trials

echo numpy
for ((i=0; i<$T; i++)) do
  python sadourny.py numpy $sc
done

echo f2py-f77
for ((i=0; i<$T; i++)) do
  python sadourny.py f77 $sc
done

echo f2py-f90
for ((i=0; i<$T; i++)) do
  python sadourny.py f90 $sc
done

echo DONE $sc
