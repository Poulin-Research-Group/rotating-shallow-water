sc="$1"
echo sc: $sc
echo numpy
for ((i=0; i<10; i++)) do
  python sadourny.py numpy $sc
done

echo f77
for ((i=0; i<10; i++)) do
  python sadourny.py f77 $sc
done

echo f90
for ((i=0; i<10; i++)) do
  python sadourny.py f90 $sc
done

echo DONE $sc
