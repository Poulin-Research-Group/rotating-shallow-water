if [ ! -d "tests" ]; then
  mkdir tests
  mkdir tests/numpy tests/f2py-f77 tests/f2py-f90
fi

for sc in 1 2 4; do
	bash test_sc.sh $sc 10
done

for sc in 8 16; do
	bash test_sc.sh $sc 2
done
