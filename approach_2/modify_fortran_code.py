"""
The hackiest solution known to mankind. (Not actually, though.)
This script reads in 'flux_sw_ener.f', finds the line
  parameter (sc=1)
(Or whatever value sc is) and replaces it with a given value of
sc. Awful, I know.
"""
import sys
try:
    sc_old, sc_new = sys.argv[1:]
except ValueError:
    sc_old, sc_new = 1, 2

# read in the script
FILE = open('./flux_sw_ener.f')
script = FILE.read()
FILE.close()

# replace sc with given value
script = script.replace('parameter (sc=%s)' % sc_old, 'parameter (sc=%s)' % sc_new)

# write out the script
FILE = open('./flux_sw_ener.f', 'w')
FILE.write(script)
FILE.close()
