#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Help determine actual units of downloaded Spitzer images.
#
# Rob Siverd
# Created:       2019-09-11
# Last modified: 2019-09-11
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

import os, sys, time

star1 = {
    'ra'    :  63.8901815,
    'dec'   :  -9.4344435,
    'w2flx' :   775532.496,
    'expt'  :       26.8,
    'ipeak' :      170.0,
    'itotal':      393.0671,
    }

star2 = {
    'ra'    :  63.8090503,
    'dec'   :  -9.4921137,
    'w2flx' : 94188959.652,
    'expt'  :       10.4,
    'ipeak' :     6700.0,
    'itotal':    40509.34,  # SPITZER_I2_49755136_0004_0000_3_cbcd.fits.cat
    }

def rcalc(datum):
    return star2[datum] / star1[datum]


## Expected flux ratio for matching exposures:
#w2ratio = star2['w2flx'] / star1['w2flx']
w2ratio = rcalc('w2flx')
sys.stderr.write("W2 flux ratio: %10.3f\n" % w2ratio)

#pkratio = star2['ipeak'] / star1['ipeak']
pkratio = rcalc('ipeak')
imratio = rcalc('itotal')
sys.stderr.write("Star peak counts ratio: %10.3f\n" % pkratio)
sys.stderr.write("Star total count ratio: %10.3f\n" % imratio)

