#!/usr/bin/env python

import astropy.io.fits as pf
import numpy as np
import os, sys, time

def imsum(filename):
    idata = pf.getdata(filename)
    return np.sum(idata[~np.isnan(idata)])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write("Syntax: %s image.fits\n" % sys.argv[0])
        sys.exit(1)
    impath = sys.argv[1]
    #print("impath: %s" % impath)
    result = imsum(impath)
    print(result)


