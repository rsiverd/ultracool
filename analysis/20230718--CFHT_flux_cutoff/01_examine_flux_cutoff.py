#!/usr/bin/env python3
import os, sys, time
import matplotlib.pyplot as plt
import numpy as np
import robust_stats as rs
from numpy.lib.recfunctions import append_fields

import extended_catalog
ecl = extended_catalog.ExtendedCatalog()

## -----------------------------------------------------------------------
## instrumental magnitude calculator:
def calc_instmag(counts, zeropt=25.0):
    return (zeropt - 2.5 * np.log10(counts))

def calc_instadu(mag, zeropt=25.0):
    return (10.0**(0.4 * (zeropt - mag)))

## -----------------------------------------------------------------------
## Quick region file maker:
def make_region_file(rpath, stars, r1pix=2, r2pix=5, color='green'):
    with open(rpath, 'w') as rf:
        for targ in stars:
            labels = "color=%s text={%.2f}" % (color, targ['instmag'])
            rf.write("image; annulus(%.3f, %.3f, %.3f, %.3f) # %s\n"
                    % (targ['x'], targ['y'], r1pix, r2pix, labels))
        pass
    return

## -----------------------------------------------------------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------

#this_ipath = 'wircam_H2_1319416p.fits.fz'
this_ipath = 'wircam_J_1319394p.fits.fz'
this_fpath = this_ipath + '.fcat'
this_rpath = this_fpath + ".reg"

ecl.load_from_fits(this_fpath)

stars = ecl.get_catalog()
nsrcs = len(stars)

instmag = calc_instmag(stars['flux'])
stars = append_fields(stars, 'instmag', instmag, usemask=False)


mag_cutoff = 18.0
nkept = np.sum(instmag <= mag_cutoff)
sys.stderr.write("Mag cutoff of %.2f keeps %d of %d stars.\n"
        % (mag_cutoff, nkept, nsrcs))

adu_cutoff = 631.

make_region_file(this_rpath, stars)
sys.stderr.write("ztf -r %s %s\n" % (this_rpath, this_ipath))


