#!/usr/bin/env python3

import os, sys, time
import numpy as np
import pandas as pd
from importlib import reload
import matplotlib.pyplot as plt

import extended_catalog
reload(extended_catalog)
ecl = extended_catalog.ExtendedCatalog()

import robust_stats as rs

## -----------------------------------------------------------------------
## Quick region file maker:
def make_region_file(rpath, stars, r1pix=2, r2pix=5, color='green'):
    with open(rpath, 'w') as rf:
        for targ in stars:
            rf.write("image; annulus(%.3f, %.3f, %.3f, %.3f) #color=%s\n"
                    % (targ['x'], targ['y'], r1pix, r2pix, color))
        pass
    return

## -----------------------------------------------------------------------

fcat_file = {}
fcat_file['dirty'] = 'unpruned_wircam_H2_1644696p.fits.fz.fcat'
fcat_file['prune'] = 'wircam_H2_1644696p.fits.fz.fcat'
#fcat_file['clean'] = 'wircam_H2_1644696p.fits.fz.fcat'

data = {}
srcs = {}
hdrs = {}
for fftag,ffpath in fcat_file.items():
    ecl.load_from_fits(ffpath)
    srcs[fftag] = ecl.get_catalog()
    hdrs[fftag] = ecl.get_header()
hdrs['clean'] = hdrs['dirty']

# summary:
stars = srcs['dirty']
nstars = len(stars)
sys.stderr.write("We have %d detections.\n" % nstars)

# double-check meaning of 'ypeak' value:
yposition = np.int_(stars['y']+0.5)

totrows = 2050
row_num = np.arange(totrows)
row_pop = {}
rpk_pop = {}
for fftag,stars in srcs.items():
    row_pop[fftag] = np.bincount(np.int_(stars['y']+0.5), minlength=totrows)
    rpk_pop[fftag] = np.bincount(stars['ypeak'], minlength=totrows)

row_pop['clean'] = row_pop['prune']
rpk_pop['clean'] = rpk_pop['prune']

##sys.exit(0)
## look for 'popular' detection rows:
##row_pop = np.bincount(np.int_(stars['y']-0.5), minlength=2050)
#row_pop = np.bincount(np.int_(stars['y']+0.5), minlength=2050)
#rpk_pop = np.bincount(stars['ypeak'], minlength=2050)
#row_num = np.arange(row_pop.size)
#
## row population statistics:
#row_med, row_sig = rs.calc_ls_med_IQR(row_pop)
#rpk_med, rpk_sig = rs.calc_ls_med_IQR(rpk_pop)

#thresh = 10
#rpk_high = (rpk_pop > thresh)
#row_high = (row_pop > thresh)

## the bad feature is at Y = 235 ...
#bad_row = 235
#
#dodgy = (bad_row-1 <= stars['y']) & (stars['y'] <= bad_row+1)
#
#baddies = stars[dodgy]

## Perform a trial cleanup:
thresh = 10
#dirty_stars =     srcs['dirty']
is_row_junk = (row_pop['dirty'] > thresh)
bad_row_num = row_num[(row_pop['dirty'] > thresh)]
bad_rpk_num = row_num[(rpk_pop['dirty'] > thresh)]
#bad_sources = is_row_junk
#bad_sources = is_rpk_junk
#bad_sources = is_row_junk | is_rpk_junk
#srcs['clean'] = srcs['dirty'][~bad_sources]

## Clean up a dirty one:
clean_srcs = np.copy(srcs['dirty'])
for rr in bad_rpk_num:
    dropme = (clean_srcs['ypeak'] == rr)
    clean_srcs = clean_srcs[~dropme]
srcs['clean'] = clean_srcs

## Rerun this to populate the 'clean' bincounts:
for fftag,stars in srcs.items():
    row_pop[fftag] = np.bincount(np.int_(stars['y']+0.5), minlength=totrows)
    rpk_pop[fftag] = np.bincount(stars['ypeak'], minlength=totrows)

## quickly make region files:
for fftag,stars in srcs.items():
    make_region_file('example_%s.reg' % fftag, stars)


## -----------------------------------------------------------------------
## Make some plots ...
fig_dims = (11, 9)
fig = plt.figure(1, figsize=fig_dims)
fig, axs = plt.subplots(3, 2, sharex=True, figsize=fig_dims, num=1, clear=True)

dtypes = ['dirty', 'prune', 'clean']

for ii,tag in enumerate(dtypes):
    npts = len(srcs[tag])
    ax1, ax2 = axs[ii]
    ax1.set_title("%s -- y(row) -- %d" % (tag, npts))
    ax1.plot(row_num, row_pop[tag])

    ax2.set_title("%s -- ypeak(row) -- %d" % (tag, npts))
    ax2.plot(row_num, rpk_pop[tag])
    pass

#ax1 = fig.add_subplot(321)
#ax1.plot(row_num, row_pop)
#ax1.set_title("raw -- y(row)")
#        
#ax2 = fig.add_subplot(322)
#ax2.plot(row_num, rpk_pop)
#ax2.set_title("raw -- ypeak(row)")
#
#ax3 = fig.add_subplot(323)
#ax3.plot(row_num, rpk_pop)
#ax3.set_title("raw -- ypeak(row)")

for ax in axs.flatten():
    ax.set_ylim(-5, 110)
    ax.set_xlabel("Y pixel (row)")
    ax.set_ylabel("N detections")
    ax.axhline(thresh, ls='--', c='r')


plot_name = 'row_artifact_cleanup.png'
fig.tight_layout()
plt.draw()
fig.savefig(plot_name, bbox_inches='tight')


