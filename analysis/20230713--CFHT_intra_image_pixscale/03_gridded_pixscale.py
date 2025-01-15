#!/usr/bin/env python3

import os, sys, time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##--------------------------------------------------------------------------##
## Load/store ability (slow calculation):
pkl_save = 'pixel_scales.pickle'

if os.path.isfile(pkl_save):
    sys.stderr.write("Loading data from %s ... " % pkl_save)
    with open(pkl_save, 'rb') as ppp:
        rscale_save = pickle.load(ppp)
    sys.stderr.write("done.\n")
else:
    sys.stderr.write("File not found: %s\n" % pkl_save)
    sys.exit(1)

## Make a large numpy array:
rsdata = np.concatenate(list(x for x in rscale_save.values()))
rx, ry, rpix, rpscl = rsdata.T


##--------------------------------------------------------------------------##
## Break down image into grid cells:
ny, nx = 2048, 2048
gridsize = 32
n_xcells = gridsize
n_ycells = gridsize

cellpix_y = float(ny) / float(n_ycells)
cellpix_x = float(nx) / float(n_xcells)

icx = np.int_((rx - 0.5) / cellpix_x)
icy = np.int_((ry - 0.5) / cellpix_y)

#rsdata = np.column_stack((rsdata, icx, icy))

## Storage for pixscale:
pscales = np.zeros((n_ycells, n_xcells))

## Iterate over cells:
for yc in range(n_ycells):
    #sys.stderr.write("yc: %d\n" % yc)
    which_ycell = (icy == yc)
    y_subset = rsdata[which_ycell]
    y_icx    = icx[which_ycell]
    for xc in range(n_xcells):
        #sys.stderr.write("yc=%2d, xc=%2d\n" % (yc, xc))
        which_xcell = (y_icx == xc)
        cell_subset = y_subset[which_xcell]
        cell_rx, cell_ry, cell_rpix, cell_pscl = cell_subset.T
        pscales[yc, xc] = np.median(cell_pscl)

##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (12, 9)
fig = plt.figure(1, figsize=fig_dims)
fig.clf()

ax1 = fig.add_subplot(111, aspect='equal')

spts = ax1.imshow(pscales)
#ax1.invert_y
ax1.invert_yaxis()
ax1.set_xlabel('X cell')
ax1.set_ylabel('Y cell')
cbar = fig.colorbar(spts, orientation='vertical')
cbar.set_label('Scale (arcsec / pixel)')


plot_name = 'pixel_scale_%dx%d.png' % (n_xcells, n_ycells)
fig.tight_layout()
plt.draw()
fig.savefig(plot_name, bbox_inches='tight')

