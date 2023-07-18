#!/usr/bin/env python3

import os, sys, time
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from importlib import reload

#import wircam_poly
#reload(wircam_poly)
#wcp  = wircam_poly.WIRCamPoly()

##--------------------------------------------------------------------------##
## Disable buffering on stdout/stderr:
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

##--------------------------------------------------------------------------##

sys.stderr.write("Loading big CSV file ... ")
tik = time.time()
csv_file = '20230629--final_matches.csv'
data = pd.read_csv(csv_file)
tok = time.time()
sys.stderr.write("done. Took %.3f seconds.\n" % (tok-tik))

# Per-cell data point picker:
def get_cell_subset(x_cell, y_cell):
    keep = (data['x_cell'] == x_cell) & (data['y_cell'] == y_cell)
    return data[keep]

# Pre-calculate relative X,Y positions:
_wir_crpix1 = 2122.69077900
_wir_crpix2 =  -81.6788876100
data['xrel'] = data['X Pixel'] - _wir_crpix1
data['yrel'] = data['Y Pixel'] - _wir_crpix2

# Pre-calculate approximate X,Y deltas:
cos_dec = np.cos(np.radians(data['Gaia Dec']))
de_diff_deg = (data['Calc Decs'] - data['Gaia Dec'])
ra_diff_deg = (data['Calc RAs' ] - data['Gaia RA' ]) * cos_dec
wircam_scale = 0.3042820    # arcsec/pixel
de_diff_pix =  1.0 * 3600.0 * de_diff_deg / wircam_scale
ra_diff_pix = -1.0 * 3600.0 * ra_diff_deg / wircam_scale
data['dx'] = ra_diff_pix
data['dy'] = de_diff_pix
data['dr'] = np.hypot(ra_diff_pix, de_diff_pix)

# How many grid cells are we using:
n_x_cells = data.x_cell.max() + 1
n_y_cells = data.y_cell.max() + 1
sys.stderr.write("Data file has %dx%d grid configuration.\n"
        % (n_x_cells, n_y_cells))

# Median dx,dy for each cell:
med_dx_save = np.zeros((n_y_cells, n_x_cells))
med_dy_save = np.zeros((n_y_cells, n_x_cells))
med_xx_save = np.zeros_like(med_dx_save)
med_yy_save = np.zeros_like(med_dx_save)

# Calculate median shifts in each grid cell:
sys.stderr.write("Starting per-cell analysis ...\n")
tik = time.time()
for yc in range(n_y_cells):
    sys.stderr.write("Y-cell %d ...\n" % yc)
    ywhich = (data['y_cell'] == yc)
    ychunk =  data[ywhich]
    for xc in range(n_x_cells):
        #sys.stderr.write("this xc,yc = %2d,%2d\n" % (xc,yc))
        xwhich = (ychunk['x_cell'] == xc)
        this_cell = ychunk[xwhich]
        median_dx = np.median(this_cell['dx'])
        median_dy = np.median(this_cell['dy'])
        med_dx_save[yc, xc] = median_dx
        med_dy_save[yc, xc] = median_dy
        med_xx_save[yc, xc] = float(xc)
        med_yy_save[yc, xc] = float(yc)
        #this_cell = get_cell_subset(xc, yc)
tok = time.time()
sys.stderr.write("Analysis completed in %.3f seconds.\n" % (tok-tik))

med_dr_save = np.hypot(med_dx_save, med_dy_save)

near_top_left = get_cell_subset(1, 14)
full_top_left = get_cell_subset(0, 15)

