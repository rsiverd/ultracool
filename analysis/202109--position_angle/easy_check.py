#!/usr/bin/env python

import os, sys, time
import numpy as np
import astropy.io.fits as pf


## Make a rotation matrix:
def rotation_matrix(theta):
    """Generate 2x2 rotation matrix for specified input angle (radians)."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

## Reflect across X-axis (reverses Y):
xref_mat = np.array([[1.0, 0.0], [0.0, -1.0]])
yref_mat = np.array([[-1.0, 0.0], [0.0, 1.0]])
xflip_mat = yref_mat
yflip_mat = xref_mat

def conversion_test(radec_resid, pos_ang_deg):
    rmat = rotation_matrix(np.radians(pos_ang_deg))
    spun = np.dot(rmat, radec_resid)
    return np.dot(xflip_mat, spun)

#def two_angle_test(radec_resid, pa_x, pa_y):
#    rmat_x = rotation_matrix(np.radians(pa_x))
#    spun_x = np.dot(rmat_x, radec_resid)
#    rmat_y = rotation_matrix(np.radians(pa_y))
#    spun_y = np.dot(rmat_y, radec_resid)
#    new_xy = np.array([spun_x[0], spun_y[1]])
#    return spun_x, spun_y, new_xy

## Comment from header:
## PA = [deg] Position angle of axis 2 (E of N)

sample_image = 'SPITZER_I2_61246720_0009_0000_1_nudge.fits'
idata, ihdrs = pf.getdata(sample_image, header=True)
pos_ang_deg = ihdrs['PA']
pos_ang_rad = np.radians(pos_ang_deg)
yax_pos_deg = pos_ang_deg
xax_pos_deg = yax_pos_deg - 90.0

yax_rpa = np.radians(yax_pos_deg)
xax_rpa = np.radians(xax_pos_deg)

rmat = rotation_matrix(pos_ang_rad)

## Snag pixel scale:
pscale_keys = ['PXSCAL1', 'PXSCAL2']
pscale_vals = np.array([ihdrs[k] for k in pscale_keys])
pixel_scale = np.average(np.abs(pscale_vals))
rebuilt_cdm = rmat * pscale_vals[:, None]

## Snag CD-matrix:
cd_keywords = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
cd_matrix = np.array([ihdrs[x] for x in cd_keywords]).reshape(2,2)

## Cardinal directions:
east_resid = np.array([1.0, 0.0])
west_resid = -1.0 * east_resid
north_resid = np.array([0.0, 1.0])
south_resid = -1.0 * north_resid

## Expectation:
## * east residual is mostly +Y, slightly +X

sys.stderr.write("Due East converts to:\n")
east_in_xy = np.dot(rmat, east_resid)
east_in_xy = conversion_test(east_resid, pos_ang_deg)
sys.stderr.write("--> %s\n" % str(east_in_xy))

sys.stderr.write("Due North converts to:\n")
north_in_xy = conversion_test(north_resid, pos_ang_deg)
sys.stderr.write("--> %s\n" % str(north_in_xy))
#sys.stderr.write("\n")

