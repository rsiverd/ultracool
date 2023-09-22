#!/usr/bin/env python

import os, sys, time
import numpy as np
import pandas as pd
import fov_rotation
rfov = fov_rotation.RotateFOV()


## Some data files:
gm_file = 'wircam_J_1838732p_fixed.fits.fz.fcat.gmatch'
wp_file = 'wircam_J_1838732p_fixed.fits.fz.fcat.wcspar'
gm_data = pd.read_csv(gm_file)
wp_data = pd.read_csv(wp_file)

## WCS defaults:
crpix1 = 2122.690779
crpix2 =  -81.678888
pscale =   0.30601957084155673

## A CD matrix for the image above:
cdmat = np.array([[-0.000085  , -0.00000042],
                   [-0.00000042,  0.000085  ]])

## Fix the xrel/yrel:
gm_data['xrel'] = gm_data['x'] - crpix1
gm_data['yrel'] = gm_data['y'] - crpix2

## Note the CRVALs:
crval1 = wp_data['fit3_crval1'][0]
crval2 = wp_data['fit3_crval2'][0]


## ----------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- ##

## Tangent projection:
def _tanproj(prj_xx, prj_yy):
    prj_rr = np.hypot(prj_xx, prj_yy)
    #sys.stderr.write("%.3f < prj_rr < %.3f\n" % (prj_rr.min(), prj_rr.max()))
    #prj_rr = np.sqrt(prj_xx**2 + prj_yy**2)
    #sys.stderr.write("%.3f < prj_rr < %.3f\n" % (prj_rr.min(), prj_rr.max()))
    useful = (prj_rr > 0.0)
    prj_theta = np.ones_like(prj_xx) * np.pi * 0.5
    prj_theta[useful] = np.arctan(np.degrees(1.0 / prj_rr[useful]))
    #prj_theta[useful] = np.arctan(_radeg / prj_rr[useful])
    #prj_phi = np.arctan2(prj_xx, prj_yy)
    prj_phi = np.arctan2(prj_xx, -prj_yy)
    #return prj_phi, prj_theta
    return np.degrees(prj_phi), np.degrees(prj_theta)

def _inv_tanproj(prj_phi_deg, prj_theta_deg):
    prj_phi_rad   = np.radians(prj_phi)
    prj_theta_rad = np.radians(prj_theta_deg)
    sin_phi       = np.sin(prj_phi_rad)
    cos_phi       = np.cos(prj_phi_rad)
    sin_theta     = np.sin(prj_theta_rad)
    cos_theta     = np.cos(prj_theta_rad)
    prj_xx        = np.zeros_like(prj_phi_rad) #+ sin_phi
    prj_yy        = np.zeros_like(prj_phi_rad) #+ cos_phi
    prj_rr        = np.zeros_like(prj_phi_rad)
    which         = (sin_theta != 0.0)      # calculate for these
    prj_rr[which] = np.degrees(cos_theta / sin_theta)
    #prj_xx_over_neg_prj_yy = np.tan(prj_phi_rad)
    #prj_xy_ratio = np.tan(prj_phi_rad)
    #prj_rr = np.radians(1.0 / np.tan(prj_theta_rad))
    #prj_yy2 = prj_rr**2 / (1.0 + prj_xy_ratio**2)
    prj_xx[which] =        prj_rr * sin_phi
    prj_yy[which] = -1.0 * prj_rr * cos_phi
    return prj_xx, prj_yy
    #return prj_rr, prj_yy2

## Low-level WCS tangent processor:
def _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=False):
    prj_xx, prj_yy = np.matmul(thisCD, relpix)
    if debug:
        sys.stderr.write("%.3f < prj_xx < %.3f\n" % (prj_xx.min(), prj_xx.max()))
        sys.stderr.write("%.3f < prj_yy < %.3f\n" % (prj_yy.min(), prj_yy.max()))

    # Perform tangent projection:
    prj_phi, prj_theta = _tanproj(prj_xx, prj_yy)
    if debug:
        sys.stderr.write("%.3f < prj_theta < %.3f\n"
                % (prj_theta.min(), prj_theta.max()))
        sys.stderr.write("%.3f < prj_phi   < %.3f\n"
                % (prj_phi.min(), prj_phi.max()))

    # Change variable names to avoid confusion:
    rel_ra, rel_de = prj_phi, prj_theta
    if debug:
        phi_range = prj_phi.max() - prj_phi.min()
        sys.stderr.write("phi range: %.4f < phi < %.4f\n"
                % (prj_phi.min(), prj_phi.max()))

    # Shift to 
    old_fov = (0.0, 90.0, 0.0)
    new_fov = (crval1, crval2, 0.0)
    stuff = rfov.migrate_fov_deg(old_fov, new_fov, (rel_ra, rel_de))
    return stuff

## Convert X,Y to RA, Dec using CD matrix and CRVAL pair:
#def xycd2radec(cdmat, xpix, ypix, crval1, crval2, debug=False):
def xycd2radec(cdmat, rel_xx, rel_yy, crval1, crval2, debug=False):
    thisCD = np.array(cdmat).reshape(2, 2)
    relpix = np.array((rel_xx, rel_yy))     # rel time = 1.05 us
    #prj_xx, prj_yy = np.matmul(thisCD, relpix)
    return _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=debug)

#def duh_xycd2radec(cdmat, rel_xx, rel_yy, crval1, crval2, debug=False):
#    thisCD = np.array(cdmat).reshape(2, 2)
#    relpix = np.array((rel_xx, rel_yy))     # rel time = 1.05 us
#    #prj_xx, prj_yy = np.matmul(thisCD, relpix)
#    return _wcs_tan_compute(thisCD, relpix, crval1, crval2, debug=debug)


def sky2xy_cd(cdmat, ra_deg, de_deg, crval1, crval2):
    inv_CD  = np.linalg.inv(cdmat)
    new_fov = (0.0, 90.0, 0.0)
    old_fov = (crval1, crval2, 0.0)
    prj_phi, prj_theta = rfov.migrate_fov_deg(old_fov, new_fov, (ra_deg, de_deg))
    prj_xx, prj_yy = _inv_tanproj(prj_phi, prj_theta)
    rel_foc = np.array((inv_xx, inv_yy))
    ccd_xrel, ccd_yrel = np.matmul(inv_CD, rel_foc)
    return ccd_xrel, ccd_yrel

## ----------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- ##

## Step through a calculation:
thisCD = cdmat
relpix = np.array((gm_data['xrel'].values, gm_data['yrel'].values))
# In [20]: %timeit relpix = np.array((gm_data['xrel'].values, gm_data['yrel'].values))
# 7.19 µs ± 75.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

#dfrpix = gm_data[['xrel', 'yrel']].values.T
# In [19]: %timeit dfrpix = gm_data[['xrel', 'yrel']].values.T
# 328 µs ± 4.64 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

prj_xx, prj_yy = np.matmul(thisCD, relpix)
prj_rr = np.hypot(prj_xx, prj_yy)


prj_phi, prj_theta = _tanproj(prj_xx, prj_yy)


inv_xx, inv_yy = _inv_tanproj(prj_phi, prj_theta)
inv_CD = np.linalg.inv(thisCD)
invfoc = np.array((inv_xx, inv_yy))

#inv_relpix = np.matmul(inv_CD, invfoc)
inv_xrel, inv_yrel = np.matmul(inv_CD, invfoc)

inv_xpix = inv_xrel + crpix1
inv_ypix = inv_yrel + crpix2

deg_ra, deg_de = xycd2radec(thisCD, gm_data['xrel'].values, gm_data['yrel'].values, crval1, crval2)
# In [38]: %timeit deg_ra, deg_de = xycd2radec(thisCD, gm_data['xrel'].values, gm_data['yrel'].values, crval1, crval2)
# 740 µs ± 669 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)

#cmp_ra, cmp_de = duh_xycd2radec(thisCD, gm_data['xrel'].values, gm_data['yrel'].values, crval1, crval2)
# In [37]: %timeit cmp_ra, cmp_de = duh_xycd2radec(thisCD, gm_data['xrel'].values, gm_data['yrel'].values, crval1, crval2)
# 734 µs ± 1.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

#new_fov = (0.0, 90.0, 0.0)
#old_fov = (crval1, crval2, 0.0)
##stuff   = rfov.migrate_fov_deg(old_fov, new_fov, (rel_ra, rel_de))
#stuff   = rfov.migrate_fov_deg(old_fov, new_fov, (deg_ra, deg_de))
#itp_phi, itp_theta = stuff

yay_xrel, yay_yrel = sky2xy_cd(thisCD, deg_ra, deg_de, crval1, crval2)

