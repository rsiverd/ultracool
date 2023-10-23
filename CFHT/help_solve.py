#!/usr/bin/env python3
#
# Various helpers for the skeletal WIRCam data processing.

import os, sys, time
import math
import astropy.time as astt
import astropy.io.fits as pf
import tangent_proj as tp
import numpy as np
import angle

## -----------------------------------------------------------------------
## Configuration items:
crpix1 = 2122.690779
crpix2 =  -81.678888
pscale =   0.30601957084155673

## Sensor size:
sensor_xpix = 2048
sensor_ypix = 2048

sensor_xmid = (sensor_xpix + 1.0) / 2.0
sensor_ymid = (sensor_xpix + 1.0) / 2.0

corner_size = 256
corner_size = 512
corner_xmin = sensor_xpix - corner_size
corner_ymax = corner_size

radial_size = 1.4142 * corner_size * 1.5

## -----------------------------------------------------------------------
## Extract field-of-view parameters from WCS:
def get_fov_params(imwcs):
    wcskw = {'ra_dec_order':True}
    #llc_ra, llc_de = np.array(imwcs.all_pix2world(1, 1, 1, **wcskw))
    llc_radec = np.array(imwcs.all_pix2world(1, 1, 1, **wcskw))
    ctr_radec = np.array(imwcs.all_pix2world(sensor_xmid, sensor_ymid, 1, **wcskw))
    # approximate quadrant centers:
    quad_radec = {}
    lrq_pixel = (0.75 * sensor_xpix, 0.25 * sensor_ypix)    # lower-right
    llq_pixel = (0.25 * sensor_xpix, 0.25 * sensor_ypix)    # lower-left
    ulq_pixel = (0.25 * sensor_xpix, 0.75 * sensor_ypix)    # upper-left
    urq_pixel = (0.75 * sensor_xpix, 0.75 * sensor_ypix)    # upper-right
    quad_radec['ll'] = np.array(imwcs.all_pix2world(*llq_pixel, 1, **wcskw))
    quad_radec['lr'] = np.array(imwcs.all_pix2world(*lrq_pixel, 1, **wcskw))
    quad_radec['ul'] = np.array(imwcs.all_pix2world(*ulq_pixel, 1, **wcskw))
    quad_radec['ur'] = np.array(imwcs.all_pix2world(*urq_pixel, 1, **wcskw))
    #coo = imwcs.all_pix2world([sensor_xmid, 1], [sensor_ymid, 1], 1, **wcskw)
    #return (coo[0][0], coo[1][0])
    #ctr_ra, ctr_de = np.array(coo)
    #diag_deg = angle.dAngSep
    halfdiag_deg = angle.dAngSep(*ctr_radec, *llc_radec)
    return ctr_radec, halfdiag_deg, quad_radec

## -----------------------------------------------------------------------
## Catalog "flavor" lurks at the end of the basename, before the "."
#def change_catalog_flavor(filename, flav1, flav2):
#    prev_dir = os.path.dirname(filename)
#    old_base = os.path.basename(filename)
#    old_parts = old_base.split('.')
#    old_first = old_parts[0]
#    if not old_first.endswith(flav1):
#        sys.stderr.write("File %s does not have flavor '%s'\n"
#                % (filename, flav1))
#        return None
#    new_first = old_first.replace(flav1, flav2)
#    new_parts = [new_first] + old_parts[1:]
#    new_base  = '.'.join(new_parts)
#    return os.path.join(prev_dir, new_base)

##--------------------------------------------------------------------------##
## Save FITS image with clobber (astropy / pyfits):
def qsave(iname, idata, header=None, padkeys=1000, **kwargs):
    this_func = sys._getframe().f_code.co_name
    parent_func = sys._getframe(1).f_code.co_name
    sys.stderr.write("Writing to '%s' ... " % iname)
    if header:
        while (len(header) < padkeys):
            header.append() # pad header
    if os.path.isfile(iname):
        os.remove(iname)
    pf.writeto(iname, idata, header=header, **kwargs)
    sys.stderr.write("done.\n")

## -----------------------------------------------------------------------
## Make region file with RA/Dec positions:
def make_radec_region(rpath, ra_deg, de_deg, 
        r1=0.0003, r2=0.001, color='green'):
    options = 'color=%s' % color
    with open(rpath, 'w') as rf:
        for rr,dd in zip(ra_deg, de_deg):
            rf.write("fk5; annulus(%.6fd, %.6fd, %.6fd, %.6fd) # %s\n"
                    % (rr, dd, r1, r2, options))
            pass
        pass
    return

## -----------------------------------------------------------------------
## Make region file with RA/Dec positions:
def make_pixel_region(rpath, xpix, ypix,
        r1=4.0, r2=8.0, color='red'):
    options = 'color=%s' % color
    with open(rpath, 'w') as rf:
        for xx,yy in zip(xpix, ypix):
            rf.write("image; annulus(%.6f, %.6f, %.2f, %.2f) # %s\n"
                    % (xx, yy, r1, r2, options))
            pass
        pass
    return

## -----------------------------------------------------------------------
## Make astropy Time object from image header:
def wircam_timestamp_from_header(header):
    obs_time = astt.Time(header['MJD-OBS'], scale='utc', format='mjd') \
            + 0.5 * astt.TimeDelta(header['EXPTIME'], format='sec')
    return obs_time

## Select lower-right corner from WIRCam images:
def get_corner_subset_rect(data):
    lower_right = (corner_xmin <= data['x']) & (data['y'] <= corner_ymax)
    return data[lower_right]

## Select lower-right sources within fixed distance from focal plane center:
def get_corner_subset_dist(data, cutoff=radial_size):
    rdist = np.hypot(data['x'] - crpix1, data['y'] - crpix2)
    return data[rdist <= cutoff]

## Select objects near the focal plane center:
def get_central_sources(data, max_rsep_pix):
    rsep = np.hypot(data['x'] - crpix1, data['y'] - crpix2)
    return data[rsep <= max_rsep_pix]

## -----------------------------------------------------------------------
## Gaia matching routine:
def find_gaia_matches(stars, tol_arcsec, ra_col='dra', de_col='dde',
        xx_col='x', yy_col='y'):
    tol_deg = tol_arcsec / 3600.0
    matches = []
    for target in stars:
        sra, sde = target[ra_col], target[de_col]
        sxx, syy = target[xx_col], target[yy_col]
        result = gm.nearest_star(sra, sde, tol_deg)
        if result['match']:
            gcoords = [result['record'][x].values[0] for x in ('ra', 'dec')]
            matches.append((sxx, syy, sra, sde, *gcoords))
            pass
        pass
    return matches

## -----------------------------------------------------------------------
## Fitting procedure:
def calc_tan_radec(pscale, pa_deg, cv1, cv2, xrel, yrel):
    this_cdmat = tp.make_cdmat(pa_deg, pscale)
    return tp.xycd2radec(this_cdmat, xrel, yrel, cv1, cv2)

def eval_tan_params(pscale, pa_deg, cv1, cv2, xrel, yrel, true_ra, true_de, expo=1):
    calc_ra, calc_de = calc_tan_radec(pscale, pa_deg, cv1, cv2, xrel, yrel)
    deltas = angle.dAngSep(calc_ra, calc_de, true_ra, true_de)
    return np.sum(deltas**expo)

def evaluator_pacrv(pacrv, pscale, xrel, yrel, true_ra, true_de, expo=1):
    pa_deg, cv1, cv2 = pacrv
    return eval_tan_params(pscale, pa_deg, cv1, cv2, 
            xrel, yrel, true_ra, true_de, expo=expo)

# Convert the answer into CD matrix values:
def cdmat_from_answer(answer):
    pa_deg, cv1, cv2 = answer

## -----------------------------------------------------------------------

# Analyze CD matrix from header:
_cd_keys = ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')
def get_cdmatrix_pa_scale(header):
    orig_cdm = np.array([header[x] for x in _cd_keys]).reshape(2, 2)
    cd_xyscl = np.sqrt(np.sum(orig_cdm**2, axis=1))
    norm_cdm = orig_cdm / cd_xyscl
    norm_rot = np.dot(tp.xflip_mat, norm_cdm)
    flat_rot = norm_rot.flatten()
    pa_guess = [math.acos(flat_rot[0]), -math.asin(flat_rot[1]),
                        math.asin(flat_rot[2]), math.acos(flat_rot[3])]
    pos_ang  = np.degrees(np.average(pa_guess))
    pixscale = np.average(cd_xyscl)
    return pos_ang, pixscale

