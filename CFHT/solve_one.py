#!/usr/bin/env python3

import os, sys, time
import math
import numpy as np
import astropy.time as astt
import astropy.io.fits as pf
import scipy.optimize as opti
from functools import partial
from numpy.lib.recfunctions import append_fields
from importlib import reload
from functools import partial

import tangent_proj as tp
import angle

import gaia_match
reload(gaia_match)
gm  = gaia_match.GaiaMatch()
#gm2 = gaia_match.GaiaMatch()

import extended_catalog
ecl = extended_catalog.ExtendedCatalog()

import help_solve as hs
reload(hs)

import wircam_fs_helpers as wfh
reload(wfh)

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

def find_gaia_matches_idx(stars, tol_arcsec, ra_col='dra', de_col='dde'):
    tol_deg = tol_arcsec / 3600.0
    matches = []
    for idx,target in enumerate(stars):
        sra, sde = target[ra_col], target[de_col]
        #sxx, syy = target[xx_col], target[yy_col]
        result = gm.nearest_star(sra, sde, tol_deg)
        if result['match']:
            gcoords = [result['record'][x].values[0] for x in ('ra', 'dec')]
            matches.append((idx, *gcoords))
            pass
        pass
    idx, gra, gde = zip(*matches)
    return np.array(idx), np.array(gra), np.array(gde)

## -----------------------------------------------------------------------
## File I/O:
cfh_abby_root = '/home/rsiverd/ucd_project/ucd_cfh_data/for_abby'
gaia_csv_path = '%s/gaia_calib1_NE.csv' % cfh_abby_root
ephcat1 = '%s/calib1_p_NE/by_runid/21AQ18/wircam_J_2626491p_eph.fits.fz.fcat' % cfh_abby_root
ephcat2 = '%s/calib1_p_NE/by_runid/21AQ13/wircam_H2_2614008p_eph.fits.fz.fcat' % cfh_abby_root
use_fcat = ephcat1

## Corresponding FITS image:
#prev_cat = hs.change_catalog_flavor(use_fcat, 'p_eph', 'p')
prev_cat = wfh.change_catalog_flavor(use_fcat, 'p_eph', 'p')
image_path = prev_cat.rstrip('.fcat')
if not os.path.isfile(image_path):
    sys.stderr.write("Original FITS image not found:\n")
    sys.stderr.write("--> '%s'\n" % image_path)
    sys.exit(1)

## WCS defaults:
crpix1 = 2122.690779
crpix2 =  -81.678888
pscale =   0.30601957084155673

## Sensor size:
#sensor_xpix = 2048
#sensor_ypix = 2048

#corner_size = 256
#corner_xmin = sensor_xpix - corner_size
#corner_ymax = corner_size

## -----------------------------------------------------------------------
## Load Gaia:
gm.load_sources_csv(gaia_csv_path)
#gm2.load_sources_csv(gaia_csv_path)

## Load ExtCat content:
ecl.load_from_fits(use_fcat)
stars = ecl.get_catalog()
header = ecl.get_header()
obs_time = hs.wircam_timestamp_from_header(header)
xrel = stars['xdw_cs23'] - crpix1
yrel = stars['ydw_cs23'] - crpix2
#xrel = stars['xdw_dl12'] - crpix1
#yrel = stars['ydw_dl12'] - crpix2
stars = append_fields(stars, ('xrel', 'yrel'), (xrel, yrel), usemask=False)

cdm_pa, cdm_pscale = hs.get_cdmatrix_pa_scale(header)
orig_cv1 = header['CRVAL1']
orig_cv2 = header['CRVAL2']
calc_ra, calc_de = hs.calc_tan_radec(pscale, cdm_pa, orig_cv1, orig_cv2, xrel, yrel)
#stars = append_fields(stars, ('calc_ra', 'calc_de'), (calc_ra, calc_de), usemask=False)

## Configure Gaia match:
gm.set_epoch(obs_time)
#gm.set_Gmag_limit(99.0)
#gm.set_Gmag_limit(20.0)
gm.set_Gmag_limit(19.0)
#gm2.set_epoch(obs_time)
#gm2.set_Gmag_limit(20.0)

## Select from corner:
#lr_stars = hs.get_corner_subset_rect(stars)
lr_stars = hs.get_corner_subset_dist(stars)

#lr_gaia_matches = hs.find_gaia_matches(lr_stars, 2.0)
#lr_gaia_matches = find_gaia_matches_idx(lr_stars, 2.0, ra_col='dra', de_col='dde')
use_cols = {'ra_col':'dra', 'de_col':'dde'}
lr_gaia_matches = find_gaia_matches_idx(lr_stars, 2.0, **use_cols)
idx, gra, gde = lr_gaia_matches
match_subset = lr_stars[idx]
use_x, use_y = match_subset['xrel'], match_subset['yrel']


## Function to minimize:
sys.stderr.write("Starting minimization ... \n")
tik = time.time()
#init_guess = [pscale, 0.0, header['CRVAL1'], header['CRVAL2']]
#init_params = np.array([0.0, header['CRVAL1'], header['CRVAL2']])
init_params = np.array([cdm_pa, header['CRVAL1'], header['CRVAL2']])
#minimize_this = partial(hs.evaluator, pscale=pscale, imdata=match_subset, gra=gra, gde=gde)
minimize_this = partial(hs.evaluator_pacrv, pscale=pscale, 
        xrel=match_subset['xrel'], yrel=match_subset['yrel'], true_ra=gra, true_de=gde)
answer = opti.fmin(minimize_this, init_params)
tok = time.time()
sys.stderr.write("Minimum found in %.3f seconds.\n" % (tok-tik))

## WCS parameters:
best_pa, best_cv1, best_cv2 = answer
best_cdmat = tp.make_cdmat(best_pa, pscale).flatten()
best_ra, best_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, use_x, use_y)


## Re-calculate RA/DE for everything:
calc_ra, calc_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, xrel, yrel)
stars = append_fields(stars, ('calc_ra', 'calc_de'), (calc_ra, calc_de), usemask=False)
lr_stars = hs.get_corner_subset_dist(stars, 2000.)
use_cols = {'ra_col':'calc_ra', 'de_col':'calc_de'}
lr_gaia_matches = find_gaia_matches_idx(lr_stars, 2.0, **use_cols)
idx, gra, gde = lr_gaia_matches
match_subset = lr_stars[idx]
use_x, use_y = match_subset['xrel'], match_subset['yrel']

new_init_params = np.array([best_pa, best_cv1, best_cv2])
minimize_this = partial(hs.evaluator_pacrv, pscale=pscale, 
        xrel=match_subset['xrel'], yrel=match_subset['yrel'], true_ra=gra, true_de=gde)
new_answer = opti.fmin(minimize_this, init_params)
best_pa, best_cv1, best_cv2 = new_answer
best_cdmat = tp.make_cdmat(best_pa, pscale).flatten()
best_ra, best_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, use_x, use_y)

## Final, full-frame matching:
calc_ra, calc_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, xrel, yrel)
stars['calc_ra'] = calc_ra
stars['calc_de'] = calc_de
use_cols = {'ra_col':'calc_ra', 'de_col':'calc_de'}
ff_stars = stars #hs.get_corner_subset_dist(stars, 2000.)
ff_gaia_matches = find_gaia_matches_idx(ff_stars, 2.0, **use_cols)
idx, gra, gde = ff_gaia_matches
match_subset = ff_stars[idx]
use_x, use_y = match_subset['xrel'], match_subset['yrel']

new_init_params = np.array([best_pa, best_cv1, best_cv2])
minimize_this = partial(hs.evaluator_pacrv, pscale=pscale, 
        xrel=match_subset['xrel'], yrel=match_subset['yrel'], true_ra=gra, true_de=gde)
new_answer = opti.fmin(minimize_this, init_params)
best_pa, best_cv1, best_cv2 = new_answer
best_cdmat = tp.make_cdmat(best_pa, pscale).flatten()
best_ra, best_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, use_x, use_y)

## -----------------------------------------------------------------------
## Region files of sources in corner region:
sky_rfile = 'lookie_radec.reg'
pix_rfile = 'lookie_pixel.reg'
cal_rfile = 'lookie_rdcal.reg'
hs.make_radec_region(sky_rfile, gra, gde)
hs.make_pixel_region(pix_rfile, match_subset['x'], match_subset['y'])
hs.make_radec_region(cal_rfile, best_ra, best_de, color='blue')

#reg_list = [sky_rfile, pix_rfile, cal_rfile]

## -----------------------------------------------------------------------
## Load original FITS image:
fixed_image = 'good_wcs.fits'
idata, raw_hdr = pf.getdata(image_path, header=True)
new_hdr = raw_hdr.copy(strip=True)
new_hdr['CRVAL1'] = best_cv1
new_hdr['CRVAL2'] = best_cv2
new_hdr['CD1_1']  = best_cdmat[0]
new_hdr['CD1_2']  = best_cdmat[1]
new_hdr['CD2_1']  = best_cdmat[2]
new_hdr['CD2_2']  = best_cdmat[3]

#_cd_keys = ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')
#orig_cdm = np.array([raw_hdr[x] for x in _cd_keys]).reshape(2, 2)
#cd_pscales = np.sqrt(np.sum(orig_cdm**2, axis=1))
#norm_cdm = orig_cdm / cd_pscales
#norm_rot = np.dot(tp.xflip_mat, norm_cdm)
#flat_rot = norm_rot.flatten()
#pa_guess = [math.acos(flat_rot[0]), -math.asin(flat_rot[1]),
#            math.asin(flat_rot[2]), math.acos(flat_rot[3])]

raw_pa, raw_pscale = hs.get_cdmatrix_pa_scale(raw_hdr)

hs.qsave(fixed_image, idata, header=new_hdr)

sys.stderr.write("\nInspect adjusted WCS with:\n")
sys.stderr.write("ztf -r %s -r %s -r %s %s\n" % (cal_rfile, sky_rfile, pix_rfile, fixed_image))
sys.stderr.write("\nCompare to original WCS with:\n")
sys.stderr.write("ztf -r %s -r %s -r %s %s\n" % (cal_rfile, sky_rfile, pix_rfile, image_path))

