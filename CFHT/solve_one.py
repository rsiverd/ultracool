#!/usr/bin/env python3

import os, sys, time
import math
import numpy as np
import astropy.time as astt
import astropy.io.fits as pf
import astropy.wcs as awcs
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

import fluxmag
reload(fluxmag)

##--------------------------------------------------------------------------##

## Region-file creation tools:
_have_region_utils = False
try:
    import region_utils
    reload(region_utils)
    rfy = region_utils
    _have_region_utils = True
except ImportError:
    sys.stderr.write(
            "\nWARNING: region_utils not found, DS9 regions disabled!\n")

##--------------------------------------------------------------------------##
## Simplified module loading:
mod_spec = [('affine', 'afx', 'AffineFit2D'),
            ('fov_rotation', 'rfov', 'RotateFOV'),
            ('segmatch1', 'smv1', 'SegMatch'),
            ('segmodel', 'smf', 'SegModelFit'),
            ]

## Load, reload, and instantiate:
for mname,oname,thing in mod_spec:
    try:
        globals()[mname] = __import__(mname)
        reload(globals()[mname])
    except ImportError:
        sys.stderr.write("\nError: module '%s' not found!\n" % mname)
        sys.exit(1)
    entity = getattr(globals()[mname], thing, False)
    if (entity == False):
        sys.stderr.write("\n"
            + "\nError: module '%s' has no attribute '%s'\n" % (mname, thing)
            + "Please update import specs and retry ...\n\n")
        sys.exit(1)
    globals()[oname] = entity()
    pass

#import segmodel
#reload(segmodel)
#smf = segmodel.SegModelFit()

#import segmatch
#reload(segmatch)
#psm = segmatch.SegMatch()
#import segmatch1
#reload(segmatch1)
#smv1 = segmatch1.SegMatch()

#import fov_rotation
#reload(fov_rotation)
#rfov = fov_rotation.RotateFOV()

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
ephcat3 = '%s/calib1_p_NE/by_runid/12BQ08/wircam_H2_1592961p_eph.fits.fz.fcat' % cfh_abby_root
ephcat4 = '%s/calib1_p_NE/by_runid/11AQ15/wircam_H2_1319399p_eph.fits.fz.fcat' % cfh_abby_root
ephcat5 = '%s/calib1_p_NE/by_runid/15BQ09/wircam_H2_1838749p_eph.fits.fz.fcat' % cfh_abby_root
ephcat6 = '%s/calib1_p_NE/by_runid/17AQ07/wircam_J_2094830p_eph.fits.fz.fcat' % cfh_abby_root
use_fcat = ephcat1
#use_fcat = ephcat2
use_fcat = ephcat3
use_fcat = ephcat5
#use_fcat = ephcat4
#use_fcat = ephcat6

view_img = use_fcat.replace('p_eph', 'p').replace('fz.fcat', 'fz')

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

## Miscellany:
def calc_MAR(residuals):
    #_MAD_scalefactor = 1.482602218
    return 1.482602218 * np.median(residuals)

## Sensor size:
#sensor_xpix = 2048
#sensor_ypix = 2048

#corner_size = 256
#corner_xmin = sensor_xpix - corner_size
#corner_ymax = corner_size


## -----------------------------------------------------------------------
## Region files of sources in corner region:
sky_rfile = 'lookie_radec.reg'
pix_rfile = 'lookie_pixel.reg'
cal_rfile = 'lookie_rdcal.reg'

rsave_dir = '.'
save_regions = True

## Region file settings and options:
#colorset = ['red', 'magenta', 'green', 'blue', 'cyan', 'orange', 'yellow']
inspection = False
inspection = True
#save_regions = False
vlevel = 1


## -----------------------------------------------------------------------
## Load Gaia:
gm.load_sources_csv(gaia_csv_path)
#gm2.load_sources_csv(gaia_csv_path)

_xdw_col = 'xdw_cs23'
_ydw_col = 'ydw_cs23'
#_xdw_col = 'xdw_dl12'
#_ydw_col = 'ydw_dl12'

## Load ExtCat content:
ecl.load_from_fits(use_fcat)
stars = ecl.get_catalog()
header = ecl.get_header()
obs_time = hs.wircam_timestamp_from_header(header)
xrel = stars[_xdw_col] - crpix1
yrel = stars[_ydw_col] - crpix2
twcs = awcs.WCS(header)
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
#gm.set_Gmag_limit(15.0)
#gm.set_Gmag_limit(16.0)

## Select from corner:
#lr_stars = hs.get_corner_subset_rect(stars)
lr_stars = hs.get_corner_subset_dist(stars)

#lr_gaia_matches = hs.find_gaia_matches(lr_stars, 2.0)
#lr_gaia_matches = find_gaia_matches_idx(lr_stars, 2.0, ra_col='dra', de_col='dde')
use_cols = {'ra_col':'dra', 'de_col':'dde'}
lr_gaia_matches = find_gaia_matches_idx(lr_stars, 2.0, **use_cols)
idx, gra, gde   = lr_gaia_matches
match_subset    = lr_stars[idx]
m_xrel, m_yrel  = match_subset['xrel'], match_subset['yrel']

## -----------------------------------------------------------------------
## -----------------------------------------------------------------------

segmatch_t1 = time.time()

## View config:
pscale = 0.304  # arcsec / pixel
pscale = 0.305  # arcsec / pixel
#parity = 'positive'
parity = 'negative'     # East-left (+RA parallel to +X)
#ctr_ra, ctr_de = hs.get_center_radec(twcs)
#ctr_radec, llc_radec = hs.get_fov_params(twcs)
ctr_radec, halfdiag, quad_radec = hs.get_fov_params(twcs)
nsrc_max = 50
ndets_max = 50
ngaia_max = 60
#nsrc_max = 100
#nsrc_max = 150

#_LOWER_RIGHT = False
#_fit_region = 'whole_frame'
#_fit_region = 'lower_right'
_LRQ_ONLY = True
#_LRQ_ONLY = False

## Segment-based matches:
_mag_col = 'phot_g_mean_mag'
#mlim_lo  = 13.0
#mlim_lo  = 10.0
mlim_lo  =  9.0
mlim_lo  =  7.0
mlim_lo  =  5.0
#mlim_hi  = 15.0
#mlim_hi  = 14.0
#mlim_hi  = 16.0
g_tmp = gm._srcdata.copy().sort_values(by=_mag_col, ascending=True)
#g_mag = gm._srcdata[_mag_col].values
#m_ord = np.argsort(g_mag)
#m_ord = np.argsort(gm._srcdata[_mag_col].values)
#g_tmp = gm._srcdata.copy().iloc[m_ord]
g_mag = g_tmp[_mag_col].values
if _LRQ_ONLY:
    g_sep = angle.dAngSep(*quad_radec['lr'], g_tmp['ra'].values, g_tmp['dec'].values)
    sep_okay = (g_sep <= 0.5 * halfdiag)
    #g_use = np.where(magok & (g_sep <= halfdiag))[0][:nsrc_max]
    mlim_hi = 17.0
else:
    g_sep = angle.dAngSep(*ctr_radec, g_tmp['ra'].values, g_tmp['dec'].values)
    sep_okay = (g_sep <= 1.0 * halfdiag)
    mlim_hi = 15.0
    #g_use = np.where(magok & (g_sep <= halfdiag))[0][:nsrc_max]
mag_okay = (mlim_lo <= g_mag) & (g_mag <= mlim_hi)
#g_sep = angle.dAngSep(*ctr_radec, gm._srcdata['ra'].values, gm._srcdata['dec'].values)
#g_use = np.where(magok & (g_sep <= halfdiag))[0][:nsrc_max]
g_use = np.where(mag_okay & sep_okay)[0][:ngaia_max]

#g_use = np.where((mlim_lo <= g_mag) & (g_mag <= mlim_hi) \
#                    & (g_sep <= halfdiag))[0][:nsrc_max]
sys.stderr.write("Segment match FOV shifting ... ")
#cat_RA, cat_DE = gm._srcdata['ra'].values[g_use], gm._srcdata['dec'].values[g_use]
cat_RA, cat_DE = g_tmp['ra'].values[g_use], g_tmp['dec'].values[g_use]
cat_mag = g_mag[g_use]
#mid_RA, mid_DE = np.average(cat_RA), np.average(cat_DE)
#old_fov = (cat_RA.mean(), cat_DE.mean(), 0.0)
old_fov = (ctr_radec[0], ctr_radec[1], 0.0)
fov_rel_RA, fov_rel_DE = rfov.roll_to_origin_deg(old_fov, (cat_RA, cat_DE))
fov_rel_RA *= np.cos(np.radians(fov_rel_DE))
fov_rel_RA *= (3600.0 / pscale)
fov_rel_DE *= (3600.0 / pscale)
if (parity == 'negative'):
    fov_rel_RA *= -1.0
sys.stderr.write("done.\n")
kgaia_reg = 'kept_gaia.reg'
hs.make_radec_region(kgaia_reg, cat_RA, cat_DE)

_USING_REL_RADEC = True

## ----------------------------------------------------------------------- ##
sys.stderr.write("Convert Gaia objects to relative X,Y (inverse tan proj) ... ")
pa_deg_guess = 0.0
guess_cdmat = tp.make_cdmat(pa_deg_guess, pscale)
inv_xrel, inv_yrel = tp.sky2xy_cd(guess_cdmat, cat_RA, cat_DE, 
        ctr_radec[0], ctr_radec[1])

fov_rel_xx = inv_xrel + crpix1
fov_rel_yy = inv_yrel + crpix2
sys.stderr.write("done.\n")

## ----------------------------------------------------------------------- ##
sys.stderr.write("Choose detections for matching ... ")
dmlimit = (0.1, 2)
dmlimit = (0.1, 7)
dmlimit = (0.5, 7)
dmlimit = (0.2, 7)
gaia_depth = 10
fcat_depth = 10
#fcat_depth = 5
#gaia_depth = 5
#gaia_depth = 9
#fcat_depth = 15
#gaia_depth = 15
fcat_depth = 30
gaia_depth = 30
#fcat_depth = 0
#gaia_depth = 0

xcenter, ycenter = 1024.5, 1024.5
min_flux =  600.0
min_flux = 2000.0
#ccd_keep = (stars['flux'] > min_flux)
if _LRQ_ONLY:
    min_flux =  600.0
    sep_okay = (xcenter <= stars['x']) & (stars['y'] <= ycenter)
else:
    sep_okay = (stars['x'] > -999)  # keep everything
    min_flux = 2000.0
flx_okay = (stars['flux'] >= min_flux)
ccd_keep = np.where(flx_okay & sep_okay)[0][:ndets_max]
#ccd_keep = np.where(stars['flux'] > min_flux)[0][:nsrc_max]
#ccd_xrel = stars['xrel'][ccd_keep]
#ccd_yrel = stars['yrel'][ccd_keep]
ccd_xrel = stars[_xdw_col][ccd_keep] - xcenter
ccd_yrel = stars[_ydw_col][ccd_keep] - ycenter
ccd_mags = fluxmag.kmag(stars['flux'][ccd_keep])
#ccd_xpos = stars[_xdw_col][ccd_keep]
#ccd_ypos = stars[_ydw_col][ccd_keep]
ccd_xpos = stars[     'x'][ccd_keep]
ccd_ypos = stars[     'y'][ccd_keep]
sys.stderr.write("region file ... ")
kstar_reg = 'kept_stars.reg'
hs.make_pixel_region(kstar_reg, ccd_xpos, ccd_ypos)
sys.stderr.write("done.\n")
ccd_xx, ccd_yy = ccd_xpos, ccd_ypos

#psm.set_catalog1(fov_rel_RA, fov_rel_DE, cat_mag, dmlimit=dmlimit, depth=gaia_depth)
##psm.set_catalog2(stars['xrel'], stars['yrel'], ccd_mag, depth=fcat_depth)
#psm.set_catalog2(ccd_xrel, ccd_yrel, ccd_mags, depth=fcat_depth)

sys.stderr.write("Setting catalogs and segments ... ")
segmatch_t2 = time.time()

_USE_MAGS = True
#_USE_MAGS = False

if _USING_REL_RADEC:
    fov_rel_xpos = fov_rel_RA
    fov_rel_ypos = fov_rel_DE
else:
    fov_rel_xpos = fov_rel_xx
    fov_rel_ypos = fov_rel_yy

smv1._vlevel = 3
if _USE_MAGS:
    #smv1.set_catalog1(x=fov_rel_RA, y=fov_rel_DE, mag=cat_mag,
    #        dmlimit=dmlimit, depth=gaia_depth)
    smv1.set_catalog1(x=fov_rel_xpos, y=fov_rel_ypos, mag=cat_mag, 
                            dmlimit=dmlimit, depth=gaia_depth)
    #smv1.set_catalog2(x=ccd_xpos, y=ccd_ypos, mag=ccd_mags, 
    smv1.set_catalog2(x=ccd_xrel, y=ccd_yrel, mag=ccd_mags, 
                        dmlimit=dmlimit, depth=fcat_depth)
    #smv1.set_catalog2(x=ccd_xrel, y=ccd_yrel, mag=ccd_mags, depth=fcat_depth)
    #smv1.set_catalog1(x=fov_rel_RA, y=fov_rel_DE, depth=gaia_depth)
    #smv1.set_catalog2(x=ccd_xrel, y=ccd_yrel, depth=fcat_depth)
else:
    smv1.set_catalog1(x=fov_rel_xpos, y=fov_rel_ypos, dmlimit=dmlimit, depth=gaia_depth)
    smv1.set_catalog2(x=ccd_xrel, y=ccd_yrel, dmlimit=dmlimit, depth=fcat_depth)
segmatch_t3 = time.time()
sys.stderr.write("done.\n")

nsegs = smv1._nsegment
sys.stderr.write("Created %d segments from Gaia sources.\n" % nsegs[0])
sys.stderr.write("Created %d segments from CFHT sources.\n" % nsegs[1])

## -----------------------------------------------------------------------
## -----------------------------------------------------------------------

sys.stderr.write("Setting ranges/tols ... ")
lentol = np.log10(1.2)
lentol = np.log10(1.1)
#lentol = np.log10(1.5)
#lentol = np.log10(2.0)
lenbins = 9
#lenbins = 5
#lenbins = 3
lenrange = smv1.bintol_range(lenbins, lentol)

angtol   = 2.0
#angbins  = 100
#angbins  = 181
angbins  =   5
#angtol   = 1.0
angbins  =  11
angrange = smv1.bintol_range(angbins, angtol)

magtol  = 8.0
magbins = 1
magrange = smv1.bintol_range(magbins, magtol)

if _USE_MAGS:
    use_ranges = (lenrange, angrange, magrange)
    use_nbins  = ( lenbins,  angbins,  magbins)
else:
    use_ranges = (lenrange, angrange)
    use_nbins  = ( lenbins,  angbins)

sys.stderr.write("done.\n")

sys.stderr.write("Matching segments ... ")
segmatch_t4 = time.time()
tdivs = (2, 2, 2)
#tdivs = (3, 3, 3)
#best_pars = psm.dither_hist_best_fit(use_ranges, use_nbins,
#        tdivs, mode='weighted') #, mode='weighted')


#tdivs = (3, 3, 3)
best_pars = smv1.dither_hist_best_fit(use_ranges, use_nbins,
        #tdivs, mode='best') #, mode='weighted')
        tdivs, mode='weighted') #, mode='weighted')
        ##tdivs, mode='best') #, mode='weighted')
#best_pars = psm.hist_best_fit(use_ranges, use_nbins)
segmatch_t4 = time.time()
t_fitting = segmatch_t4 - segmatch_t3
sys.stderr.write("done.  Matching took %.3f seconds.\n" % t_fitting)

## Illustrate which stars were matched:
sys.stderr.write("Select matched sources ... ")
matched_idx = smv1.matched_source_indexes()
gidx, sidx = zip(*matched_idx)
matched_gra = cat_RA[gidx,]
matched_gde = cat_DE[gidx,]
matched_mag = cat_mag[gidx,]
matched_sxx = ccd_xpos[sidx,]
matched_syy = ccd_ypos[sidx,]
sys.stderr.write("region files ... ")
mstar_reg = 'matched_stars.reg'
mgaia_reg = 'matched_gaia.reg'
hs.make_pixel_region(mstar_reg, matched_sxx, matched_syy)
hs.make_radec_region(mgaia_reg, matched_gra, matched_gde)
segmatch_t5 = time.time()
sys.stderr.write("done.\n")

total_smv1 = segmatch_t5 - segmatch_t1
#sys.stderr.write("Total segmatch run-time: %.3f seconds\n" % total_smv1)

n_matches = len(matched_gra)
#sys.stderr.write("Cross-matched sources: %d\n" % n_matches)
sys.stderr.write("Cross-matched %d sources in %.2f seconds.\n\n"
        % (n_matches, total_smv1))

sys.stderr.write("fztf %s\n\n" % view_img)

sys.stderr.write("ztf -r %s -r %s %s\n\n" % (kstar_reg, kgaia_reg, view_img))
sys.stderr.write("ztf -r %s -r %s %s\n\n" % (mstar_reg, mgaia_reg, view_img))

## Note stuff about the solution:
obj_pairs = smv1.matched_source_indexes()
n_raw_pairs = len(obj_pairs)
#astrometry_result['n_raw_matches'] = n_raw_pairs
sys.stderr.write("Found %d object pairing(s).\n" % n_raw_pairs)
if (n_raw_pairs == 0):
    red_warning("NO MATCHED SOURCES!")
    sys.exit(1)
elif (n_raw_pairs < 2):
    red_warning("Too few matches for astrometric fit!")
    sys.exit(1)
vmatch, smatch = zip(*obj_pairs)

## Inspect results (SMR = seg match raw):
smr_cat_match_ids, smr_ccd_match_ids = zip(*obj_pairs)
smr_matched_ra = cat_RA[smr_cat_match_ids,]
smr_matched_de = cat_DE[smr_cat_match_ids,]
smr_matched_xx = ccd_xx[smr_ccd_match_ids,]
smr_matched_yy = ccd_yy[smr_ccd_match_ids,]

#match_subset = stars['

## Parameters for minimization:
match_vecs = {'true_ra' : smr_matched_ra,
              'true_de' : smr_matched_de,
                 'xrel' : stars[ccd_keep][smr_ccd_match_ids,]['xrel'],
                 'yrel' : stars[ccd_keep][smr_ccd_match_ids,]['yrel']}
#gra, gde = smr_matched_ra, smr_matched_de

## Simple RA/Dec offsets:
ra_offset = np.median(stars[ccd_keep][smr_ccd_match_ids,]['dra'] - smr_matched_ra)
de_offset = np.median(stars[ccd_keep][smr_ccd_match_ids,]['dde'] - smr_matched_de)

## -----------------------------------------------------------------------
## -----------------------------------------------------------------------
## -----------------------------------------------------------------------

## -----------------------------------------------------------------------
## Match finder for affine transform coordinates:
def recheck_for_matches(coords1, coords2, match_tol):
    matches_1 = []
    for i,star in enumerate(coords1):
        rdist = np.sqrt(np.sum((coords2 - star)**2, axis=1))
        min_dex, min_sep = rdist.argmin(), rdist.min()
        if (min_sep <= match_tol):
            matches_1.append((i, min_dex))
        pass
    matches_2 = []
    for j,star in enumerate(coords2):
        rdist = np.sqrt(np.sum((star - coords1)**2, axis=1))
        min_dex, min_sep = rdist.argmin(), rdist.min()
        if (min_sep <= match_tol):
            matches_2.append((min_dex, j))
        pass
    #return match_list
    #sys.stderr.write("matches_1:\n%s\n" % str(matches_1))
    #sys.stderr.write("matches_2:\n%s\n" % str(matches_2))
    return list(set(matches_1) & set(matches_2))

## Refinement!

coo1, coo2 = smv1.get_matched_coords()
smf.setup(coo1, coo2, best_pars)
good_matches, tuned_pars = smf.clean()

## Look for additional matches:
#cat_rel_radec = np.column_stack((fov_rel_RA, fov_rel_DE))
cat_rel_radec = np.column_stack((fov_rel_xpos, fov_rel_ypos))
ccd_rel_xypix = np.column_stack((  ccd_xrel,   ccd_yrel))
ccd_rel_radec = smf.xform(ccd_rel_xypix, tuned_pars)

## Check match tolerance of known-good matches:
safe_ccd_indxs = np.array(smr_ccd_match_ids,)[good_matches,]
safe_cat_indxs = np.array(smr_cat_match_ids,)[good_matches,]
good_ccd_radec = ccd_rel_radec[safe_ccd_indxs]
good_cat_radec = cat_rel_radec[safe_cat_indxs]
good_resid = np.sqrt(np.sum((good_ccd_radec - good_cat_radec)**2.0, axis=1))
aug_tol = 1.02 * good_resid.max()
sys.stderr.write("Augmentation match tolerance: %10.5f\n" % aug_tol)


xtra_matches = recheck_for_matches(cat_rel_radec, ccd_rel_radec, aug_tol)
n_aug_pairs = len(xtra_matches)
sys.stderr.write("Additional pass identified %d matching sources.\n"
        % n_aug_pairs)

## Inspect results:
if (n_aug_pairs > 0):
    sma_cat_match_ids, sma_ccd_match_ids = zip(*xtra_matches)
    sma_matched_ra = cat_RA[sma_cat_match_ids,]
    sma_matched_de = cat_DE[sma_cat_match_ids,]
    sma_matched_xx = ccd_xx[sma_ccd_match_ids,]
    sma_matched_yy = ccd_yy[sma_ccd_match_ids,]

    ## Color-matched comparison regions:
    cmv_region = '%s/cmatched_cat_raw.reg' % rsave_dir
    cms_region = '%s/cmatched_ccd_raw.reg' % rsave_dir
    if save_regions:
        rfy.regify_sky(cmv_region, sma_matched_ra, sma_matched_de,
                colors=rfy.colorset, vlevel=vlevel, rdeg=0.003)
        rfy.regify_ccd(cms_region, sma_matched_xx, sma_matched_yy,
                colors=rfy.colorset, vlevel=vlevel, rpix=13)
        if inspection:
            sys.stderr.write("\n")
            rfy.reg_announce("Inspect AUGMENTED matched sources (color-coded)",
                    view_img, [cmv_region, cms_region])
            pass
        pass

    match_vecs = {'true_ra' : sma_matched_ra,
                  'true_de' : sma_matched_de,
                     'xrel' : stars[ccd_keep][sma_ccd_match_ids,]['xrel'],
                     'yrel' : stars[ccd_keep][sma_ccd_match_ids,]['yrel']}


## -----------------------------------------------------------------------
## Fitting exponent:
fitting_exponent = 2.0
fitting_exponent = 1.0

## Function to minimize:
sys.stderr.write("Starting minimization ... \n")
tik = time.time()
#init_guess = [pscale, 0.0, header['CRVAL1'], header['CRVAL2']]
#init_params = np.array([0.0, header['CRVAL1'], header['CRVAL2']])
init_params = np.array([cdm_pa, header['CRVAL1'], header['CRVAL2']])
#init_params = np.array([cdm_pa, header['CRVAL1'] - ra_offset, header['CRVAL2'] - de_offset])
#minimize_this = partial(hs.evaluator, pscale=pscale, imdata=match_subset, gra=gra, gde=gde)
minimize_this = partial(hs.evaluator_pacrv, pscale=pscale, 
        expo=fitting_exponent, **match_vecs)
        #xrel=match_subset['xrel'], yrel=match_subset['yrel'],
        #true_ra=gra, true_de=gde, expo=fitting_exponent)
answer = opti.fmin(minimize_this, init_params)
tok = time.time()
sys.stderr.write("Minimum found in %.3f seconds.\n" % (tok-tik))

## WCS parameters:
best_pa, best_cv1, best_cv2 = answer
best_cdmat = tp.make_cdmat(best_pa, pscale).flatten()
best_ra, best_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, 
        match_vecs['xrel'], match_vecs['yrel'])
#best_ra, best_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, m_xrel, m_yrel)

## Angular offsets:
#match_sep_asec = 3600.0 * angle.dAngSep(best_ra, best_de, gra, gde)
match_sep_asec = 3600.0 * angle.dAngSep(best_ra, best_de, match_vecs['true_ra'], match_vecs['true_de'])
med_resid_asec = np.median(match_sep_asec)
resid_MAR = calc_MAR(match_sep_asec)
sys.stderr.write("After 1st round, median residual: %.2f arcsec\n" % med_resid_asec)

#sys.exit(0)
## Re-calculate RA/DE for everything:
calc_ra, calc_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, xrel, yrel)
stars = append_fields(stars, ('calc_ra', 'calc_de'), (calc_ra, calc_de), usemask=False)
lr_stars = hs.get_corner_subset_dist(stars, 2000.)
use_cols = {'ra_col':'calc_ra', 'de_col':'calc_de'}
#lr_gaia_matches = find_gaia_matches_idx(lr_stars, 2.0, **use_cols)
lr_gaia_matches = find_gaia_matches_idx(lr_stars, 3.0, **use_cols)
idx, gra, gde   = lr_gaia_matches
match_subset    = lr_stars[idx]
m_xrel, m_yrel  = match_subset['xrel'], match_subset['yrel']

new_init_params = np.array([best_pa, best_cv1, best_cv2])
minimize_this   = partial(hs.evaluator_pacrv, pscale=pscale, 
                        xrel=match_subset['xrel'], yrel=match_subset['yrel'],
                        true_ra=gra, true_de=gde, expo=fitting_exponent)
new_answer      = opti.fmin(minimize_this, init_params)
best_pa, best_cv1, best_cv2 = new_answer
best_cdmat = tp.make_cdmat(best_pa, pscale).flatten()
best_ra, best_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, m_xrel, m_yrel)

## Angular offsets:
match_sep_asec = 3600.0 * angle.dAngSep(best_ra, best_de, gra, gde)
med_resid_asec = np.median(match_sep_asec)
resid_MAR = calc_MAR(match_sep_asec)
sys.stderr.write("After 2nd round, median residual: %.2f arcsec\n" % med_resid_asec)

## Run full-frame matching:
calc_ra, calc_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, xrel, yrel)
stars['calc_ra'] = calc_ra
stars['calc_de'] = calc_de
use_cols = {'ra_col':'calc_ra', 'de_col':'calc_de'}
ff_stars = stars #hs.get_corner_subset_dist(stars, 2000.)
#ff_gaia_matches = find_gaia_matches_idx(ff_stars, 2.0, **use_cols)
ff_gaia_matches = find_gaia_matches_idx(ff_stars, 3.0, **use_cols)
idx, gra, gde   = ff_gaia_matches
match_subset    = ff_stars[idx]
m_xrel, m_yrel  = match_subset['xrel'], match_subset['yrel']

new_init_params = np.array([best_pa, best_cv1, best_cv2])
minimize_this   = partial(hs.evaluator_pacrv, pscale=pscale, 
                        xrel=match_subset['xrel'], yrel=match_subset['yrel'],
                        true_ra=gra, true_de=gde, expo=fitting_exponent)
new_answer      = opti.fmin(minimize_this, init_params)
best_pa, best_cv1, best_cv2 = new_answer
best_cdmat = tp.make_cdmat(best_pa, pscale).flatten()
best_ra, best_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, m_xrel, m_yrel)

## Angular offsets:
match_sep_asec = 3600.0 * angle.dAngSep(best_ra, best_de, gra, gde)
med_resid_asec = np.median(match_sep_asec)
resid_MAR = calc_MAR(match_sep_asec)
sys.stderr.write("After 3rd round, median residual: %.2f arcsec\n" % med_resid_asec)

### One last full-frame matching, with tighter tolerance:
#calc_ra, calc_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, xrel, yrel)
#stars['calc_ra'] = calc_ra
#stars['calc_de'] = calc_de
#use_cols = {'ra_col':'calc_ra', 'de_col':'calc_de'}
#ff_stars = stars
#ff_gaia_matches = find_gaia_matches_idx(ff_stars, 1.0, **use_cols)
#idx, gra, gde   = ff_gaia_matches
#match_subset    = ff_stars[idx]
#m_xrel, m_yrel  = match_subset['xrel'], match_subset['yrel']
#
#new_init_params = np.array([best_pa, best_cv1, best_cv2])
#minimize_this   = partial(hs.evaluator_pacrv, pscale=pscale, 
#                        xrel=match_subset['xrel'], yrel=match_subset['yrel'],
#                        true_ra=gra, true_de=gde, expo=fitting_exponent)
#new_answer      = opti.fmin(minimize_this, init_params)
#best_pa, best_cv1, best_cv2 = new_answer
#best_cdmat = tp.make_cdmat(best_pa, pscale).flatten()
#best_ra, best_de = hs.calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, m_xrel, m_yrel)
#
### Angular offsets:
#match_sep_asec = 3600.0 * angle.dAngSep(best_ra, best_de, gra, gde)
#resid_MAR = calc_MAR(match_sep_asec)
#sys.stderr.write("After 4th round, median residual: %.2f arcsec\n" % med_resid_asec)

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

