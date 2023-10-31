#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Test and benchmark different methods for two-way matching.
#
# Rob Siverd
# Created:       2023-10-24
# Last modified: 2023-10-30
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.2.0"

## Python version-agnostic module reloading:
try:
    reload                              # Python 2.7
except NameError:
    try:
        from importlib import reload    # Python 3.4+
    except ImportError:
        from imp import reload          # Python 3.0 - 3.3

## Modules:
#import argparse
#import shutil
#import resource
#import signal
#import glob
import gc
import os
import sys
import time
#import vaex
#import calendar
#import ephem
import numpy as np
from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#import scipy.linalg as sla
#import scipy.signal as ssig
#import scipy.ndimage as ndi
#import scipy.optimize as opti
#import scipy.interpolate as stp
#import scipy.spatial.distance as ssd
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import matplotlib.cm as cm
#import matplotlib.ticker as mt
#import matplotlib._pylab_helpers as hlp
#from matplotlib.colors import LogNorm
#import matplotlib.colors as mplcolors
#import matplotlib.collections as mcoll
#import matplotlib.gridspec as gridspec
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import theil_sen as ts
import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

import tangent_proj as tp
import angle

import gaia_match
reload(gaia_match)
gm  = gaia_match.GaiaMatch()
#gm2 = gaia_match.GaiaMatch()

import extended_catalog
ecl = extended_catalog.ExtendedCatalog()

## WCS tune-up helpers (beta):
try:
    import wircam_wcs_tuneup
    reload(wircam_wcs_tuneup)
    #ecl = extended_catalog.ExtendedCatalog()
    wwt = wircam_wcs_tuneup
except ImportError:
    logger.error("failed to import wircam_wcs_tuneup module!")
    sys.exit(1)

## DS9 region utilities:
import region_utils
reload(region_utils)
rfy = region_utils

## Cutesy divider strings:
fulldiv = 80 * '-'
halfdiv = 40 * '-'

##--------------------------------------------------------------------------##
## Projections with cartopy:
#try:
#    import cartopy.crs as ccrs
#    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#    from cartopy.feature.nightshade import Nightshade
#    #from cartopy import config as cartoconfig
#except ImportError:
#    sys.stderr.write("Error: cartopy module not found!\n")
#    sys.exit(1)

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

## Home-brew robust statistics:
#try:
#    import robust_stats
#    reload(robust_stats)
#    rs = robust_stats
#except ImportError:
#    logger.error("module robust_stats not found!  Install and retry.")
#    sys.stderr.write("\nError!  robust_stats module not found!\n"
#           "Please install and try again ...\n\n")
#    sys.exit(1)

## Various from astropy:
try:
#    import astropy.io.ascii as aia
    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
#    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
#    logger.error("astropy module not found!  Install and retry.")
    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

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

##--------------------------------------------------------------------------##
## Save FITS image with clobber (fitsio):
#def qsave(iname, idata, header=None, **kwargs):
#    this_func = sys._getframe().f_code.co_name
#    parent_func = sys._getframe(1).f_code.co_name
#    sys.stderr.write("Writing to '%s' ... " % iname)
#    #if os.path.isfile(iname):
#    #    os.remove(iname)
#    fitsio.write(iname, idata, clobber=True, header=header, **kwargs)
#    sys.stderr.write("done.\n")

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

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

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#def twoway_gaia_matches_1(stars, tol_arcsec, ra_col='dra', de_col='dde'):
def twoway_gaia_matches_1(ccd_ra, ccd_de, tol_arcsec):
    tol_deg = tol_arcsec / 3600.0
    #matches = []
    star_ids = []
    gaia_ids = []
    #for idx,target in enumerate(stars):
    for idx,(sra, sde) in enumerate(zip(ccd_ra, ccd_de)):
        #sra, sde = target[ra_col], target[de_col]
        #sxx, syy = target[xx_col], target[yy_col]
        result = gm.nearest_star(sra, sde, tol_deg)
        if result['match']:
            gaia_ids.append(int(result['record']['source_id']))
            #star_ids.append(idx)
            #import pdb; pdb.set_trace()
            #gcoords = [result['record'][x].values[0] for x in ('ra', 'dec')]
            #matches.append((idx, *gcoords))
            pass
        pass


    ## we COULD also grab a subset of 'stars' that matched ...
    #sys.stderr.write("len(stars): %d\n" % len(stars))
    #sys.stderr.write("Of those, %d might match.\n" % len(star_ids))

    # The subset of Gaia stars that we want to use:
    use_gaia = gm._srcdata[gm._srcdata.source_id.isin(gaia_ids)]
    #use_ccd_ra = ccd_ra[star_ids]
    #use_ccd_de = ccd_de[star_ids]

    # Match the other way:
    #sidx = []
    #hits = []
    matches = []
    for gi,(gix, gsrc) in enumerate(use_gaia.iterrows(), 1):
        sep_deg = angle.dAngSep(gsrc.ra, gsrc.dec,
                ccd_ra, ccd_de)
                #stars[ra_col], stars[de_col])
        #sys.stderr.write("sep_deg: %s\n" % str(sep_deg))
        #import pdb; pdb.set_trace()
        #midx = sep_deg.argmin()         # index of match in stars array
        #sidx = star_ids[midx]
        sidx = sep_deg.argmin()         # index of match in stars array
        #matches.append((star_ids[sidx], gsrc.ra, gsrc.dec))
        matches.append((sidx, gsrc.ra, gsrc.dec))
        #if gi == 1:
        #    sys.stderr.write("len(sep_deg): %d\n" % len(sep_deg))
        #    sys.stderr.write("this_gaia_id: %d\n" % gsrc.source_id)
        #    min_dist = sep_deg[sidx]
        #    sys.stderr.write("min_dist:     %.5f\n" % min_dist)
        #    sys.stderr.write("sidx (stars): %d\n" % sidx)
        pass
 
    if not matches:
        #sys.stderr.write("NO MATCHES!\n")
        return np.array([]), np.array([]), np.array([])

    idx, gra, gde = zip(*matches)
    iorder = np.argsort(idx)
    #return np.array(idx), np.array(gra), np.array(gde)
    #return gaia_ids
    #return hits
    return np.array(idx)[iorder], np.array(gra)[iorder], np.array(gde)[iorder]

#def twoway_gaia_matches_1(stars, tol_arcsec, ra_col='dra', de_col='dde'):
def twoway_gaia_matches_2(ccd_ra, ccd_de, tol_arcsec):
    tol_deg = tol_arcsec / 3600.0
    #matches = []
    star_ids = []
    gaia_ids = []
    #for idx,target in enumerate(stars):
    for idx,(sra, sde) in enumerate(zip(ccd_ra, ccd_de)):
        #sra, sde = target[ra_col], target[de_col]
        #sxx, syy = target[xx_col], target[yy_col]
        result = gm.nearest_star(sra, sde, tol_deg)
        if result['match']:
            gaia_ids.append(int(result['record']['source_id']))
            star_ids.append(idx)
            #import pdb; pdb.set_trace()
            #gcoords = [result['record'][x].values[0] for x in ('ra', 'dec')]
            #matches.append((idx, *gcoords))
            pass
        pass


    # we COULD also grab a subset of 'stars' that matched ...
    #sys.stderr.write("len(stars): %d\n" % len(stars))
    #sys.stderr.write("Of those, %d might match.\n" % len(star_ids))

    # The subset of Gaia stars that we want to use:
    use_gaia = gm._srcdata[gm._srcdata.source_id.isin(gaia_ids)]
    use_ccd_ra = ccd_ra[star_ids]
    use_ccd_de = ccd_de[star_ids]

    # Match the other way:
    #sidx = []
    #hits = []
    matches = []
    for gi,(gix, gsrc) in enumerate(use_gaia.iterrows(), 1):
        sep_deg = angle.dAngSep(gsrc.ra, gsrc.dec,
                use_ccd_ra, use_ccd_de)
                #ccd_ra, ccd_de)
                #stars[ra_col], stars[de_col])
        #sys.stderr.write("sep_deg: %s\n" % str(sep_deg))
        #import pdb; pdb.set_trace()
        midx = sep_deg.argmin()         # index of match in stars array
        sidx = star_ids[midx]
        #sidx = sep_deg.argmin()         # index of match in stars array
        #matches.append((star_ids[sidx], gsrc.ra, gsrc.dec))
        matches.append((sidx, gsrc.ra, gsrc.dec))
        #if gi == 1:
        #    sys.stderr.write("len(use_ccd_ra): %d\n" % len(use_ccd_ra))
        #    sys.stderr.write("len(sep_deg): %d\n" % len(sep_deg))
        #    sys.stderr.write("this_gaia_id: %d\n" % gsrc.source_id)
        #    min_dist = sep_deg[midx]
        #    sys.stderr.write("min_dist:     %.5f\n" % min_dist)
        #    sys.stderr.write("sidx (stars): %d\n" % sidx)
        pass

    if not matches:
        #sys.stderr.write("NO MATCHES!\n")
        return np.array([]), np.array([]), np.array([])

    idx, gra, gde = zip(*matches)
    iorder = np.argsort(idx)
    #return np.array(idx), np.array(gra), np.array(gde)
    #return gaia_ids
    #return hits
    return np.array(idx)[iorder], np.array(gra)[iorder], np.array(gde)[iorder]



##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Configuration:
_xdw_col = 'xdw_cs23'
_ydw_col = 'ydw_cs23'

_xdw_col = 'x'
_ydw_col = 'y'

##--------------------------------------------------------------------------##
## Load Gaia:
gaia_csv_path = 'gaia_calib1_NE.csv'
gm.load_sources_csv(gaia_csv_path)


##--------------------------------------------------------------------------##
## Input file:
use_fcat = 'wircam_H2_1319412p_eph.fits.fz.fcat'
#use_fcat = 'wircam_J_1319395p_eph.fits.fz.fcat'
#use_fcat = 'wircam_H2_1592961p_eph.fits.fz.fcat'
use_fcat = 'wircam_H2_1838760p_eph.fits.fz.fcat'
#use_fcat = 'wircam_H2_2200977p_eph.fits.fz.fcat'
use_fcat = 'wircam_H2_1838749p_eph.fits.fz.fcat'

view_img = use_fcat.replace('p_eph', 'p').replace('fz.fcat', 'fz')

ds9_opts = "-scale limits 6868 7575 -scale sqrt"

## Announce image:
sys.stderr.write("\n%s\n" % halfdiv)
sys.stderr.write("Processing fcat:\n")
sys.stderr.write("--> %s\n" % use_fcat)

## Load ExtCat content:
ecl.load_from_fits(use_fcat)
stars = ecl.get_catalog()
header = ecl.get_header()
obs_time = wwt.wircam_timestamp_from_header(header)
xrel = stars[_xdw_col] - wwt.crpix1
yrel = stars[_ydw_col] - wwt.crpix2
stars = append_fields(stars, ('xrel', 'yrel'), (xrel, yrel), usemask=False)


## Mine header for WCS info:
cdm_pa, cdm_pscale = wwt.get_cdmatrix_pa_scale(header)
cdm_pscale_asec = 3600.0 * cdm_pscale
orig_cv1 = header['CRVAL1']
orig_cv2 = header['CRVAL2']
calc_ra, calc_de = wwt.calc_tan_radec(wwt.pscale, cdm_pa,
                            orig_cv1, orig_cv2, xrel, yrel)
calc_ra = calc_ra % 360.0

sys.stderr.write("\n")
sys.stderr.write("Initial WCS parameters (from header):\n")
sys.stderr.write("orig_cv1 (RA):    %+15.8f deg\n" % orig_cv1)
sys.stderr.write("orig_cv2 (DE):    %+15.8f deg\n" % orig_cv2)
sys.stderr.write("cdm_pscale:       %+15.8f arcsec/pix\n" % cdm_pscale_asec)
sys.stderr.write("cdm_pos_ang:      %+15.8f deg\n" % cdm_pa)

#sys.exit(0)
##--------------------------------------------------------------------------##

## Configure Gaia match:
gm.set_epoch(obs_time)
#gm.set_Gmag_limit(99.0)
#gm.set_Gmag_limit(20.0)
gm.set_Gmag_limit(19.0)

##--------------------------------------------------------------------------##

## Select stars near focal plane center:
corner_dist = 800.0
#corner_dist = 2000.0
lr_stars = wwt.get_corner_subset_dist(stars, corner_dist, minflux=1000)
match_tol = 2.0
#match_tol = 3.0

use_cols = {'ra_col':'dra', 'de_col':'dde'}
lr_gaia_matches = find_gaia_matches_idx(lr_stars, match_tol, **use_cols)
idx, gra, gde   = lr_gaia_matches
match_subset    = lr_stars[idx]
m_xrel, m_yrel  = match_subset['xrel'], match_subset['yrel']

## Make a region file with all the matched detections:
oneway_pix_reg = view_img + '.oneway.pix_reg'
rfy.regify_ccd(oneway_pix_reg, match_subset['x'], match_subset['y'])

## Note typical offset:
sys.stderr.write("\n%s\n" % fulldiv)
med_ra_shift = np.median(match_subset['dra'] - gra)
med_de_shift = np.median(match_subset['dde'] - gde)
ra_nudge_arcsec = 3600.0 * med_ra_shift
de_nudge_arcsec = 3600.0 * med_de_shift
sys.stderr.write("Median RA offset: %+.3f arcsec\n" % ra_nudge_arcsec)
sys.stderr.write("Median DE offset: %+.3f arcsec\n" % de_nudge_arcsec)
sys.stderr.write("%s\n" % fulldiv)

## Make a region file with all the matched Gaia positions:
oneway_sky_reg = view_img + '.oneway.sky_reg'
rfy.regify_sky(oneway_sky_reg, gra, gde, colors=['blue'], rdeg=0.0005)
#rfy.regify_sky(oneway_sky_reg, gra + med_ra_shift, gde + med_de_shift,
#        colors=['blue'], rdeg=0.0005)

##--------------------------------------------------------------------------##

## Check for any duplicate matches:
if len(np.unique(gra)) < len(gra):
    sys.stderr.write("Found non-unique RA ...\n")

## Proper check using sets ...
matched_gaia_radecs = list(zip(gra, gde))
unique_gaia_matches =  set(matched_gaia_radecs)
sys.stderr.write("Total  Gaia matches: %d\n" % len(matched_gaia_radecs))
sys.stderr.write("Unique Gaia matches: %d\n" % len(unique_gaia_matches))

## Hit count for each match:
gaia_hit_counter = {radec:0 for radec in matched_gaia_radecs}
for radec in matched_gaia_radecs:
    gaia_hit_counter[radec] += 1

dupes_counter = {kk:vv for kk,vv in gaia_hit_counter.items() if vv > 1}
#sys.stderr.write("Duplicate Gaia matches:\n")
#for radec in dupes_counter.keys():
#    sys.stderr.write("--> %s\n" % str(radec))


## Test run of two-way matcher:
#lr_uniq_matches  = twoway_gaia_matches_1(lr_stars, match_tol, **use_cols)
#lr_uniq_matches  = twoway_gaia_matches_1(lr_stars['dra'],
lr_uniq_matches  = twoway_gaia_matches_2(lr_stars['dra'],
                                         lr_stars['dde'], match_tol)
idx2, gra2, gde2 = lr_uniq_matches
twoway_subset    = lr_stars[idx2]



## ---------------------------------------------
## Check that module version gives same results:
lr_test_matches  = gm.twoway_gaia_matches(lr_stars['dra'],
                                        lr_stars['dde'], match_tol)

## Evaluate sameness:
idx3, gra3, gde3, gid3 = lr_test_matches
if (len(idx2) != len(idx3)):
    sys.stderr.write("Test failed: unequal match count!\n")
    sys.stderr.write("len(idx2) = %d\n" % len(idx2))
    sys.stderr.write("len(idx3) = %d\n" % len(idx3))
    sys.exit(1)
if not np.all(idx2 == idx3):
    sys.stderr.write("Test failed: match indices differ!\n")
    sys.stderr.write("idx2: %s\n" % str(idx2))
    sys.stderr.write("idx3: %s\n" % str(idx3))
    sys.exit(1)

## ------------------------------------

#sys.exit(0)

## The detections we dropped:
missings = set(idx) - set(idx2)

## Announce de-duplicated sources:
#sys.stderr.write("\n\nindex, X, Y of de-duplicated sources:\n")
#for ii in missings:
#    sys.stderr.write("%4d, %10.5f, %10.5f\n"
#            % (ii, stars[ii]['x'], stars[ii]['y']))


## Make a region file for examination:
dedupe_reg = view_img + '.dedupe.pix_reg'
dedupe_idx = list(missings)
rfy.regify_ccd(dedupe_reg, 
        lr_stars['x'][dedupe_idx], 
        lr_stars['y'][dedupe_idx],
        colors=['red'], rpix=10)

## Prompt for visual confirmation:

sys.stderr.write("\n%s\n" % fulldiv)
sys.stderr.write("Inspect the dropped sources (dense areas) with:\n")
sys.stderr.write("ztf --cfht -r %s -r %s -r %s %s\n"
        % (oneway_pix_reg, oneway_sky_reg, dedupe_reg, view_img))



# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Brute-force CRVAL / PA adjustment and matching:

#test_pas = [-0.3, -0.1

top_ten_idx = np.arange(10)
ntop = 10
ntop = 20
ntop = min(len(gra2), ntop)     # not more than we have points

top_ten_combos = itt.combinations(range(ntop), 2)
top_ten_ij  = np.array(list(itt.combinations(range(ntop), 2)))
top10_idx1, top10_idx2 = top_ten_ij.T
sky_delta_ra = gra2[top10_idx1] - gra2[top10_idx2]
sky_delta_de = gde2[top10_idx1] - gde2[top10_idx2]
sky_delta_ra *= np.cos(np.radians(gde2[top10_idx1]))
sky_angles   = np.arctan2(sky_delta_de, sky_delta_ra)
sky_distance = angle.dAngSep(gra2[top10_idx1], gde2[top10_idx1],
                             gra2[top10_idx2], gde2[top10_idx2])

distance_order = np.argsort(sky_distance)

ccd_ra = twoway_subset['dra']
ccd_de = twoway_subset['dde']

ccd_delta_ra = ccd_ra[top10_idx1] - ccd_ra[top10_idx2]
ccd_delta_de = ccd_de[top10_idx1] - ccd_de[top10_idx2]
ccd_delta_ra *= np.cos(np.radians(ccd_de[top10_idx1]))
ccd_angles   = np.arctan2(ccd_delta_de, ccd_delta_ra)

## Pre-adjustment RA/DE deltas:
old_delta_ra = (gra2 - ccd_ra) * np.cos(np.radians(gde2))
old_delta_de = (gde2 - ccd_de)
old_delta_tot = 3600*np.hypot(old_delta_ra, old_delta_de)

#top_ten_ccd = twoway_subset[:ntop]
#top_ten_gra = gra2[:ntop]
#top_ten_gde = gde2[:ntop]

## Relative segment rotations, in degrees:
segment_angles = np.degrees(sky_angles - ccd_angles)

## Illustrate the unreliability of angle differences from short arcs:
_DO_SHORT_ARC_PLOT = True
if _DO_SHORT_ARC_PLOT:
    import matplotlib.pyplot as plt
    fig = plt.figure(1, figsize=(10, 7))
    fig.clf()
    ax1 = fig.add_subplot(111)
    ax1.grid(True)
    ax1.scatter(sky_distance, segment_angles)
    ax1.set_xlabel('Point-to-Point Separation (deg)')
    ax1.set_ylabel('Segment Rotation (deg)')

## Another way, with roll to use consecutive data points:
roll_sky_delta_ra  =   gra2 - np.roll(gra2, 1)
roll_sky_delta_de  =   gde2 - np.roll(gde2, 1)
roll_sky_delta_ra *=   np.cos(np.radians(gde2))
roll_sky_angles    = np.arctan2(roll_sky_delta_de, roll_sky_delta_ra)

roll_ccd_delta_ra  = ccd_ra - np.roll(ccd_ra, 1)
roll_ccd_delta_de  = ccd_de - np.roll(ccd_de, 1)
roll_ccd_delta_ra *=   np.cos(np.radians(ccd_de))
roll_ccd_angles    = np.arctan2(roll_ccd_delta_de, roll_ccd_delta_ra)

pa_adjustment  = np.degrees(np.median(sky_angles - ccd_angles))
#pa_adjustment = np.median(sky_angles - ccd_angles)
sys.stderr.write("pa_adjustment: %.4f\n" % pa_adjustment)

#cdm_pa_Hband = -3.8559558164649175e-09
#cdm_pa_Jband = -0.05973236686203742


adj_cdm_pa = cdm_pa - pa_adjustment
#adj_cdm_pa = cdm_pa_Jband
#adj_cdm_pscale = cdm_pscale     # no adjustment
#orig_cv1 = header['CRVAL1']
#orig_cv2 = header['CRVAL2']
adj_cv1 = orig_cv1 - med_ra_shift
adj_cv2 = orig_cv2 - med_de_shift
calc_ra, calc_de = wwt.calc_tan_radec(wwt.pscale, adj_cdm_pa,
                            adj_cv1, adj_cv2, 
                            twoway_subset['xrel'], twoway_subset['yrel'])
calc_ra = calc_ra % 360.0


new_delta_ra = (gra2 - calc_ra) * np.cos(np.radians(gde2))
new_delta_de = (gde2 - calc_de)
new_delta_tot = 3600*np.hypot(new_delta_ra, new_delta_de)

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

#match_func = twoway_gaia_matches_1
match_func = twoway_gaia_matches_2

# Count matches at a variety of position angles:
def gmatch_at_pa_crval(xrel, yrel, trial_pa, trial_cv1, trial_cv2, match_tol):
    calc_ra, calc_de = wwt.calc_tan_radec(wwt.pscale, trial_pa,
                                trial_cv1, trial_cv2, xrel, yrel)
    calc_ra = calc_ra % 360.0

    #matches_1 = twoway_gaia_matches_1(calc_ra, calc_de, match_tol)
    #matches_2 = twoway_gaia_matches_2(calc_ra, calc_de, match_tol)
    matches = match_func(calc_ra, calc_de, match_tol)
    #matches = twoway_gaia_matches_2(calc_ra, calc_de, match_tol)
    #import pdb; pdb.set_trace()
    idx, gra, gde = matches
    n_matches = len(idx)
    total_sep = np.nan
    if n_matches >= 1:
        seps = angle.dAngSep(gra, gde, calc_ra[idx], calc_de[idx])
        #sys.stderr.write("seps: %s\n" % str(seps))
        #total_sep = 3600.0 * np.sum(seps)
        total_sep = 3600.0 * np.average(seps)
    return n_matches, total_sep, matches

lr_xrel, lr_yrel = lr_stars['xrel'], lr_stars['yrel']

hits, tsep, _ = gmatch_at_pa_crval(lr_xrel, lr_yrel, 
                            adj_cdm_pa, adj_cv1, adj_cv2, 2)

avg_dec_deg = np.average(lr_stars['dde'])
avg_cos_dec = np.cos(np.radians(avg_dec_deg))
use_mtol = 1.0
nsteps = 7
#nsteps = 3
nhalf  = (nsteps - 1) / 2
steparr = np.arange(nsteps) - nhalf
#steparr = np.arange(
#de_adjustments = 2.0 / 3600.0 * steparr
#ra_adjustments = 2.0 / 3600.0 * steparr
de_adjustments = use_mtol / 3600.0 * steparr
ra_adjustments = use_mtol / 3600.0 * steparr / avg_cos_dec

tik = time.time()
shifted_cv1 = orig_cv1 - med_ra_shift
shifted_cv2 = orig_cv2 - med_de_shift
#shifted_cv1 = orig_cv1
#shifted_cv2 = orig_cv2
hits_matrix = []
hits_max = 0
best_sep = 999.999
best_ra_nudge = 0.0
best_de_nudge = 0.0
best_cv1 = shifted_cv1
best_cv2 = shifted_cv2
sys.stderr.write("Entering nudges loop ...\n")
for ra_nudge in ra_adjustments:
    #use_cv1 = orig_cv1 + ra_nudge
    use_cv1 = shifted_cv1 + ra_nudge
    ra_hitcount = []
    for de_nudge in de_adjustments:
        use_cv2 = shifted_cv2 + de_nudge
        hits, tsep, tmatch = gmatch_at_pa_crval(lr_xrel, lr_yrel,
                adj_cdm_pa, use_cv1, use_cv2, use_mtol)
        sys.stderr.write("Hits=%4d, sep=%.5f at RA/DE nudge %+.4f,%+.4f\n"
                % (hits, tsep, ra_nudge, de_nudge))
        if (hits > hits_max) or ((hits == hits_max) and (tsep < best_sep)):
            hits_max = hits
            best_sep = tsep
            best_ra_nudge = ra_nudge
            best_de_nudge = de_nudge
            best_cv1 = use_cv1
            best_cv2 = use_cv2
            match_info = tmatch
            sys.stderr.write("--> new best!\n")
        #sys.stderr.write("hits: %d\n" % hits)
        ra_hitcount.append(hits)
        pass
    hits_matrix.append(ra_hitcount)

hits_matrix = np.array(hits_matrix)
tok = time.time()
sys.stderr.write("Gridded check took %.3f sec\n" % (tok-tik))
sys.stderr.write("best_ra_nudge:  %+.7f\n" % best_ra_nudge)
sys.stderr.write("best_de_nudge:  %+.7f\n" % best_de_nudge)

best_ra_offset = best_ra_nudge - med_ra_shift
best_de_offset = best_de_nudge - med_de_shift
sys.stderr.write("best_ra_offset: %+.7f\n" % best_ra_offset)
sys.stderr.write("best_de_offset: %+.7f\n" % best_de_offset)

## -----------------------------------------------------------------------
## Make region files showing tune-up stars:

lr_idx, lr_gra, lr_gde = match_info
#lr_sep = angle.dAngSep(calc_ra[lr_idx], calc_de[lr_idx], lr_gra, lr_gde)
matched_dets = lr_stars[lr_idx]

## Gaia matches from tune-up:
tuneup_sky_reg = view_img + '.tuneup.sky_reg'
rfy.regify_sky(tuneup_sky_reg, lr_gra, lr_gde, colors=['blue'], rdeg=0.0005)

## Pixel matches from tune-up:
tuneup_pix_reg = view_img + '.tuneup.pix_reg'
rfy.regify_ccd(tuneup_pix_reg, matched_dets['x'], matched_dets['y'])

## -----------------------------------------------------------------------
## -----------------------------------------------------------------------

## Further CRVAL1, CRVAL2 fine-tune using known-good sources:
sys.stderr.write("%s\n" % fulldiv)
sys.stderr.write("RECOVERING BEST MATCHES ...\n")
calc_ra, calc_de = wwt.calc_tan_radec(wwt.pscale, adj_cdm_pa,
                            best_cv1, best_cv2, lr_xrel, lr_yrel)
calc_ra = calc_ra % 360.0

## RA/DE offsets:
cos_dec = np.cos(np.radians(lr_gde))
tuneup_de_offsets = (calc_de[lr_idx] - lr_gde)
#tuneup_ra_offsets = (calc_ra[lr_idx] - lr_gra) * cos_dec
tuneup_ra_offsets = (calc_ra[lr_idx] - lr_gra)
tuneup_ra_nudge = np.median(tuneup_ra_offsets)
tuneup_de_nudge = np.median(tuneup_de_offsets)

final_cv1 = best_cv1 - tuneup_ra_nudge
final_cv2 = best_cv2 - tuneup_de_nudge

fhits, fsep, fmatch = gmatch_at_pa_crval(lr_xrel, lr_yrel, adj_cdm_pa,
        final_cv1, final_cv2, use_mtol)

# -----------------------------------------------------------------------

# best ra_adjustments: -0.00102, -0.00068
#        med_ra_shift:  0.0003383379843739931
# best de_adjustments: -0.00056, -0.00028
#        med_de_shift:  0.00011615665715680734

cdm_pa_Hband = -3.8559558164649175e-09
cdm_pa_Jband = -0.05973236686203742

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
#
## Produce an image with updated WCS:
sys.stderr.write("%s\n" % fulldiv)
## PA from input image is 'cdm_pa'

#fixed_cdmat = tp.make_cdmat(cdm_pa-0.1, wwt.pscale)
#fixed_cdmat = tp.make_cdmat(cdm_pa_Jband, wwt.pscale)
fixed_cdmat = tp.make_cdmat(adj_cdm_pa, wwt.pscale)
#fixed_cdmat = tp.make_cdmat(cdm_pa+0.05, wwt.pscale)
#fixed_cdmat = tp.make_cdmat(cdm_pa+0.2, wwt.pscale)
fcd11, fcd12, fcd21, fcd22 = fixed_cdmat.flatten()
fixed_image = 'fixed_wcs.fits'
idata, raw_hdrs = pf.getdata(view_img, header=True)
upd_hdrs = raw_hdrs.copy(strip=True)
#upd_hdrs['CRVAL1'] = raw_hdrs['CRVAL1'] - best_ra_offset
#upd_hdrs['CRVAL2'] = raw_hdrs['CRVAL2'] - best_de_offset
#upd_hdrs['CRVAL1'] = best_cv1
#upd_hdrs['CRVAL2'] = best_cv2
upd_hdrs['CRVAL1'] = final_cv1
upd_hdrs['CRVAL2'] = final_cv2
upd_hdrs[ 'CD1_1'] = fcd11
upd_hdrs[ 'CD1_2'] = fcd12
upd_hdrs[ 'CD2_1'] = fcd21
upd_hdrs[ 'CD2_2'] = fcd22
qsave(fixed_image, idata, header=upd_hdrs)

#sys.stderr.write("\n%s\n" % fulldiv)
#sys.stderr.write("Inspect the nudged image WCS with:\n")
#sys.stderr.write("ztf --cfht -r %s -r %s -r %s %s\n"
#        % (oneway_pix_reg, oneway_sky_reg, dedupe_reg, fixed_image))

sys.stderr.write("\n%s\n" % fulldiv)
sys.stderr.write("Inspect the nudged image WCS with:\n")
sys.stderr.write("ztf --cfht -r %s -r %s %s\n"
        % (tuneup_pix_reg, tuneup_sky_reg, fixed_image))

sys.exit(0)




######################################################################
# CHANGELOG (15_matching_tests.py):
#---------------------------------------------------------------------
#
#  2023-10-24:
#     -- Increased __version__ to 0.0.1.
#     -- First created 15_matching_tests.py.
#
