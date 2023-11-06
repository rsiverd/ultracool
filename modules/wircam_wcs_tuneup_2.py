#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Helper routine to fine-tune core WCS parameters (tangent projection)
# using known distortion solution and dubious CFHT/WIRCam header data.
#
# Rob Siverd
# Created:       2023-07-26
# Last modified: 2023-11-06
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

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
reload(angle)

import gaia_match
reload(gaia_match)
gm  = gaia_match.GaiaMatch()
#gm2 = gaia_match.GaiaMatch()

#import extended_catalog
#ecl = extended_catalog.ExtendedCatalog()

import wircam_fs_helpers as wfh
reload(wfh)

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

## -----------------------------------------------------------------------

## WCS defaults:
crpix1 = 2122.690779
crpix2 =  -81.678888
pscale =   0.30601957084155673

## Which columns to use:
_xdw_col = 'xdw_cs23'
_ydw_col = 'ydw_cs23'
#_xdw_col = 'xdw_dl12'
#_ydw_col = 'ydw_dl12'

## Sensor size:
sensor_xpix = 2048
sensor_ypix = 2048

corner_size = 256
corner_size = 512
corner_xmin = sensor_xpix - corner_size
corner_ymax = corner_size

radial_dist = 1.4142 * corner_size

## -----------------------------------------------------------------------
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

## Select lower-right sources within fixed distance from focal plane center.
## Impose flux cut to avoid spurious matches.
#def get_corner_subset_dist(data, cutoff, minflux=600.0):
def get_corner_subset_dist(data, cutoff, minflux=2000.0):
    rdist = np.hypot(data['x'] - crpix1, data['y'] - crpix2)
    flxok = (data['flux'] > minflux)
    return data[(rdist <= cutoff) & flxok]

## Select objects near the focal plane center:
def get_central_sources(data, max_rsep_pix):
    rsep = np.hypot(data['x'] - crpix1, data['y'] - crpix2)
    return data[rsep <= max_rsep_pix]

## -----------------------------------------------------------------------
## Gaia matching routine:
#def find_gaia_matches(stars, tol_arcsec, ra_col='dra', de_col='dde',
#        xx_col='x', yy_col='y'):
#    tol_deg = tol_arcsec / 3600.0
#    matches = []
#    for target in stars:
#        sra, sde = target[ra_col], target[de_col]
#        sxx, syy = target[xx_col], target[yy_col]
#        result = gm.nearest_star(sra, sde, tol_deg)
#        if result['match']:
#            gcoords = [result['record'][x].values[0] for x in ('ra', 'dec')]
#            matches.append((sxx, syy, sra, sde, *gcoords))
#            pass
#        pass
#    return matches

#def find_gaia_matches_idx_stars(stars, tol_arcsec, ra_col, de_col): #ra_col='dra', de_col='dde'):
#    tol_deg = tol_arcsec / 3600.0
#    matches = []
#    for idx,target in enumerate(stars):
#        sra, sde = target[ra_col], target[de_col]
#        #sxx, syy = target[xx_col], target[yy_col]
#        result = gm.nearest_star(sra, sde, tol_deg)
#        if result['match']:
#            gcoords = [result['record'][x].values[0] for x in ('ra', 'dec')]
#            matches.append((idx, *gcoords))
#            pass
#        pass
#    idx, gra, gde = zip(*matches)
#    return np.array(idx), np.array(gra), np.array(gde)
#
#def find_gaia_matches_idx_radec(ra_vals, de_vals, tol_arcsec):
#    tol_deg = tol_arcsec / 3600.0
#    matches = []
#    for idx,(sra, sde) in enumerate(zip(ra_vals, de_vals)):
#        #sra, sde = target[ra_col], target[de_col]
#        #sxx, syy = target[xx_col], target[yy_col]
#        result = gm.nearest_star(sra, sde, tol_deg)
#        if result['match']:
#            gcoords = [result['record'][x].values[0] for x in ('ra', 'dec')]
#            matches.append((idx, *gcoords))
#            pass
#        pass
#    idx, gra, gde = zip(*matches)
#    return np.array(idx), np.array(gra), np.array(gde)

## -----------------------------------------------------------------------
## Fitting procedure:
def calc_tan_radec(pscale, pa_deg, cv1, cv2, xrel, yrel):
    this_cdmat = tp.make_cdmat(pa_deg, pscale)
    return tp.xycd2radec(this_cdmat, xrel, yrel, cv1, cv2)

def eval_tan_params(pscale, pa_deg, cv1, cv2,
                            xrel, yrel, true_ra, true_de, expo=1.0):
    calc_ra, calc_de = calc_tan_radec(pscale, pa_deg, cv1, cv2, xrel, yrel)
    deltas = angle.dAngSep(calc_ra, calc_de, true_ra, true_de)
    return np.sum(deltas**expo)

def evaluator_pacrv(pacrv, pscale, xrel, yrel, true_ra, true_de, expo=1.0):
    pa_deg, cv1, cv2 = pacrv
    return eval_tan_params(pscale, pa_deg, cv1, cv2,
            xrel, yrel, true_ra, true_de, expo=expo)

## Convert the answer into CD matrix values:
def cdmat_from_answer(answer):
    pa_deg, cv1, cv2 = answer

## -----------------------------------------------------------------------

## Analyze CD matrix from header:
_cd_keys = ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')
def get_cdmatrix_pa_scale(header, debug=False):
    orig_cdm = np.array([header[x] for x in _cd_keys]).reshape(2, 2)
    cd_xyscl = np.sqrt(np.sum(orig_cdm**2, axis=1))
    norm_cdm = orig_cdm / cd_xyscl
    #norm_rot = np.dot(tp.xflip_mat, norm_cdm)
    norm_rot = np.dot(norm_cdm, tp.xflip_mat)
    flat_rot = norm_rot.flatten()
    pa_guess = [math.acos(flat_rot[0]), -math.asin(flat_rot[1]),
                        math.asin(flat_rot[2]), math.acos(flat_rot[3])]
    if debug:
        sys.stderr.write("cd_xyscl: %s\n" % str(cd_xyscl))
        sys.stderr.write("cds_asec: %s\n" % str(3600*cd_xyscl))
        sys.stderr.write("pa_guess: %s\n" % str(pa_guess))
    pos_ang  = np.degrees(np.average(pa_guess))
    pos_ang *= -1.0     # fix direction
    pixscale = np.average(cd_xyscl)
    return pos_ang, pixscale

## -----------------------------------------------------------------------

## Save X,Y,RA,DE matches for later analysis:
def save_gaia_matches(filename, col_names, data_vecs):
    #import pdb; pdb.set_trace()
    with open(filename, 'w') as gmf:
        # write header:
        gmf.write("%s\n" % ','.join(col_names))
        # write data:
        for vals in zip(*data_vecs):
            gmf.write("%s\n" % ','.join([str(x) for x in vals]))
        pass
    return

## Save PA / CRVALx iterations for later analysis:
def save_wcspar_results(filename, col_names, data_vals):
    with open(filename, 'w') as wrf:
        # write header:
        wrf.write("%s\n" % ','.join(col_names))
        # write data line:
        wrf.write("%s\n" % ','.join([str(x) for x in data_vals]))
        pass
    return

## -----------------------------------------------------------------------
## -----------------          Tune-Up Helpers               --------------
## -----------------------------------------------------------------------

## Calculate total, average, and median angular separation between two
## sets of RA/DE points. Inputs in DEGREES.
def calc_angsep_stats(ra1, de1, ra2, de2):
    delta_ra = (ra2 - ra1) * np.cos(np.radians(de2))
    delta_de = (de2 - de1)
    total_sep = 3600*np.hypot(delta_ra, delta_de)
    avg_delta = np.average(total_sep)
    med_delta = np.median(total_sep)
    tot_delta = np.sum(total_sep)
    return len(delta_ra), avg_delta, med_delta, tot_delta

## Compute PA of two-point segments:
def calc_seg_angles(ra, de, idx1, idx2):
    delta_ra = ra[idx1] - ra[idx2]
    delta_de = de[idx1] - de[idx2]
    #midpt_de = 0.5 * (de[idx1] + de[idx2])
    #delta_ra *= np.cos(np.radians(midpt_de))
    delta_ra *= np.cos(np.radians(de[idx1]))
    return np.arctan2(delta_de, delta_ra)

## Compute the set of segment rotations to rotationally align two data sets:
def calc_pa_rotation(det_ra, det_de, cat_ra, cat_de, nmax=20):
    n_use = min(len(det_ra), nmax)
    #top_combos = itt.combinations(range(n_use), 2)
    sys.stderr.write("n_use: %d\n" % n_use)
    top_few_ij = np.array(list(itt.combinations(range(n_use), 2)))
    #sys.stderr.write("top_few_ij: %s\n" % str(top_few_ij))
    top_idx1, top_idx2 = top_few_ij.T
    cat_angles = calc_seg_angles(cat_ra, cat_de, top_idx1, top_idx2)
    #sys.stderr.write("cat_angles: %s\n" % str(cat_angles))
    det_angles = calc_seg_angles(det_ra, det_de, top_idx1, top_idx2)
    #sys.stderr.write("det_angles: %s\n" % str(det_angles))
    seg_deltas = np.degrees(cat_angles - det_angles)
    return seg_deltas


## Count matches at a specific trial (PA, CRVAL1, CRVAL2) parameter set:
def gmatch_at_pa_crval(xrel, yrel, trial_pa, trial_cv1, trial_cv2, match_tol):
    calc_ra, calc_de = calc_tan_radec(pscale, trial_pa,
                                trial_cv1, trial_cv2, xrel, yrel)
    calc_ra = calc_ra % 360.0

    matches = gm.twoway_gaia_matches(calc_ra, calc_de, match_tol)
    idx, gra, gde, gid = matches
    n_matches = len(idx)
    total_sep = np.nan
    if n_matches >= 1:
        seps = angle.dAngSep(gra, gde, calc_ra[idx], calc_de[idx])
        #sys.stderr.write("seps: %s\n" % str(seps))
        #total_sep = 3600.0 * np.sum(seps)
        total_sep = 3600.0 * np.average(seps)
        sys.stderr.write("seps: %s\n" % str(3600*seps))
    return n_matches, total_sep, matches

## Brute-force CRVAL tune-up:
#def brute_force_crval_tuneup(calc_ra, calc_de, 
def brute_force_crval_tuneup(xrel, yrel, nominal_pa, nominal_cv1, nominal_cv2,
        use_mtol=1.0, nsteps=7):
    hits_matrix = []
    hits_max = 0
    best_sep = 999.999
    #best_ra_nudge = 0.0
    #best_de_nudge = 0.0
    #use_mtol = 1.0
    #nsteps = 7
    best_cv1 = nominal_cv1
    best_cv2 = nominal_cv2
    avg_cos_dec = np.cos(np.radians(nominal_cv2))
    nhalf = (nsteps - 1) / 2
    steparr = np.arange(nsteps) - nhalf
    de_adjustments = use_mtol / 3600.0 * steparr
    ra_adjustments = use_mtol / 3600.0 * steparr / avg_cos_dec
    for ra_nudge in ra_adjustments:
        use_cv1 = nominal_cv1 + ra_nudge
        ra_hitcount = []
        for de_nudge in de_adjustments:
            use_cv2 = nominal_cv2 + de_nudge
            hits, tsep, tmatch = gmatch_at_pa_crval(xrel, yrel,
                    nominal_pa, use_cv1, use_cv2, use_mtol)
            sys.stderr.write("Hits=%4d, sep=%.5f at RA/DE nudge %+.4f,%+.4f\n"
                    % (hits, tsep, ra_nudge, de_nudge))
            if (hits > hits_max) or ((hits == hits_max) and (tsep < best_sep)):
                hits_max = hits
                best_sep = tsep
                #best_ra_nudge = ra_nudge
                #best_de_nudge = de_nudge
                best_cv1 = use_cv1
                best_cv2 = use_cv2
                match_info = tmatch
                sys.stderr.write("--> new best!\n")
            #sys.stderr.write("hits: %d\n" % hits)
            ra_hitcount.append(hits)
            pass
        hits_matrix.append(ra_hitcount)
        pass
    hits_matrix = np.array(hits_matrix)
    return best_cv1, best_cv2, match_info, hits_matrix



## -----------------------------------------------------------------------
## -----------------         Tune-Up Procedure              --------------
## -----------------------------------------------------------------------

## Gaia match radius and focal plane distance for each round:
_fitting_spec = (
        {'tol_arcsec':2.0, 'rdist':radial_dist},
        {'tol_arcsec':2.0, 'rdist':2000.0     },
        )

def wcs_tuneup(stars, header, save_matches=None, save_wcspars=None, 
        pixreg1=None, skyreg1=None, pixreg2=None, skyreg2=None,
        pixreg3=None, skyreg3=None):

    obs_time = wircam_timestamp_from_header(header)

    # Warn if regions requested but not available:
    if not _have_region_utils:
        for rfile in (pixreg1, skyreg1, pixreg2, skyreg2, pixreg3, skyreg3):
            if rfile:
                sys.stderr.write("WARNING: %s requested" % rfile
                    + " but not possible\n")

    # column headings and 
    wcspar_cols = []
    wcspar_vals = []

    #xrel = stars['xdw_cs23'] - crpix1
    #yrel = stars['ydw_cs23'] - crpix2
    xrel  = stars[_xdw_col] - crpix1
    yrel  = stars[_ydw_col] - crpix2
    stars = append_fields(stars, ('xrel', 'yrel'), 
                                (xrel, yrel), usemask=False)

    cdm_pa, cdm_pscale = get_cdmatrix_pa_scale(header)
    #cdm_pa = 0.0

    # Note PA+CRVALx from header:
    wcspar_cols.extend(['hdr_pa_deg', 'hdr_crval1', 'hdr_crval2'])
    wcspar_vals.extend([cdm_pa, header['CRVAL1'], header['CRVAL2']])

    # settings for initial matching:
    gm.set_epoch(obs_time)
    gm.set_Gmag_limit(19.0)

    # initialize "calc" columns:
    stars = append_fields(stars, ('calc_ra', 'calc_de'),
            (np.copy(stars['dra']), np.copy(stars['dde'])), usemask=False)
    radec_cols = {'ra_col':'calc_ra', 'de_col':'calc_de'}

    # Initial corner matching:
    initial_mtol_asec  = 3.0
    initial_min_flux   = 1000.0
    initial_corner_sep =  800.0
    #lr_stars         = get_corner_subset_rect(stars)
    lr_stars           = get_corner_subset_dist(stars, initial_corner_sep,
            minflux=initial_min_flux)
    #lr_gaia_matches  = find_gaia_matches_idx_stars(lr_stars, 2.0, **radec_cols)
    #lr_gaia_matches  = find_gaia_matches_idx_stars(lr_stars, 3.0, **radec_cols)
    lr_gaia_matches    = gm.twoway_gaia_matches(
            lr_stars['calc_ra'], lr_stars['calc_de'], initial_mtol_asec)
    idx, gra, gde, gid = lr_gaia_matches
    if not idx:
        sys.stderr.write("NO MATCHES!!!???\n")
        sys.exit(1)
    match_subset       = lr_stars[idx]
    m_xrel, m_yrel     = match_subset['xrel'], match_subset['yrel']

    # Calculate bulk CRVALs offsets from matched subset:
    med_ra_shift = np.median(match_subset['calc_ra'] - gra)
    med_de_shift = np.median(match_subset['calc_de'] - gde)

    # Brute-force CRVAL tune-up (to nearest grid cell):
    guess_cv1 = header['CRVAL1'] - med_ra_shift
    guess_cv2 = header['CRVAL2'] - med_de_shift
    guess_pa  = cdm_pa
    #lr_xrel, lr_yrel   = lr_stars['xrel'], lr_stars['yrel']
    #brute_mtol = 1.0
    brute_mtol = 2.0
    best_cv1, best_cv2, match_info, hits_matrix = \
        brute_force_crval_tuneup(lr_stars['xrel'], lr_stars['yrel'],
                    guess_pa, guess_cv1, guess_cv2, use_mtol=brute_mtol)
    sys.stderr.write("hits_matrix:\n")
    sys.stderr.write("%s\n" % str(hits_matrix))
 
    # Subtract remaining bulk offset:
    lr_idx, lr_gra, lr_gde = match_info[:3]
    calc_ra, calc_de = calc_tan_radec(pscale, guess_pa, best_cv1, best_cv2, 
            lr_stars['xrel'], lr_stars['yrel'])
    tuneup_ra_nudge = np.median(calc_ra[lr_idx] - lr_gra)
    tuneup_de_nudge = np.median(calc_de[lr_idx] - lr_gde)
    final_cv1 = best_cv1 - tuneup_ra_nudge
    final_cv2 = best_cv2 - tuneup_de_nudge
    fhits, fsep, fmatch = gmatch_at_pa_crval(lr_stars['xrel'], lr_stars['yrel'],
        guess_pa, final_cv1, final_cv2, brute_mtol)

    # Now optimize from a better starting point:
    #init_params1       = np.array([cdm_pa, header['CRVAL1'], header['CRVAL2']])
    init_params1       = np.array([guess_pa, final_cv1, final_cv2])
    minimize_this      = partial(evaluator_pacrv, pscale=pscale, xrel=m_xrel,
                                    yrel=m_yrel, true_ra=gra, true_de=gde)
    answer1            = opti.fmin(minimize_this, init_params1)

    # Make region files if requested:
    if _have_region_utils:
        if pixreg1:
            rfy.regify_ccd(pixreg1, match_subset['x'], match_subset['y'],
                    colors=rfy.colorset, vlevel=1)
        if skyreg1:
            rfy.regify_sky(skyreg1, gra, gde, colors=rfy.colorset, vlevel=1)


    # Note fit results:
    wcspar_cols.extend(['fit1_pa_deg', 'fit1_crval1', 'fit1_crval2'])
    wcspar_vals.extend(answer1.tolist())
    sys.stderr.write("wcspar_cols: %s\n" % str(wcspar_cols))
    sys.stderr.write("wcspar_vals: %s\n" % str(wcspar_vals))

    # Re-calculate RA/DE:
    #calc_ra, calc_de = calc_tan_radec(pscale, best_pa, best_cv1, best_cv2, xrel, yrel)
    calc_ra, calc_de = calc_tan_radec(pscale, *answer1, xrel, yrel)
    stars['calc_ra'] = calc_ra % 360.0
    stars['calc_de'] = calc_de

    # -------------------------

    # Repeat tuning with larger region:
    lr_stars           = get_corner_subset_dist(stars, 2000.0)
    #lr_gaia_matches  = find_gaia_matches_idx_stars(lr_stars, 2.0, **radec_cols)
    #lr_gaia_matches  = find_gaia_matches_idx_stars(lr_stars, 3.0, **radec_cols)
    lr_gaia_matches    = gm.twoway_gaia_matches(
            lr_stars['calc_ra'], lr_stars['calc_de'], 3.0)
    idx, gra, gde, gid = lr_gaia_matches
    match_subset       = lr_stars[idx]
    m_xrel, m_yrel     = match_subset['xrel'], match_subset['yrel']
    init_params2       = np.copy(answer1)
    minimize_this      = partial(evaluator_pacrv, pscale=pscale, xrel=m_xrel,
                                          yrel=m_yrel, true_ra=gra, true_de=gde)
    answer2            = opti.fmin(minimize_this, init_params2)

    # Make region files if requested:
    if _have_region_utils:
        if pixreg2:
            rfy.regify_ccd(pixreg2, match_subset['x'], match_subset['y'],
                    colors=rfy.colorset, vlevel=1)
        if skyreg2:
            rfy.regify_sky(skyreg2, gra, gde, colors=rfy.colorset, vlevel=1)

    # Note fit results:
    wcspar_cols.extend(['fit2_pa_deg', 'fit2_crval1', 'fit2_crval2'])
    wcspar_vals.extend(answer2.tolist())
    sys.stderr.write("wcspar_cols: %s\n" % str(wcspar_cols))
    sys.stderr.write("wcspar_vals: %s\n" % str(wcspar_vals))

    # Re-calculate RA/DE:
    calc_ra, calc_de = calc_tan_radec(pscale, *answer2, xrel, yrel)
    stars['calc_ra'] = calc_ra % 360.0
    stars['calc_de'] = calc_de

    # -------------------------

    # Repeat tuning with full image:
    ff_stars           = stars
    #ff_gaia_matches  = find_gaia_matches_idx_stars(ff_stars, 2.0, **radec_cols)
    #ff_gaia_matches  = find_gaia_matches_idx_stars(ff_stars, 3.0, **radec_cols)
    ff_gaia_matches    = gm.twoway_gaia_matches(
            ff_stars['calc_ra'], ff_stars['calc_de'], 3.0)
    idx, gra, gde, gid = ff_gaia_matches
    match_subset       = ff_stars[idx]
    m_xrel, m_yrel     = match_subset['xrel'], match_subset['yrel']
    init_params3       = np.copy(answer2)
    minimize_this      = partial(evaluator_pacrv, pscale=pscale, xrel=m_xrel,
                                          yrel=m_yrel, true_ra=gra, true_de=gde)
    answer3            = opti.fmin(minimize_this, init_params3)

    # Make region files if requested:
    if _have_region_utils:
        if pixreg3:
            rfy.regify_ccd(pixreg3, match_subset['x'], match_subset['y'],
                    colors=rfy.colorset, vlevel=1)
        if skyreg3:
            rfy.regify_sky(skyreg3, gra, gde, colors=rfy.colorset, vlevel=1)

    # Note fit results:
    wcspar_cols.extend(['fit3_pa_deg', 'fit3_crval1', 'fit3_crval2'])
    wcspar_vals.extend(answer3.tolist())
    sys.stderr.write("wcspar_cols: %s\n" % str(wcspar_cols))
    sys.stderr.write("wcspar_vals: %s\n" % str(wcspar_vals))

    # Re-calculate RA/DE:
    calc_ra, calc_de = calc_tan_radec(pscale, *answer3, xrel, yrel)
    stars['calc_ra'] = calc_ra % 360.0
    stars['calc_de'] = calc_de

    # -------------------------

    # Prepare match data for storage:
    if save_matches:
        matched_x = match_subset['x']
        matched_y = match_subset['y']
        save_cols = ['x', 'y', 'xrel', 'yrel', 'gra', 'gde']
        save_vecs = [matched_x, matched_y, m_xrel, m_yrel, gra, gde]
        save_gaia_matches(save_matches, save_cols, save_vecs)

    # Dump fitted WCS parameters on request:
    if save_wcspars:
        save_wcspar_results(save_wcspars, wcspar_cols, wcspar_vals)

    # -------------------------

    # TO-DO while solving:
    # * evaluate match distances and drop things?
    # * prune duplicate matches?

    # TO-DO AT END:
    # * one more round of Gaia association with updated RA/DE
    # * prune final associations (outliers should be clear)
    # * include matched Gaia IDs in stars data table

    # -------------------------
    # Added value for catalogs
    # -------------------------

    # Size of stars catalog:
    nstars = len(stars)

    # New column names and arrays:
    add_cnames = []
    add_arrays = []

    # Record the number of Gaia matches:
    ngaia = len(idx)
    add_cnames.append('ngaia')
    add_arrays.append(np.repeat(ngaia, nstars))

    # Include match RMS as diagnostic??

    # Append the position angle:
    best_pa, best_crval1, best_crval2 = answer3
    add_cnames.append('pa_deg')
    add_arrays.append(np.repeat(best_pa, nstars))

    # Append PA and CD matrix info to catalog:
    best_cdmat = tp.make_cdmat(best_pa, pscale).flatten()
    add_cnames.extend(['cd11', 'cd12', 'cd21', 'cd22'])
    add_arrays.extend([np.repeat(x, nstars) for x in best_cdmat])

    # Update the catalog:
    stars = append_fields(stars, add_cnames, add_arrays, usemask=False)

    # -------------------------

    return stars
    

