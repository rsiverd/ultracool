#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# This module sets aside the relative PA rotation and CRVAL1/2 offset
# calculations for testing before incorporation into the WCS tuneup module.
#
# Rob Siverd
# Created:       2023-11-06
# Last modified: 2023-11-06
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

import itertools as itt
import numpy as np
import os, sys, time

import angle
#reload(angle)

import wircam_wcs_tuneup
#reload(wircam_wcs_tuneup)
wwt  = wircam_wcs_tuneup

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

def calc_seg_angles(ra, de, idx1, idx2):
    delta_ra = ra[idx1] - ra[idx2]
    delta_de = de[idx1] - de[idx2]
    midpt_de = 0.5 * (de[idx1] + de[idx2])
    delta_ra *= np.cos(np.radians(midpt_de))
    #delta_ra *= np.cos(np.radians(de[idx1]))
    return np.arctan2(delta_de, delta_ra)

def calc_pa_rotations(det_ra, det_de, cat_ra, cat_de, nmax=20):
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

### Calculate an improved position angle using two-point segments
### drawn among matched sources.
#def improve_pos_ang_segments(current_pa_deg, 
#        ccd_ra, ccd_de, cat_ra, cat_de, nmax=20):
#    seg_deltas = calc_pa_rotations(ccd_ra, ccd_de, cat_ra, cat_de, nmax=nmax)
#    adjustment = np.median(seg_deltas)
#    return current_pa_deg - adjustment

### Iterate on WCS parameters once (PA, CRVAL1, CRVAL2):
#def iterate_pa_crvals(current_pa_deg, current_crval1, current_crval2,
#        ccd_xrel, ccd_yrel, cat_ra, cat_de):
#
#    # First, calculate current detection RA/DE using current params:
#    now_ccd_ra, now_ccd_de = wwt.calc_tan_radec(wwt.pscale, current_pa_deg,
#                            current_cv1, current_cv2, ccd_xrel, ccd_yrel)
#    now_ccd_ra = now_ccd_ra % 360.0
#
#    # Get updated position angle from two-point segments:
#    new_pa_deg = improve_pos_ang_segments(current_pa_deg, 
#            now_ccd_ra, now_ccd_de, cat_ra, cat_de)
#
#    
#    # first, update PA using segments in RA/DE space:
#    new_pa_deg = improve_pos_ang_segments(current_pa_deg, 
#                                    current_ccd_ra, ccd_de, cat_ra, cat_de)
#

def iter_update_wcs_pars(initial_pa_deg, initial_crval1, initial_crval2,
                                        ccd_xrel, ccd_yrel, cat_ra, cat_de,
                                        passes=3, nmax=20):

    ## Calculate detection RA/DE:
    #calc_ccd_ra, calc_ccd_de = wwt.calc_tan_radec(wwt.pscale, initial_pa_deg,
    #                    initial_crval1, initial_crval2, ccd_xrel, ccd_yrel)
    #calc_ccd_ra = calc_ccd_ra % 360.0

    current_pa_deg = initial_pa_deg
    current_crval1 = initial_crval1
    current_crval2 = initial_crval2
    for ii in range(passes):
        sys.stderr.write("iteration %d of %d ...\n" % (ii+1, passes))

        # Update detection RA/DE:
        calc_ccd_ra, calc_ccd_de = wwt.calc_tan_radec(wwt.pscale, current_pa_deg,
                            current_crval1, current_crval2, ccd_xrel, ccd_yrel)
        calc_ccd_ra = calc_ccd_ra % 360.0

        # Get updated position angle from two-point segments (rotation):
        #new_pa_deg = improve_pos_ang_segments(current_pa_deg, 
        #        calc_ccd_ra, calc_ccd_de, cat_ra, cat_de)
        #seg_deltas = improve_pos_ang_segments(current_pa_deg, 
        #        calc_ccd_ra, calc_ccd_de, cat_ra, cat_de)
        seg_deltas = calc_pa_rotations(calc_ccd_ra, calc_ccd_de, 
                cat_ra, cat_de, nmax=nmax)
        adjustment = np.median(seg_deltas)
        current_pa_deg -= adjustment

        # Update detection RA/DE:
        calc_ccd_ra, calc_ccd_de = wwt.calc_tan_radec(wwt.pscale, current_pa_deg,
                            current_crval1, current_crval2, ccd_xrel, ccd_yrel)
        calc_ccd_ra = calc_ccd_ra % 360.0

        # Compute RA/DE offsets (translation):
        ra_nudge = np.median(calc_ccd_ra - cat_ra)
        de_nudge = np.median(calc_ccd_de - cat_de)
        current_crval1 -= ra_nudge
        current_crval2 -= de_nudge
        pass

    # Return the updated parameters:
    return current_pa_deg, current_crval1, current_crval2

### Given a list of matches (relative detector X,Y and catalog RA,DE),
### calculate a rotation adjustment to bring matches into better alignment.
#def improve_pos_ang_polar(current_pa_deg, current_crval1, current_crval2,
#        ccd_X_rel, ccd_Y_rel, cat_ra, cat_de):
#
#    # First, de-project catalog RA/DE onto detector coordinates using
#    # current PA and CRVALs. Convert to polar (R, theta) coordinates:
#    current_cdmat = tp.make_cdmat(current_pa_deg, wwt.pscale)
#    inv_X_rel, inv_Y_rel = wwt.tp.sky2xy_cd(current_cdmat, 
#            cat_ra, cat_de, current_crval1, current_crval2)
#    inv_R_rel = np.hypot(inv_X_rel, inv_Y_rel)
#    inv_theta = np.degrees(np.arctan2(inv_Y_rel, inv_X_rel))
#
#    # Next, convert relative detector X,Y coordinates to polar:
#    ccd_R_rel = np.hypot(ccd_X_rel, ccd_Y_rel)
#    ccd_theta = np.degrees(np.arctan2(ccd_Y_rel, ccd_X_rel))
#
#    # Accept the median difference in thetas as the adjustment:
#    theta_diff = np.median(ccd_theta - inv_theta)
#    new_pa_deg = current_pa_deg - theta_diff
#    return new_pa_deg


## Count matches at a variety of position angles:
def gmatch_at_pa_crval(xrel, yrel, trial_pa, trial_cv1, trial_cv2, match_tol, gm):
    calc_ra, calc_de = wwt.calc_tan_radec(wwt.pscale, trial_pa,
                                trial_cv1, trial_cv2, xrel, yrel)
    calc_ra = calc_ra % 360.0

    #matches_1 = twoway_gaia_matches_1(calc_ra, calc_de, match_tol)
    #matches_2 = twoway_gaia_matches_2(calc_ra, calc_de, match_tol)
    #matches = match_func(calc_ra, calc_de, match_tol)
    matches = gm.twoway_gaia_matches(calc_ra, calc_de, match_tol)
    #matches = twoway_gaia_matches_2(calc_ra, calc_de, match_tol)
    #import pdb; pdb.set_trace()
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
def brute_force_crval_tuneup(xrel, yrel,
        nominal_pa, nominal_cv1, nominal_cv2, gm,
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
                    nominal_pa, use_cv1, use_cv2, use_mtol, gm)
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





