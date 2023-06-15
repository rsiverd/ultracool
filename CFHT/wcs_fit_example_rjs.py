# This file contains some routines that I have used to fit WCS astrometric
# parameters using detected sources. The list of parameters to fit is:
# * CD1_1, CD1_2, CD2_1, CD2_2      (rotation/scale/skew matrix)
# * CRVAL1, CRVAL2                  (RA/DE at reference point)
#
# A basic overview of operation is:
# 1) Develop the forward (X,Y --> RA,DE) routine. (Done, in tangent_proj.py)
# 2) Write an 'evaluator' routine. This is a function that takes a set of
#       parameters (CDi_j, CRVALx) and computes a total godness-of-fit.
# 3) Pick a set of detections (X, Y positions) and corresponding stars 
#       (Gaia RA, DE) to use and provide them to the evaluator.
# 4) Use scipy.optimize routines (I use fmin below) to adjust parameters
#       iteratively until it finds a good answer.
#
# In practice, it looks like we will have poor coordinates in our initial
# images. This means that some of the Gaia matches will be wrong. So when
# we put this procedure in operation, it will fit into an iterative procedure
# that goes something like this:
# A) Extract sources from images assuming initial WCS is okay.
# B) Match our detections from a single image to Gaia.
# C) Improve the initial WCS using the procdure described above (the part you
#       are now working on). 
# D) Using the improved fit, *redo* the matching against Gaia. The results will
#       probably be different after the WCS is improved.
# E) Rerun the optimization using the new-and-improved set of Gaia matches.
# F) Repeat this until convergence (i.e., until the Gaia matches stop changing)
#       


# For now, let's work on the procedure for improving a single image's WCS. You
# will probably need to use a few of the modules we have worked with to run
# these examples. You may need to modify some of this code to make it work
# correctly. The imports:

import os, sys, time
import numpy as np
import scipy.optimize as opti
from functools import partial
import gaia_match
import extended_catalog
import angle
import tangent_proj as tp
from numpy.lib.recfunctions import append_fields

# Make instances of the classes we plan to use:
gm  = gaia_match.GaiaMatch()
ecl = extended_catalog.ExtendedCatalog()

def instmag(counts, zeropt=25.0):
    return (zeropt - 2.5 * np.log10(counts))


# -----------------------------------------------------------------------
# Next, we need to make sure that we have appropriate functions for Gaia
# matching and for WCS parameter optimization. The optimization will make
# use of X,Y positions of stars and RA,DE positions from Gaia. For now,
# we don't need to worry about errors in any of those measurements.
# -----------------------------------------------------------------------

# This is a version of the match-finder that:
# a) has the column names we want to use as the defaults
# b) includes star X,Y positions in its returned product
# c) returns data in the format expected by the evaluator below
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

# This is a version of the match-finder that:
# a) has the column names we want to use as the defaults
# b) includes star X,Y positions in its returned product
# c) returns data in the format expected by the evaluator below
def find_gaia_matches_with_mags(stars, tol_arcsec, ra_col='dra', de_col='dde',
        xx_col='x', yy_col='y'):
    tol_deg = tol_arcsec / 3600.0
    matches = []
    for target in stars:
        sra, sde = target[ra_col], target[de_col]
        sxx, syy = target[xx_col], target[yy_col]
        smag = target['instmag']
        result = gm.nearest_star(sra, sde, tol_deg)
        if result['match']:
            #import pdb; pdb.set_trace()
            gcoords = [result['record'][x].values[0] \
                    for x in ('ra', 'dec', 'phot_rp_mean_mag')]
            gmag = result['record']['phot_rp_mean_mag'].values[0]
            matches.append((smag, gmag))
            #matches.append((sxx, syy, sra, sde, *gcoords))
            pass
        pass
    return matches


# The following is a WCS parameter evaluator for a single image. 
# It needs a few inputs:
# * a set of WCS parameters to evaluate (CD matrix, CRVALs)
# * a list of star X,Y positions we measured
# * a list of corresponding (Gaia) RA, Dec sky positions to compare against
# During minimization, the parameters will be changed repeatedly but the
# lists of X,Y and RA,DE will remain constant.
# IMPORTANT NOTES:
# * parameter order is [CD1_1, CD1_2, CD2_1, CD2_2, CRVAL1, CRVAL2]
# * X,Y,RA,DE variable names have convention that c='catalog' and g='gaia'
def project_pars(wcs_params, xrel, yrel):
    cdmat = wcs_params[:4]
    cv1   = wcs_params[4]
    cv2   = wcs_params[5]
    pra, pde = tp.xycd2radec(cdmat, xrel, yrel, cv1, cv2)
    return (pra % 360.0, pde)

def wcs_par_evaluator(wcs_params, matches):
    #cdmat = wcs_params[:4]
    ##crval = wcs_params[4:].reshape(-1, 2)
    #cv1   = wcs_params[4]
    #cv2   = wcs_params[5]

    # a handy way to convert the match list into numpy arrays:
    cxx, cyy, cra, cde, gra, gde = (np.array(x) for x in zip(*matches))

    ## shift to relative X,Y coordinates:
    ##cxx -= header['CRPIX1']
    ##cyy -= header['CRPIX2']

    ## first, compute the test RA/DE from X,Y positions:
    #tra, tde = tp.xycd2radec(cdmat, cxx, cyy, cv1, cv2)
    tra, tde = project_pars(wcs_params, cxx, cyy)

    # next, compute angular separations of test RA/DE from gaia RA/DE
    deltas = angle.dAngSep(tra, tde, gra, gde)

    # finally, use total angular separation is figure of merit:
    return np.sum(deltas)

# -----------------------------------------------------------------------
# Now that the functions have been defined, we can load up a catalog and
# perform the optimization of its WCS.
# -----------------------------------------------------------------------

# Make instance of ExtendedCatalog to use:
ecl = extended_catalog.ExtendedCatalog()

# Start by first picking one of the extracted "fcat" catalogs. I'll let you 
# choose which one. Starting with the same image you used already is wise:
path_to_catalog = "/path/to/fcat/file"
path_to_catalog = "/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/calib1_p_NE/by_runid/11AQ15/wircam_H2_1319397p.fits.fz.fcat"

# Prime the Gaia source matcher:
gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/gaia_calib1_NE.csv'
gm.load_sources_csv(gaia_csv_path)

# Load up the catalog, retrieve stars and header:
ecl.load_from_fits(path_to_catalog)
stars = ecl.get_catalog()
header = ecl.get_header()

# Pre-create bestra, bestde columns:
stars = append_fields(stars, ('bestra', 'bestde'), 
                            (stars['dra'], stars['dde']), usemask=False)
imags = instmag(stars['flux'])
stars = append_fields(stars, 'instmag', imags, usemask=False)

# We are interested in several of the WCS parameters. Specifically we want
# the CD matrix (rotation/scale/skew) and the CRVALs (RA, Dec zero-point):
want_wcs_pars = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CRVAL1', 'CRVAL2']
header_wcs_params = [header[x] for x in want_wcs_pars]   # this makes a list
crpix1, crpix2 = header['CRPIX1'], header['CRPIX2']

# This routine adds/updates relative X,Y positions in a star catalog:
def update_rel_xy(starcat, crpix1, crpix2, 
        xcol='x', ycol='y', rxcol='xrel', rycol='yrel'):
    updcat = np.copy(starcat)
    new_xrel = starcat[xcol] - crpix1
    new_yrel = starcat[ycol] - crpix2

    # update relative X column:
    if rxcol in updcat.dtype.names:
        sys.stderr.write("catalog already has xrel!\n")
        updcat[rxcol] = new_xrel
    else:
        sys.stderr.write("xrel column not found\n")
        updcat = append_fields(updcat, rxcol, new_xrel, usemask=False)

    # update relative Y column:
    if rycol in updcat.dtype.names:
        sys.stderr.write("catalog already has yrel!\n")
        updcat[rycol] = new_yrel
    else:
        sys.stderr.write("yrel column not found\n")
        updcat = append_fields(updcat, rycol, new_yrel, usemask=False)

    return updcat

# This routine adds/updates best-fit RA/Dec in a star catalog:
def update_bestfit_radec_all(starcat, new_wcs_pars):
    bestra, bestde = project_pars(bestfit_params, starcat['xrel'], starcat['yrel'])
    updcat   = np.copy(starcat)
    updcat['bestra'] = bestra
    updcat['bestde'] = bestde
    return updcat

xcenter, ycenter = 1024.5, 1024.5
stars = update_rel_xy(stars, crpix1, crpix2)
stars = update_rel_xy(stars, xcenter, ycenter)
#stars = update_rel_xy(stars, crpix1, crpix2)

# We can use the initial set of WCS parameters (from the FITS header) as the
# initial guess. These do not seem to be too awful.
initial_guess = np.array(header_wcs_params)
#sys.exit(0)

# Do an initial round of gaia matching:
match_tol_arcsec = 3.0
gaia_matches = find_gaia_matches(stars, match_tol_arcsec, xx_col='xrel', yy_col='yrel')
gaia_matches_mag = find_gaia_matches_with_mags(stars, match_tol_arcsec, xx_col='xrel', yy_col='yrel')

# I like to use the 'partial' routine to automagically handle the non-parameter
# arguments to the evaluator function. The partial routine takes a function
# as its first argument and parameters to that function afterwards, like so:
minimize_this = partial(wcs_par_evaluator, matches=gaia_matches)

# The following should minimize the total badness-of-fit and provide some
# new parameters. 
bestfit_params = opti.fmin(minimize_this, initial_guess)


# TASK: using the new best-fit parameters, re-compute the RA, Dec positions
# of stars detected on the image. Redo the Gaia matches and make a quiver
# plot of the residuals. The result should now look a lot more reasonable 
# than what we saw from the input image.

# Evaluate new solution:
cxx, cyy, cra, cde, gra, gde = (np.array(x) for x in zip(*gaia_matches))
tra, tde = project_pars(bestfit_params, cxx, cyy)

# Coordinate differences:
ra_err = 3600.0 * (gra - tra) * np.cos(np.radians(gde))
de_err = 3600.0 * (gde - tde)

# Look at it:
import matplotlib.pyplot as plt
fig_dims = (12, 11)
fig = plt.figure(1, figsize=fig_dims)
fig.clf()

ax1 = fig.add_subplot(221, aspect='equal')
ax2 = fig.add_subplot(222, aspect='equal')
ax3 = fig.add_subplot(223, aspect='equal')
ax4 = fig.add_subplot(224, aspect='equal')
axlist = [ax1, ax2, ax3, ax4]
for ax in axlist:
    ax.grid(True)

ax1.scatter(ra_err, de_err, lw=0, s=5)
ax2.set_title("Gaia - bestfit")
ax2.quiver(cxx, cyy, ra_err, de_err)


# Update bestra/bestde in catalog:
stars = update_bestfit_radec_all(stars, bestfit_params)

# Re-detect matches:
match_tol_arcsec = 5.0
new_gaia_matches = find_gaia_matches(stars, match_tol_arcsec, xx_col='xrel', yy_col='yrel')
new_minimize_this = partial(wcs_par_evaluator, matches=new_gaia_matches)
new_bestfit_params = opti.fmin(new_minimize_this, bestfit_params)

ncxx, ncyy, ncra, ncde, ngra, ngde = (np.array(x) for x in zip(*new_gaia_matches))
ntra, ntde = project_pars(new_bestfit_params, ncxx, ncyy)

# Coordinate differences:
new_ra_err = 3600.0 * (ngra - ntra) * np.cos(np.radians(ngde))
new_de_err = 3600.0 * (ngde - ntde)

ax3.scatter(new_ra_err, new_de_err, lw=0, s=5)
ax4.quiver(ncxx, ncyy, new_ra_err, new_de_err)

fig.tight_layout()
plt.draw()
#plt.show()

