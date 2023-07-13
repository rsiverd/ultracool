#!/usr/bin/env python3
# 
# These are some commands to demonstrate that the Gaia epoch adjustment
# is working.

from importlib import reload
import os, sys, time
import numpy as np
from functools import partial
from numpy.lib.recfunctions import append_fields
import astropy.time as astt

# Imports of my own code:
import gaia_match
reload(gaia_match)
import extended_catalog
reload(extended_catalog)
import angle
reload(angle)
#import tangent_proj
#reload(tangent_proj)

# Short-hand:
#tp = tangent_proj

# Make instances of the classes we plan to use:
gm  = gaia_match.GaiaMatch()
ecl = extended_catalog.ExtendedCatalog()

def instmag(counts, zeropt=25.0):
    return (zeropt - 2.5 * np.log10(counts))

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

# -----------------------------------------------------------------------

# Make instance of ExtendedCatalog to use:
ecl = extended_catalog.ExtendedCatalog()

# Start by first picking one of the extracted "fcat" catalogs. I'll let you 
# choose which one. Starting with the same image you used already is wise:
#path_to_catalog = "/path/to/fcat/file"
#path_to_catalog = "/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/calib1_p_NE/by_runid/11AQ15/wircam_H2_1319397p.fits.fz.fcat"
path_to_catalog = "./wircam_H2_1319397p.fits.fz.fcat"

# Prime the Gaia source matcher:
#gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/gaia_calib1_NE.csv'
gaia_csv_path = './gaia_calib1_NE.csv'
gm.load_sources_csv(gaia_csv_path)

# Load up the catalog, retrieve stars and header:
ecl.load_from_fits(path_to_catalog)
stars = ecl.get_catalog()
header = ecl.get_header()

# The epoch of this image:
obs_time = astt.Time(header['MJD-OBS'], scale='utc', format='mjd') \
        + 0.5 * astt.TimeDelta(header['EXPTIME'], format='sec')

#sys.exit(0)
# Pre-create bestra, bestde columns:
stars = append_fields(stars, ('bestra', 'bestde'),
                            (stars['dra'], stars['dde']), usemask=False)
imags = instmag(stars['flux'])
stars = append_fields(stars, 'instmag', imags, usemask=False)


# find matches with reference epoch:
ref_gaia_matches = find_gaia_matches(stars, 2.0, xx_col='x', yy_col='y')
ref_sx, ref_sy, ref_sra, ref_sde, ref_gra, ref_gde = zip(*ref_gaia_matches)

# find matches with adjusted epoch:
sys.stderr.write("Setting epoch ...\n")
gm.set_epoch(obs_time)

img_gaia_matches = find_gaia_matches(stars, 2.0, xx_col='x', yy_col='y')
img_sx, img_sy, img_sra, img_sde, img_gra, img_gde = zip(*img_gaia_matches)

# use dictionaries and sets to identify overlapping matches:
#img_hits = {}
#for stuff in img_gaia_matches:
#    sx, sy, sra, sde, gra, gde = stuff
#    img_hits[(sx,sy)] = stuff
#    pass

ref_hits = {(x[0],x[1]):x for x in ref_gaia_matches}
img_hits = {(x[0],x[1]):x for x in img_gaia_matches}

ref_xycoo = set(ref_hits.keys())
img_xycoo = set(img_hits.keys())

common_xy = list(set.intersection(ref_xycoo, img_xycoo))

comm_img_hits = [img_hits[x] for x in common_xy]
comm_ref_hits = [ref_hits[x] for x in common_xy]

ci_sx, ci_sy, ci_sra, ci_sde, ci_gra, ci_gde = np.array(comm_img_hits).T
cr_sx, cr_sy, cr_sra, cr_sde, cr_gra, cr_gde = np.array(comm_ref_hits).T

ci_seps = angle.dAngSep(ci_sra, ci_sde, ci_gra, ci_gde)
cr_seps = angle.dAngSep(cr_sra, cr_sde, cr_gra, cr_gde)

sys.exit(0)

