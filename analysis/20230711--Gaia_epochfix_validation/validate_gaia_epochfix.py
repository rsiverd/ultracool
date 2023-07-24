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
            gcoords = [result['record'][x].values[0] for x in ('ra', 'dec', 'source_id')]
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
ref_sx, ref_sy, ref_sra, ref_sde, ref_gra, ref_gde, ref_gid = zip(*ref_gaia_matches)

# recover Gaia catalog entries (ref):
r_hits = gm._srcdata.source_id.isin(ref_gid)
r_matches = gm._srcdata[r_hits]
r_speeds = np.hypot(r_matches['pmra'], r_matches['pmdec'])
#fastest = speeds.argmax()
r_top_ten_idx = np.argsort(r_speeds)[-10:]
#fastpar = matches.iloc[fastest]
r_top_ten_cat = r_matches.iloc[r_top_ten_idx]

# find matches with adjusted epoch:
sys.stderr.write("Setting epoch ...\n")
gm.set_epoch(obs_time)

img_gaia_matches = find_gaia_matches(stars, 2.0, xx_col='x', yy_col='y')
img_sx, img_sy, img_sra, img_sde, img_gra, img_gde, img_gid = zip(*img_gaia_matches)

# recover Gaia catalog entries (ref):
i_hits = gm._srcdata.source_id.isin(img_gid)
i_matches = gm._srcdata[i_hits]
i_speeds = np.hypot(i_matches['pmra'], i_matches['pmdec'])
#fastest = speeds.argmax()
i_top_ten_idx = np.argsort(i_speeds)[-10:]
#fastpar = matches.iloc[fastest]
i_top_ten_cat = i_matches.iloc[i_top_ten_idx]
#hits = gm._srcdata.source_id.isin(ref_gid)
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

ci_sx, ci_sy, ci_sra, ci_sde, ci_gra, ci_gde, ci_gid = np.array(comm_img_hits).T
cr_sx, cr_sy, cr_sra, cr_sde, cr_gra, cr_gde, cr_gid = np.array(comm_ref_hits).T

ci_seps = angle.dAngSep(ci_sra, ci_sde, ci_gra, ci_gde)
cr_seps = angle.dAngSep(cr_sra, cr_sde, cr_gra, cr_gde)

# average match distance at image and reference epochs:
sys.stderr.write("\n------------------------------------\n")
avg_ci_sep = 3600.0 * np.average(ci_seps)
avg_cr_sep = 3600.0 * np.average(cr_seps)
sys.stderr.write("average ref epoch sep: %.8f arcsec\n" % avg_cr_sep)
sys.stderr.write("average img epoch sep: %.8f arcsec\n" % avg_ci_sep)


# median match distance at image and reference epochs:
sys.stderr.write("\n------------------------------------\n")
med_ci_sep = 3600.0 * np.median(ci_seps)
med_cr_sep = 3600.0 * np.median(cr_seps)
sys.stderr.write("median  ref epoch sep: %.8f arcsec\n" % med_cr_sep)
sys.stderr.write("median  img epoch sep: %.8f arcsec\n" % med_ci_sep)

# median improvement (reduction) in match distance by switching to
# image epoch:
sys.stderr.write("\n------------------------------------\n")
med_ri_improvement = 3600.0 * np.median(cr_seps - ci_seps)
sys.stderr.write("median improvement (ref_sep - img_sep): %.6f arcsec\n"
        % med_ri_improvement)


# -----------------------------------------------------------------------
# ALTERNATE APPROACH -- dump the top 10 fastest movers to region file

# recover Gaia catalog entries (img):

# -------------
# epoch shifter:
def shifted_positions(data, tdiff_yr):
    cos_dec = np.cos(np.radians(data['dec']))


# parameter fetcher:
def get_4par(target):
    return target['ra'], target['dec'], 

# region file maker:
def make_speedy_rfile(rpath, rdata, color):
    with open(rpath, 'w') as rf:
        for i,data in rdata.iterrows():
            rf.write("fk5; annulus(%.5fd, %.5fd, 0.001d, 0.0003d) # color=%s\n"
                    % (data['ra'], data['dec'], color))

r_rpath = 'fast_movers_ref.reg'
i_rpath = 'fast_movers_img.reg'
make_speedy_rfile(r_rpath, r_top_ten_cat, 'red')
make_speedy_rfile(i_rpath, i_top_ten_cat, 'green')


sys.exit(0)

