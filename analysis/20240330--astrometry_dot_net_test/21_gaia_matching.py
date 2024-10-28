#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Produce a set of reference matches to Gaia using astrometry.net solutions.
#
# Rob Siverd
# Created:       2024-07-22
# Last modified: 2024-07-22
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.0.1"

## Optional matplotlib control:
#from matplotlib import use, rc, rcParams
#from matplotlib import use
#from matplotlib import rc
#from matplotlib import rcParams
#use('GTKAgg')  # use GTK with Anti-Grain Geometry engine
#use('agg')     # use Anti-Grain Geometry engine (file only)
#use('ps')      # use PostScript engine for graphics (file only)
#use('cairo')   # use Cairo (pretty, file only)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('font',**{'sans-serif':'Arial','family':'sans-serif'})
#rc('text', usetex=True) # enables text rendering with LaTeX (slow!)
#rcParams['axes.formatter.useoffset'] = False   # v. 1.4 and later
#rcParams['agg.path.chunksize'] = 10000
#rcParams['font.size'] = 10

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
#import signal
import glob
import gc
import os
import sys
import time
import pickle
import random
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
#import PIL.Image as pli
#import seaborn as sns
#import cmocean
#import theil_sen as ts
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ec = extended_catalog
except ImportError:
    sys.stderr.write("failed to import extended_catalog module!")
    sys.exit(1)

## Make objects:
ccc = ec.ExtendedCatalog()

## Angular math:
import angle
reload(angle)

## Gaia catalog matching:
import gaia_match
reload(gaia_match)
gm  = gaia_match.GaiaMatch()

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

## Pickle store routine:
def stash_as_pickle(filename, thing):
    with open(filename, 'wb') as sapf:
        pickle.dump(thing, sapf)
    return

## Pickle load routine:
def load_pickled_object(filename):
    with open(filename, 'rb') as lpof:
        thing = pickle.load(lpof)
    return thing

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

## Home-brew KDE:
#try:
#    import my_kde
#    reload(my_kde)
#    mk = my_kde
#except ImportError:
#    logger.error("module my_kde not found!  Install and retry.")
#    sys.stderr.write("\nError!  my_kde module not found!\n"
#           "Please install and try again ...\n\n")
#    sys.exit(1)

## Fast FITS I/O:
#try:
#    import fitsio
#except ImportError:
#    logger.error("fitsio module not found!  Install and retry.")
#    sys.stderr.write("\nError: fitsio module not found!\n")
#    sys.exit(1)

## Various from astropy:
try:
#    import astropy.io.ascii as aia
    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
    import astropy.time as astt
    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
#    logger.error("astropy module not found!  Install and retry.")
    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

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
            matches.append((sxx, syy, *gcoords))
            pass
        pass
    return matches



##--------------------------------------------------------------------------##
## Colors for fancy terminal output:
NRED    = '\033[0;31m'   ;  BRED    = '\033[1;31m'
NGREEN  = '\033[0;32m'   ;  BGREEN  = '\033[1;32m'
NYELLOW = '\033[0;33m'   ;  BYELLOW = '\033[1;33m'
NBLUE   = '\033[0;34m'   ;  BBLUE   = '\033[1;34m'
NMAG    = '\033[0;35m'   ;  BMAG    = '\033[1;35m'
NCYAN   = '\033[0;36m'   ;  BCYAN   = '\033[1;36m'
NWHITE  = '\033[0;37m'   ;  BWHITE  = '\033[1;37m'
ENDC    = '\033[0m'

## Suppress colors in cron jobs:
if (os.getenv('FUNCDEF') == '--nocolors'):
    NRED    = ''   ;  BRED    = ''
    NGREEN  = ''   ;  BGREEN  = ''
    NYELLOW = ''   ;  BYELLOW = ''
    NBLUE   = ''   ;  BBLUE   = ''
    NMAG    = ''   ;  BMAG    = ''
    NCYAN   = ''   ;  BCYAN   = ''
    NWHITE  = ''   ;  BWHITE  = ''
    ENDC    = ''

## Fancy text:
degree_sign = u'\N{DEGREE SIGN}'

## Dividers:
halfdiv = '-' * 40
fulldiv = '-' * 80

##--------------------------------------------------------------------------##
def ldmap(things):
    return dict(zip(things, range(len(things))))

def argnear(vec, val):
    return (np.abs(vec - val)).argmin()




##--------------------------------------------------------------------------##
##------------------           Preflight Stuff              ----------------##
##--------------------------------------------------------------------------##

## Load Gaia catalog:
gaia_csv_path = '/home/rsiverd/ucd_project/ucd_cfh_data/for_abby/gaia_calib1_NE.csv'
gm.load_sources_csv(gaia_csv_path)

## Grab the ibase from a filename:
def ibase_from_filename(fits_path):
    return os.path.basename(fits_path).split('.')[0]

## Create a by-ibase lookup dictionary from a list of paths:
def make_ibase_lookup_dict(files_list):
    ibases = [ibase_from_filename(x) for x in files_list]
    lookup = dict(zip(ibases, files_list))
    return lookup

## Load the list of fcat paths that created the solutions:
fp_file = 'fcat_paths.txt'
with open(fp_file, 'r') as fff:
    fcat_paths = [x.strip() for x in fff.readlines()]

## Make a lookup table using image base:
fcat_ibases = [ibase_from_filename(x) for x in fcat_paths]
#ibase2fcat  = dict(zip(fcat_ibases, fcat_paths))
ibase2fcat = make_ibase_lookup_dict(fcat_paths)

##--------------------------------------------------------------------------##

## Load a list of all pickled solutions:
p2_pickle_list = sorted(glob.glob('solutions/wircam_*.p2.pickle'))
p3_pickle_list = sorted(glob.glob('solutions/wircam_*.p3.pickle'))
p4_pickle_list = sorted(glob.glob('solutions/wircam_*.p4.pickle'))
p5_pickle_list = sorted(glob.glob('solutions/wircam_*.p5.pickle'))

## Make ibase lists and mappings for solutions:
#p2_ibases = [ibase_from_filename(x) for x in p2_pickle_list]
#p3_ibases = [ibase_from_filename(x) for x in p3_pickle_list]
#p4_ibases = [ibase_from_filename(x) for x in p4_pickle_list]
#p5_ibases = [ibase_from_filename(x) for x in p5_pickle_list]
ibase2pickle2 = make_ibase_lookup_dict(p2_pickle_list)
ibase2pickle3 = make_ibase_lookup_dict(p3_pickle_list)
ibase2pickle4 = make_ibase_lookup_dict(p4_pickle_list)
ibase2pickle5 = make_ibase_lookup_dict(p5_pickle_list)

pickle_lookups = {2:ibase2pickle2, 3:ibase2pickle3,
                  4:ibase2pickle4, 5:ibase2pickle5}
#pickle_lookups = [ibase2pickle2, ibase2pickle3, ibase2pickle4, ibase2pickle5]
#solve_pickles = sorted(glob.glob('solutions/wircam_*.p?.pickle'))

## Master list of all pickles for each ibase:
solutions_by_ibase = {}
for ibase in fcat_ibases:
    hits = {kk:vv[ibase] for kk,vv in pickle_lookups.items() if (ibase in vv.keys())}
    solutions_by_ibase[ibase] = hits

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Where to save updated catalogs:
save_root = 'matched'
if not os.path.isdir(save_root):
    os.mkdir(save_root)

##--------------------------------------------------------------------------##

#    matches = gm.twoway_gaia_matches(calc_ra, calc_de, match_tol)
#    idx, gra, gde, gid = matches
#    n_matches = len(idx)
#    total_sep = np.nan

##--------------------------------------------------------------------------##
##------------------            Helper Routines             ----------------##
##--------------------------------------------------------------------------##

def wircam_timestamp_from_header(header):
    obs_time = astt.Time(header['MJD-OBS'], scale='utc', format='mjd') \
            + 0.5 * astt.TimeDelta(header['EXPTIME'], format='sec')
    return obs_time


def make_aligned_gvec(ndata, idx, vals, placeholder):
    vec = np.zeros(ndata, dtype=type(placeholder)) + placeholder
    vec[idx] = vals
    return vec

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Gaia matching tolerance:
mtol_arcsec = 2.

## Randomly permute the ibases before starting (for parallel):
random.shuffle(fcat_ibases)

## Iterate over fcat bases:
total = len(fcat_ibases)
nproc = 0
ntodo = 0
results = []
for ii,ibase in enumerate(fcat_ibases, 1):
    this_fcat = ibase2fcat[ibase]
    runid     = os.path.basename(os.path.dirname(this_fcat))

    # Skip anything that isn't calib1 for now:
    if not ('calib1' in this_fcat):
        continue

    # Skip anything without a solution:
    if not solutions_by_ibase[ibase]:
        sys.stderr.write("Skipping %s (no solutions)\n" % ibase)
        continue

    # Where to put updated catalogs:
    save_dir  = os.path.join(save_root, runid)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_fcat = os.path.join(save_dir, os.path.basename(this_fcat))

    # Skip if we already have it:
    if os.path.isfile(save_fcat):
        sys.stderr.write("Skipping, catalog exists: %s\n" % save_fcat)
        continue

    # Load solutions and perform matching:
    nproc += 1
    sys.stderr.write("\n%s\n" % fulldiv)
    sys.stderr.write("Processing %s (%d of %d) ...\n" % (ibase, ii, total))
    sys.stderr.write("Loading fcat ... ")
    ccc.load_from_fits(this_fcat)
    imcat = ccc.get_catalog()
    nsrcs = len(imcat)
    imhdr = ccc.get_header()
    sys.stderr.write("done. Got %d srcs.\n" % nsrcs)

    # extract observation midpoint, propagate Gaia:
    obs_time = wircam_timestamp_from_header(imhdr)
    gm.set_epoch(obs_time)
    #gm.set_Gmag_limit(19.0)

    ra_coords = []
    de_coords = []
    for kk,pfile in solutions_by_ibase[ibase].items():
        wcs_headers = load_pickled_object(pfile)
        sln = awcs.WCS(wcs_headers)
        anra, ande = sln.all_pix2world(imcat['x'], imcat['y'], 1)
        ra_coords.append(anra)
        de_coords.append(ande)
        #test_ra, test_de = sln.all_pix2world(test_xpix, test_ypix, 1)
        # calc pixel scales:
        #xscale = 3600.0 * angle.dAngSep(test_ra[0], test_de[0], test_ra[1], test_de[1])
        #yscale = 3600.0 * angle.dAngSep(test_ra[0], test_de[0], test_ra[2], test_de[2])
        #payload = (ibase, kk, xscale, yscale)
        #results.append(payload)
        #save_line(save_file, payload)
        #sys.stderr.write("poly=%d, xscale=%.5f, yscale=%.5f\n" % (kk, xscale, yscale))
        pass
    mean_ra = np.mean(ra_coords, axis=0)
    mean_de = np.mean(de_coords, axis=0)

    matches = gm.twoway_gaia_matches(mean_ra, mean_de, mtol_arcsec)
    idx, gra, gde, gid = matches
    aligned_gra = make_aligned_gvec(nsrcs, idx, gra, np.nan)
    aligned_gde = make_aligned_gvec(nsrcs, idx, gde, np.nan)
    aligned_gid = make_aligned_gvec(nsrcs, idx, gid,     -1)


    newcat = imcat.copy()
    newcat = append_fields(newcat, ('mean_anet_ra', 'mean_anet_de'),
            (mean_ra, mean_de), usemask=False)
    newcat = append_fields(newcat, ('gaia_ra', 'gaia_de', 'gaia_id'),
            (aligned_gra, aligned_gde, aligned_gid), usemask=False)

    # update catalog and save:
    ccc.set_catalog(newcat)
    ccc.save_as_fits(save_fcat, overwrite=True)

    if (ntodo > 0) and (nproc >= ntodo):
        break



##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

sys.exit(0)


######################################################################
# CHANGELOG (21_gaia_matching.py):
#---------------------------------------------------------------------
#
#  2024-07-22:
#     -- Increased __version__ to 0.0.1.
#     -- First created 21_gaia_matching.py.
#
