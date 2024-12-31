#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Iterate over Gaia-matched catalogs and sift data according to Gaia ID.
#
# Rob Siverd
# Created:       2024-12-03
# Last modified: 2024-12-03
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
import resource
#import signal
import glob
import gc
import os
import sys
import time
#import pickle
#import vaex
#import calendar
#import ephem
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#import scipy.linalg as sla
#import scipy.signal as ssig
#import scipy.ndimage as ndi
#import scipy.optimize as opti
#import scipy.interpolate as stp
#import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
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
import pandas as pd
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

unlimited = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
if (resource.getrlimit(resource.RLIMIT_DATA) == unlimited):
    resource.setrlimit(resource.RLIMIT_DATA,  (3e9, 6e9))
if (resource.getrlimit(resource.RLIMIT_AS) == unlimited):
    resource.setrlimit(resource.RLIMIT_AS, (3e9, 6e9))

## Memory management:
#def get_memory():
#    with open('/proc/meminfo', 'r') as mem:
#        free_memory = 0
#        for i in mem:
#            sline = i.split()
#            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
#                free_memory += int(sline[1])
#    return free_memory
#
#def memory_limit():
#    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

## Measure memory used so far:
def check_mem_usage_MB():
    max_kb_used = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max_kb_used / 1000.0


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
#try:
#    import astropy.io.ascii as aia
#    import astropy.io.fits as pf
#    import astropy.io.votable as av
#    import astropy.table as apt
#    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
#except ImportError:
#    logger.error("astropy module not found!  Install and retry.")
#    sys.stderr.write("\nError: astropy module not found!\n")
#    sys.exit(1)

## Star extraction:
#try:
#    import easy_sep
#    reload(easy_sep)
#except ImportError:
#    logger.error("easy_sep module not found!  Install and retry.")
#    sys.stderr.write("Error: easy_sep module not found!\n\n")
#    sys.exit(1)
#pse = easy_sep.EasySEP()

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
## Save FITS image with clobber (astropy / pyfits):
#def qsave(iname, idata, header=None, padkeys=1000, **kwargs):
#    this_func = sys._getframe().f_code.co_name
#    parent_func = sys._getframe(1).f_code.co_name
#    sys.stderr.write("Writing to '%s' ... " % iname)
#    if header:
#        while (len(header) < padkeys):
#            header.append() # pad header
#    if os.path.isfile(iname):
#        os.remove(iname)
#    pf.writeto(iname, idata, header=header, **kwargs)
#    sys.stderr.write("done.\n")

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
def ldmap(things):
    return dict(zip(things, range(len(things))))

def argnear(vec, val):
    return (np.abs(vec - val)).argmin()


##--------------------------------------------------------------------------##


## Grab the ibase from a filename:
def ibase_from_filename(fits_path):
    return os.path.basename(fits_path).split('.')[0]

##--------------------------------------------------------------------------##
## Where to look for update catalogs:
load_root = 'jointupd'
#load_root = 'matched'
#load_root = 'matched_v2'
if not os.path.isdir(load_root):
    sys.stderr.write("Folder not found: %s\n" % load_root)
    sys.stderr.write("Run 21_gaia_matching.py first ...\n")
    sys.exit(1)

## Get a list of runid folders and RUNIDs themselves:
runids_list = sorted(glob.glob('%s/??????' % load_root))
runid_dirs  = {os.path.basename(x):x for x in runids_list}
runid_files = {kk:sorted(glob.glob('%s/wir*fcat'%dd)) \
                            for kk,dd in runid_dirs.items()}

## Pick one for now:
#use_runids = ['12BQ01']
#use_runids = ['12BQ01', '14AQ08']
#use_runids = ['14AQ08']
use_runids = sorted(runid_dirs.keys())

## Output folder:
save_dir = 'by_object'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

##--------------------------------------------------------------------------##
flux_cut = 1000.  # ~15th percentile
flux_cut = 10000.  # ~15th percentile

## Master list of Gaia IDs:
have_gaia_ids = set()
gaia_match_dfs = []
gaia_match_cats = []
nfcat = 0
ntodo = 0
nruns = 0

## Iterate over runids:
for this_runid in use_runids:
    nruns += 1
    have_files = runid_files[this_runid]

    for ii,this_fcat in enumerate(have_files, 1):
        nfcat += 1
        ibase = ibase_from_filename(this_fcat)
        sys.stderr.write("Loading %s ... " % this_fcat)
        ccc.load_from_fits(this_fcat)
        stars = ccc.get_catalog()
        #cats[ibase] = stars
        sys.stderr.write("done.\n")
        which_gaia = (stars['gaia_id'] > 0)
        brightish  = (stars['flux'] > flux_cut)
        matches    = stars[which_gaia & brightish]
        #srcs[ibase] = matches
        have_gaia_ids = have_gaia_ids.union(set(matches['gaia_id']))
        #gaia_match_cats.append(matches.copy())
        gaia_match_dfs.append(pd.DataFrame.from_records(matches))

    if (ntodo > 0) and (nruns >= ntodo):
        break

## Concatenate the Gaia matches:
sys.stderr.write("RAM before concat: %.3f\n" % check_mem_usage_MB())
#all_gaia_matches = pd.concat(gaia_match_dfs)
gaia_hits = pd.concat(gaia_match_dfs)
sys.stderr.write("RAM  after concat: %.3f\n" % check_mem_usage_MB())
#sys.exit(0)

## Next, iterate over Gaia IDs:
use_gaia_ids = sorted(list(have_gaia_ids))
n_matches = len(use_gaia_ids)

## Group by Gaia ID:
chunks = gaia_hits.groupby('gaia_id')

## What fraction of images should have the detection? If something is only
## visible in J, the count is already cut in half. 
filters = pd.unique(gaia_hits['filter'])
nfilter = len(filters)
detfrac = 0.5
min_pts = int(nfcat / nfilter * detfrac)

## Note the number of points for each object:
ts_sizes = {x:len(y) for x,y in chunks}
## Iterate over objects and check points per Gaia source:
#ts_sizes = {}
#tik = time.time()
#for ii,(gid,subset) in enumerate(chunks, 1):
##for ii,gid in enumerate(use_gaia_ids, 1):
#    #sys.stderr.write("Gaia match %d of %d ... " % (ii, n_matches))
#    npts = len(subset)
#    #sys.stderr.write("gid=%d, len=%d\n" % (gid, npts))
#    ts_sizes[gid] = npts
#tok = time.time()
#taken = tok - tik
#sys.stderr.write("Iterated in %.3f sec\n" % taken)

## Iterate over objects again and dump data for objects that 
## exceed the threshold:
tik = time.time()
for ii,(gid,subset) in enumerate(chunks, 1):
    #save_base = 'gm_%d.fits' % gid
    save_base = 'gm_%d.csv' % gid
    save_file = os.path.join(save_dir, save_base)
    npts = len(subset)
    if npts < min_pts:
        continue
    sys.stderr.write("gid=%d npts=%d\n" % (gid, npts))
    subset.to_csv(save_file, index=False)
    #if npts > 50:
    #    subset.to_csv(save_file, index=False)
tok = time.time()
taken = tok - tik
sys.stderr.write("Iterated in %.3f sec\n" % taken)


######################################################################
# CHANGELOG (24_sift_by_gid.py):
#---------------------------------------------------------------------
#
#  2024-12-03:
#     -- Increased __version__ to 0.0.1.
#     -- First created 24_sift_by_gid.py.
#
