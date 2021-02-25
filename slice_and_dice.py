#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Standalone script to test coordinate fitting and analysis methods.
#
# Rob Siverd
# Created:       2019-10-16
# Last modified: 2020-02-09
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Logging setup:
import logging
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

## Current version:
__version__ = "0.3.5"

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
import argparse
#import resource
#import signal
import pickle
#import glob
import gc
import os, errno
import sys
import time
import numpy as np
from numpy.lib.recfunctions import append_fields
#import scipy.linalg as sla
#import scipy.signal as ssig
#import scipy.ndimage as ndi
#import scipy.optimize as opti
#import scipy.interpolate as stp
#import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib.ticker as mt
#import matplotlib._pylab_helpers as hlp
#from matplotlib.colors import LogNorm
#from matplotlib import colors
import matplotlib.colors as mplcolors
import matplotlib.collections as mcoll
#import matplotlib.gridspec as gridspec
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
import theil_sen as ts
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

import angle
reload(angle)

import fluxmag
reload(fluxmag)

import astrom_test
reload(astrom_test)
af = astrom_test.AstFit()
eee = astrom_test.SSTEph()

## Easy Gaia source matching:
try:
    import gaia_match
    reload(gaia_match)
    gm = gaia_match.GaiaMatch()
except ImportError:
    logger.error("failed to import gaia_match module!")
    sys.exit(1)

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ec = extended_catalog
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

## Handy MarkerUpdater class:
try:
    import marker_updater
    reload(marker_updater)
    mu = marker_updater
except ImportError:
    logger.error("failed to import marker_updater module!")
    sys.exit(1)

## Because obviously:
#import warnings
#if not sys.warnoptions:
#    warnings.simplefilter("ignore", category=DeprecationWarning)
#    warnings.simplefilter("ignore", category=UserWarning)
#    warnings.simplefilter("ignore")
#with warnings.catch_warnings():
#    some_risky_activity()
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
#    import problem_child1, problem_child2

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
## Recursive directory creation:
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

##--------------------------------------------------------------------------##

## Home-brew robust statistics:
try:
    import robust_stats
    reload(robust_stats)
    rs = robust_stats
except ImportError:
    logger.error("module robust_stats not found!  Install and retry.")
    sys.stderr.write("\nError!  robust_stats module not found!\n"
           "Please install and try again ...\n\n")
    sys.exit(1)

## Various from astropy:
try:
#    import astropy.io.ascii as aia
#    import astropy.io.fits as pf
    import astropy.table as apt
    import astropy.time as astt
#    import astropy.wcs as awcs
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
#    logger.error("astropy module not found!  Install and retry.")
    sys.stderr.write("\nError: astropy module not found!\n")
    sys.exit(1)

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
## Catch interruption cleanly:
#def signal_handler(signum, frame):
#    sys.stderr.write("\nInterrupted!\n\n")
#    sys.exit(1)
#
#signal.signal(signal.SIGINT, signal_handler)

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
##------------------         Parse Command Line             ----------------##
##--------------------------------------------------------------------------##

## Parse arguments and run script:
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

## Enable raw text AND display of defaults:
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                        argparse.RawDescriptionHelpFormatter):
    pass

## Parse the command line:
if __name__ == '__main__':

    # ------------------------------------------------------------------
    prog_name = os.path.basename(__file__)
    descr_txt = """
    Position/proper motion analysis method testbed.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(thing1='value1', thing2='value2')
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    iogroup.add_argument('-C', '--cat_list', default=None, required=True,
            help='ASCII file with list of catalog paths in column 1')
    iogroup.add_argument('-g', '--gaia_csv', default=None, required=False,
            help='CSV file with Gaia source list', type=str)
    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
    #        help='Output filename', type=str)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Miscellany:
    miscgroup = parser.add_argument_group('Miscellany')
    miscgroup.add_argument('--debug', dest='debug', default=False,
            help='Enable extra debugging messages', action='store_true')
    miscgroup.add_argument('-q', '--quiet', action='count', default=0,
            help='less progress/status reporting')
    miscgroup.add_argument('-v', '--verbose', action='count', default=0,
            help='more progress/status reporting')
    # ------------------------------------------------------------------

    context = parser.parse_args()
    context.vlevel = 99 if context.debug else (context.verbose-context.quiet)
    context.prog_name = prog_name

## Long live ipython!
gc.collect()

##--------------------------------------------------------------------------##
##------------------       load Gaia sources from CSV       ----------------##
##--------------------------------------------------------------------------##

if context.gaia_csv:
    try:
        logger.info("Loading sources from %s" % context.gaia_csv)
        gm.load_sources_csv(context.gaia_csv)
    except:
        logger.error("Yikes ...")
        sys.exit(1)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## RA/DE coordinate keys for various methods:
centroid_colmap = {
        'simple'    :   ('dra', 'dde'),
        'window'    :   ('wdra', 'wdde'),
        'pp_fix'    :   ('ppdra', 'ppdde'),
        }


##--------------------------------------------------------------------------##
## Example and data-saving config:
exdir = 'examples'
csv_name = os.path.basename(context.gaia_csv)
targname = csv_name.split('.')[0].split('_')[-1]
targ_dir = os.path.join(exdir, targname)

##--------------------------------------------------------------------------##
## Read ASCII file to list:
def read_column(filename, column=0, delim=' ', strip=True):
    with open(filename, 'r') as f:
        content = f.readlines()
    content = [x.split(delim)[column] for x in content]
    if strip:
        content = [x.strip() for x in content] 
    return content

def irac_channel_from_filename(filename):
    chtag = os.path.basename(filename).split('_')[1]
    return int(chtag[1])



##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##






##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Load list of catalogs:
cat_files = read_column(context.cat_list)

## Load those catalogs:
tik = time.time()
cdata_all = []
total = len(cat_files)
for ii,fname in enumerate(cat_files, 1):
    sys.stderr.write("\rLoading catalog %d of %d ... " % (ii, total))
    ccc = ec.ExtendedCatalog()
    ccc.load_from_fits(fname)
    cdata_all.append(ccc)
tok = time.time()
sys.stderr.write("done. Took %.3f seconds.\n" % (tok-tik))

cdata = [x for x in cdata_all]  # everything
#cdata = [x for x in cdata_all if (x.get_header()['AP_ORDER'] > 3)]

#summary = []
#for ccc in cdata:
#    imname = ccc.get_imname()
#    irchan = irac_channel_from_filename(imname)
#    imbase = os.path.basename(imname)
#    tmphdr = ccc.get_header()
#    expsec = tmphdr['EXPTIME']
#    #expsec = ccc.get_header()['EXPTIME']
#    #obdate = ccc.get_header()
#    porder = tmphdr['AP_ORDER']
#    nfound = len(ccc.get_catalog())
#    summary.append((imname, irchan, expsec, nfound, porder))

#cbcd, irac, expt, nsrc = zip(*summary)

## Useful summary data:
cbcd_name = [x.get_imname() for x in cdata]
irac_band = np.array([irac_channel_from_filename(x) for x in cbcd_name])
expo_time = np.array([x.get_header()['EXPTIME'] for x in cdata])
n_sources = np.array([len(x.get_catalog()) for x in cdata])
sip_order = np.array([x.get_header()['AP_ORDER'] for x in cdata])
timestamp = astt.Time([x.get_header()['DATE_OBS'] for x in cdata],
        format='isot', scale='utc')
jdutc = timestamp.jd
#jdutc = ['%.6f'%x for x in timestamp.jd]
jd2im = {kk:vv for kk,vv in zip(jdutc, cbcd_name)}
im2jd = {kk:vv for kk,vv in zip(cbcd_name, jdutc)}
im2ex = {kk:vv for kk,vv in zip(cbcd_name, expo_time)}

##--------------------------------------------------------------------------##
## Concatenated list of RA/Dec coordinates:

centroid_method = 'simple'
#centroid_method = 'window'
#centroid_method = 'pp_fix'
_ra_key, _de_key = centroid_colmap[centroid_method]
every_dra = np.concatenate([x._imcat[_ra_key] for x in cdata])
every_dde = np.concatenate([x._imcat[_de_key] for x in cdata])
every_jdutc = np.concatenate([n*[jd] for n,jd in zip(n_sources, jdutc)])   
#every_jdutc = np.float_(every_jdutc)
gc.collect()


##--------------------------------------------------------------------------##
##-----------------   Cross-Match to Gaia, Extract Target  -----------------##
##--------------------------------------------------------------------------##

ntodo = 100
toler_sec = 3.0
gcounter = {x:0 for x in gm._srcdata.source_id}
n_gaia = len(gm._srcdata)

## Iterate over individual image tables (prevents double-counting):
#for ci,ccat in enumerate(cdata, 1):
#    sys.stderr.write("\n------------------------------\n")
#    sys.stderr.write("Checking image %d of %d ...\n" % (ci, len(cdata)))
#    for gi,(gix, gsrc) in enumerate(gm._srcdata.iterrows(), 1):
#        sys.stderr.write("Checking Gaia source %d of %d ...\n" % (gi, n_gaia))
#        pass
#    pass
#    if (ntodo > 0) and (ii >= ntodo):
#        break

## First, check which Gaia sources might get used:
tik = time.time()
for ii,(index, gsrc) in enumerate(gm._srcdata.iterrows(), 1):
    sys.stderr.write("\rChecking Gaia source %d of %d ... " % (ii, n_gaia))
    sep_sec = 3600. * angle.dAngSep(gsrc.ra, gsrc.dec, every_dra, every_dde)
    gcounter[gsrc.source_id] += np.sum(sep_sec <= toler_sec)
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))
gc.collect()

## Make Gaia subset of useful objects:
need_srcs = 3
useful_ids = [kk for kk,vv in gcounter.items() if vv>need_srcs]
use_gaia = gm._srcdata[gm._srcdata.source_id.isin(useful_ids)]
n_useful = len(use_gaia)
sys.stderr.write("Found possible matches to %d of %d Gaia sources.\n"
        % (n_useful, len(gm._srcdata)))
gc.collect()
if n_useful < 5:
    sys.stderr.write("Gaia match error: found %d useful objects\n" % n_useful)
    sys.exit(1)

## Total Gaia-detected PM in surviving object set:
use_gaia = use_gaia.assign(pmtot=np.hypot(use_gaia.pmra, use_gaia.pmdec))
gaia_pmsrt = use_gaia.sort_values(by='pmtot', ascending=False)

#for nmin in range(100):
#    passing = [kk for kk,vv in gcounter.items() if vv>nmin]
#    nkept = len(passing)
#    sys.stderr.write("Kept %d sources for nmin=%d.\n" % (nkept, nmin))

## Robust (non-double-counted) matching of Gaia sources using slimmed list:
sys.stderr.write("Associating catalog objects with Gaia sources:\n") 
tik = time.time()
gmatches = {x:[] for x in use_gaia.source_id}
for ci,extcat in enumerate(cdata, 1):
#for ci,extcat in enumerate(cdata[:10], 1):
    #sys.stderr.write("\n------------------------------\n")
    sys.stderr.write("\rChecking image %d of %d ... " % (ci, len(cdata)))
    #ccat = extcat._imcat
    ccat = extcat.get_catalog()
    #cat_jd = jdutc[ci]
    jd_info = {'jd':jdutc[ci-1], 'iname':extcat.get_imname()}
    for gi,(gix, gsrc) in enumerate(use_gaia.iterrows(), 1):
        #sys.stderr.write("Checking Gaia source %d of %d ... " % (gi, n_useful))
        sep_sec = 3600.0 * angle.dAngSep(gsrc.ra, gsrc.dec, 
                                    ccat[_ra_key], ccat[_de_key])
                                    #ccat['dra'], ccat['dde'])
        matches = sep_sec <= toler_sec
        nhits = np.sum(matches)
        if (nhits == 0):
            #sys.stderr.write("no match!\n")
            continue
        else:
            #sys.stderr.write("got %d match(es).  " % nhits)
            hit_sep = sep_sec[matches]
            hit_cat = ccat[matches]
            sepcheck = 3600.0 * angle.dAngSep(gsrc.ra, gsrc.dec, 
                    hit_cat[_ra_key], hit_cat[_de_key])
                    #hit_cat['dra'], hit_cat['dde'])
            #sys.stderr.write("sepcheck: %.4f\n" % sepcheck)
            nearest = hit_sep.argmin()
            m_info = {}
            m_info.update(jd_info)
            m_info['sep'] = hit_sep[nearest]
            m_info['cat'] = hit_cat[nearest]
            #import pdb; pdb.set_trace()
            #sys.exit(1)
            gmatches[gsrc.source_id].append(m_info)
    pass
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))
gc.collect()

## Stop here if no Gaia matches:
if not gmatches:
    sys.stderr.write("No matches to Gaia detected! Something is wrong ...\n")
    sys.exit(1)

#first = [x for x in gmatches.keys()][44]
##derp = np.vstack([x['cat'].array for x in gmatches[first]]) # for FITSRecord
#derp = np.vstack([x['cat'] for x in gmatches[first]])
##derp = [apt.Table(x['cat']) for x in gmatches[first]]
#jtmp = np.array([x['jd'] for x in gmatches[first]])
#derp = append_fields(derp, 'jdutc', jtmp, usemask=False)

##--------------------------------------------------------------------------##
##-----------------        Box Extract Target Data         -----------------##
##--------------------------------------------------------------------------##

## Target extraction region:
tbox_de = (-9.585, -9.581)
#tbox_ra = (63.837, 63.844)
tbox_ra = (63.825, 63.844)
trent_ra, trent_pmra = 63.83469, 2.204 
trent_de, trent_pmde = -9.58437, 0.541
trent_epoch_mjd = 53024.06
trent_epoch_jd = trent_epoch_mjd + 2400000.5
trent_pars = [trent_ra, trent_de, 
        trent_pmra/np.cos(np.radians(trent_de)), trent_pmde]
j2000_epoch = astt.Time('2000-01-01T12:00:00', scale='tt', format='isot')
def targpos(dt_years, params):
    _asec_per_deg = 3600.
    tra = params[0] + (params[2] / _asec_per_deg * dt_years)
    tde = params[1] + (params[3] / _asec_per_deg * dt_years)
    return tra, tde

## Find target data points:
sys.stderr.write("Extracting %s data ... " % targname) 
tik = time.time()
tgt_data = []
tgt_tol_asec = 3.
for ci,extcat in enumerate(cdata, 1):
    ccat = extcat.get_catalog()
    jd_info = {'jd':jdutc[ci-1], 'iname':extcat.get_imname()}
    #elapsed_yr = (jd_info['jd'] - j2000_epoch.utc.jd) / 365.25
    elapsed_yr = (jd_info['jd'] - trent_epoch_jd) / 365.25
    _ra, _de = targpos(elapsed_yr, trent_pars)
    sep_sec = 3600. * angle.dAngSep(_ra, _de, ccat[_ra_key], ccat[_de_key])
    matches = sep_sec <= tgt_tol_asec
    nhits = np.sum(matches)
    #sys.stderr.write("nhits: %d\n" % nhits)


    # box selection:
    which = (tbox_ra[0] <= ccat[_ra_key]) & (ccat[_ra_key] <= tbox_ra[1]) \
            & (tbox_de[0] <= ccat[_de_key]) & (ccat[_de_key] <= tbox_de[1])
    #for match in ccat[which]:
    for match in ccat[matches]:
        m_info = {}
        m_info.update(jd_info)
        m_info['cat'] = match
        tgt_data.append(m_info)
    #nhits = np.sum(which)
    #if (nhits > 1):
    #    sys.stderr.write("Found multiple in-box sources in catalog %d!\n" % ci)
    #    sys.exit(1)
    #if np.any(which):
    #    m_info = {}
    #    m_info.update(jd_info)
    #    m_info['cat'] = ccat[which]
    #    tgt_data.append(m_info)
    pass
tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))
gc.collect()

## Stop here in case of matching failure(s):
if not tgt_data:
    sys.stderr.write("No matches for target data??  Please investigate ...\n")
    sys.exit(1)

##--------------------------------------------------------------------------##
## How to repackage matched data points:
def repack_matches(match_infos):
    ccat = np.vstack([x['cat'] for x in match_infos])
    jtmp = np.array([x['jd'] for x in match_infos])
    itmp = np.array([x['iname'] for x in match_infos])
    return append_fields(ccat, ('jdutc', 'iname'), (jtmp, itmp), usemask=False)

##--------------------------------------------------------------------------##
## Collect and export target data set for analysis:
tgt_ccat = repack_matches(tgt_data)

#sys.exit(0)
## Collect data sets by Gaia source for analysis:
gtargets = {}
for ii,gid in enumerate(gmatches.keys(), 1):
    sys.stderr.write("\rGathering gaia source %d of %d ..." % (ii, n_useful))
    #derp = np.vstack([x['cat'] for x in gmatches[gid]])
    #jtmp = np.array([x['jd'] for x in gmatches[gid]])
    #itmp = np.array([x['iname'] for x in gmatches[gid]])
    #derp = append_fields(derp, ('jdutc', 'iname'), (jtmp, itmp), usemask=False)
    #gtargets[gid] = derp
    gtargets[gid] = repack_matches(gmatches[gid])
sys.stderr.write("done.\n")
#sys.stderr.write("At this point, RAM use (MB): %.2f\n" % check_mem_usage_MB())

gtg_npts = {gg:len(cc) for gg,cc in gtargets.items()} 
npts_100 = [gg for gg,nn in gtg_npts.items() if nn>100]

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
## Convenient, percentile-based plot limits:
def gaia_limits(vec, pctiles=[1,99], pad=1.2):
    ends = np.percentile(vec[~np.isnan(vec)], pctiles)
    middle = np.average(ends)
    return (middle + pad * (ends - middle))

##--------------------------------------------------------------------------##
## Plot a single set of ra/dec vs jdutc:
def get_targ_by_n(gtargets, nn):
    which = list(gtargets.keys())[nn]
    return gtargets[which]

def justplot(nn):
    gdata = get_targ_by_n(gtargets, nn)
    plt.clf()
    plt.scatter(gdata[_ra_key], gdata[_de_key], lw=0, s=5, c=gdata['jdutc'])
    return

def plotsingle(ax, gdata):
    ax.scatter(gdata[_ra_key], gdata[_de_key], lw=0, s=5, c=gdata['jdutc'])

def plotgsource(srcid):
    gdata = gtargets[srcid]
    plt.clf()
    plt.scatter(gdata[_ra_key], gdata[_de_key], lw=0, s=35, c=gdata['jdutc'])
    ax = plt.gca()
    ax.set_xlim(gaia_limits(gdata[_ra_key]))
    ax.set_ylim(gaia_limits(gdata[_de_key]))
    result = ax.invert_xaxis() if not ax.xaxis_inverted() else None
    return

def gather_by_id(gaia_srcid):
    neato_gaia = use_gaia[use_gaia.source_id == gaia_srcid]
    neato_sptz = gtargets[gaia_srcid]
    return neato_gaia, neato_sptz

lookie = [
    3192149715934444800,
    3192153147611258880,
    3192147963587789184,
    3192149715934444544,
    3192152842669673088,
    3192149647215037440,
    3192149780357313920,
    3192152911390108416,
    #3192149917795842688,
    3192150334409729536,
    ]

#lookie = list(set(lookie + npts_100))

lookie = npts_100

##--------------------------------------------------------------------------##
## Kludgey Spitzer ephemeris:
#use_epoch_tdb = 2456712.3421157757
use_epoch_tdb = 2457174.500000000
use_epoch = astt.Time(2457174.50000000, format='jd', scale='tdb')
use_epoch_tdb = use_epoch.tdb.jd
sst_eph_file = 'ephemerides/spitz_ssb_data.csv'
eee.load(sst_eph_file)
gse_tuple_savefile = 'GSE_tuple.pickle'

def get_fit_residuals(sneat, use_eph, sigcut):
    tmpres = {}
    sra, sde = sneat[_ra_key], sneat[_de_key]
    af.setup(use_epoch_tdb, sra, sde, use_eph)
    bestpars = af.fit_bestpars(sigcut=sigcut)
    best_ra, best_de = af.eval_model(bestpars)
    resid_ra, resid_de = af._calc_radec_residuals(bestpars)
    tmpres['resid_ra'], tmpres['resid_de'] = resid_ra, resid_de
    tmpres['jdtdb'] = use_eph['jdtdb'].copy()
    tmpres['flux'] = sneat['flux'].copy()
    return tmpres

## Check several:
_DO_EXPORT = True
if _DO_EXPORT:
    gse_data = {}
    res_data = {}
    sigcut = 5.0

    # single-object version of target data:
    tsrc_dir = os.path.join(targ_dir, 'target')
    mkdir_p(tsrc_dir)
    _tsave = os.path.join(tsrc_dir, gse_tuple_savefile)
    tgt_eph = eee.retrieve(tgt_ccat['iname'])
    with open(_tsave, 'wb') as ff:
        pickle.dump((None, tgt_ccat, tgt_eph), ff)
    res_data['tgt'] = get_fit_residuals(tgt_ccat, tgt_eph, sigcut)

    # fit residuals for field objects:
    for gid in lookie:
        sys.stderr.write("Examining %d ... \n" % gid) 
        gneat, sneat = gather_by_id(gid)
        use_eph = eee.retrieve(sneat['iname'])
        gse_data[gid] = (gneat, sneat, use_eph)
    
        #tmpres = {}
        #sra, sde = sneat[_ra_key], sneat[_de_key]
        #af.setup(use_epoch_tdb, sra, sde, use_eph)
        #bestpars = af.fit_bestpars(sigcut=sigcut)
        #best_ra, best_de = af.eval_model(bestpars)
        #resid_ra, resid_de = af._calc_radec_residuals(bestpars)
        #tmpres['resid_ra'], tmpres['resid_de'] = resid_ra, resid_de
        #tmpres['jdtdb'] = use_eph['jdtdb'].copy()
        #res_data[gid] = tmpres
        res_data[gid] = get_fit_residuals(sneat, use_eph, sigcut)

        # single-object save files:
        gsrc_dir = os.path.join(targ_dir, 'gaia_%d' % gid)
        mkdir_p(gsrc_dir)
        _gsave = os.path.join(gsrc_dir, gse_tuple_savefile)
        with open(_gsave, 'wb') as ff:
            pickle.dump((gneat, sneat, use_eph), ff)

    # Save GSE and residuals for external use:
    export_dir  = os.path.join(targ_dir, 'combo')
    mkdir_p(export_dir)
    export_file = os.path.join(export_dir, 'resid_and_gse.pickle')
    with open(export_file, 'wb') as ef:
        pickle.dump((res_data, gse_data, centroid_method), ef)
    target_file = os.path.join(export_dir, 'target_data.pickle')
    with open(target_file, 'wb') as tf:
        pickle.dump(tgt_ccat, tf)

##--------------------------------------------------------------------------##

#sys.exit(0)
##--------------------------------------------------------------------------##
## GOT ONE:
sys.stderr.write("%s\n" % fulldiv)
#gneat, sneat = gather_by_id(3192149715934444800)
use_gid = lookie[0]
gneat, sneat = gather_by_id(use_gid)

# ----------------
sys.stderr.write("Gaia info:\n\n") 
sys.stderr.write("RA:       %15.7f\n" % gneat['ra'])
sys.stderr.write("DE:       %15.7f\n" % gneat['dec'])
sys.stderr.write("parallax: %10.4f +/- %8.4f\n"
        % (gneat.parallax, gneat.parallax_error))
sys.stderr.write("pmRA:     %10.4f +/- %8.4f\n"
        % (gneat.pmra, gneat.pmra_error))
sys.stderr.write("pmDE:     %10.4f +/- %8.4f\n"
        % (gneat.pmdec, gneat.pmdec_error))

## Kludgey Spitzer ephemeris:
#use_epoch_tdb = 2456712.3421157757
sst_eph_file = 'ephemerides/spitz_ssb_data.csv'
eee.load(sst_eph_file)
use_eph = eee.retrieve(sneat['iname'])
#sjd_tdb = use_eph['jdtdb']

## Optionally save data for external plotting:
save_example = True
if save_example:
    gsrc_dir = os.path.join(exdir, targname, 'gaia_%d' % use_gid)
    mkdir_p(gsrc_dir)
    _gsave = os.path.join(gsrc_dir, 'GSE_tuple.pickle')
    with open(_gsave, 'wb') as ff:
        pickle.dump((gneat, sneat, use_eph), ff)
    pass

sjd_utc, sra, sde = sneat['jdutc'], sneat[_ra_key], sneat[_de_key]
syr = 2000.0 + ((sjd_utc - 2451544.5) / 365.25)
smonth = (syr % 1.0) * 12.0

ts_ra_model = ts.linefit(syr, sra)
ts_de_model = ts.linefit(syr, sde)
ts_pmra_masyr = ts_ra_model[1] * 3.6e6 / np.cos(np.radians(ts_de_model[0]))
ts_pmde_masyr = ts_de_model[1] * 3.6e6

# initial RA/Dec guess:
sys.stderr.write("%s\n" % fulldiv)
guess_ra = sra.mean()
guess_de = sde.mean()
sys.stderr.write("guess_ra:  %15.7f\n" % guess_ra)
sys.stderr.write("guess_de:  %15.7f\n" % guess_de)

afpars = [guess_ra, guess_de, ts_pmra_masyr/1e3, ts_pmde_masyr/1e3, 1.0]
appcoo = af.apparent_radec(use_epoch_tdb, afpars, use_eph)

# proper fit:
design_matrix = np.column_stack((np.ones(syr.size), syr))
#de_design_matrix = np.column_stack((np.ones(syr.size), syr))
ra_ols_res = sm.OLS(sra, design_matrix).fit()
de_ols_res = sm.OLS(sde, design_matrix).fit()
ra_rlm_res = sm.RLM(sra, design_matrix).fit()
de_rlm_res = sm.RLM(sde, design_matrix).fit()
rlm_pmde_masyr = de_rlm_res.params[1] * 3.6e6
rlm_pmra_masyr = ra_rlm_res.params[1] * 3.6e6 \
        * np.cos(np.radians(de_rlm_res.params[0]))


sys.stderr.write("\nTheil-Sen intercepts:\n")
sys.stderr.write("RA:   %15.7f\n" % ts_ra_model[0]) 
sys.stderr.write("DE:   %15.7f\n" % ts_de_model[0]) 

sys.stderr.write("\nTheil-Sen proper motions:\n")
sys.stderr.write("RA:   %10.6f mas/yr\n" % ts_pmra_masyr)
sys.stderr.write("DE:   %10.6f mas/yr\n" % ts_pmde_masyr)

sys.stderr.write("\nRLM (Huber) proper motions:\n")
sys.stderr.write("RA:   %10.6f mas/yr\n" % rlm_pmra_masyr)
sys.stderr.write("DE:   %10.6f mas/yr\n" % rlm_pmde_masyr)

sys.stderr.write("\n")
sys.stderr.write("%s\n" % fulldiv)

bfde_path = de_rlm_res.params[0] + de_rlm_res.params[1]*syr
bfra_path = ra_rlm_res.params[0] + ra_rlm_res.params[1]*syr

# TO INSPECT:
# plotgsource(3192149715934444800); plt.plot(bfra_path, bfde_path)
sys.stderr.write("NOTE TO SELF: Gaia pmRA includes cos(dec)!\n")


#spts = plt.scatter(syr, sra, c=smonth)
#cbar = fig.colorbar(spts)

##--------------------------------------------------------------------------##
##------------------         WCS Quality Evaluation         ----------------##
##--------------------------------------------------------------------------##

matched_gaia_ids = [x for x in gmatches.keys()]
#eg_gid = 3192153353769693568
#something = gtargets[eg_gid]

def calc_objseps(gid):
    _gaia, _spit = gather_by_id(eg_gid)
    _gra, _gde = _gaia.ra.data, _gaia.dec.data
    _gcoords = _gaia.ra.data, _gaia.dec.data
    raw_arcsep = 3.6e3 * angle.dAngSep(*_gcoords, _spit[ 'dra'], _spit[ 'dde'])
    win_arcsep = 3.6e3 * angle.dAngSep(*_gcoords, _spit['wdra'], _spit['wdde'])
    return {'raw':raw_arcsep, 'win':win_arcsep}

#sep_by_jd = {x:[] for x in jdutc}
sep_by_im = {x:[] for x in cbcd_name}
#sep_by_id = {x:[] for x in matched_gaia_ids}
sep_by_id = {}
#for gid in [eg_gid]:
for gid in matched_gaia_ids:
    _gaia, _spit = gather_by_id(gid)
    _gra, _gde = _gaia.ra.data, _gaia.dec.data
    _gcoords = _gaia.ra.data, _gaia.dec.data
    raw_arcsep = 3.6e3 * angle.dAngSep(*_gcoords, _spit[ 'dra'], _spit[ 'dde'])
    win_arcsep = 3.6e3 * angle.dAngSep(*_gcoords, _spit['wdra'], _spit['wdde'])
    #for jj,rsep,wsep in zip(_spit['jdutc'], raw_arcsep, win_arcsep):
    #    sep_by_jd[jj].append((gid, rsep, wsep))
    for im,rsep,wsep in zip(_spit['iname'], raw_arcsep, win_arcsep):
        sep_by_im[im].append((gid, rsep, wsep))


    obs_coords = angle.spheremean_deg(_spit['dra'], _spit['dde'])
    obs_arcsep = 3.6e3 * angle.dAngSep(*obs_coords, _spit['dra'], _spit['dde'])
    favg = np.average(_spit['flux'])
    fstd = np.std(_spit['flux'])
    savg = np.average(obs_arcsep)
    #sys.stderr.write("raw_arcsep (%.3f, %.3f): %s\n" 
    #        % (favg, fstd, str(raw_arcsep)))
    #sys.stderr.write("flx: %.3f, sep: %.4f\n" % (favg, savg))
    sep_by_id[gid] = (favg, savg)

#sep_by_jd = {kk:vv for kk,vv in sep_by_jd.items() if len(vv)>1} # drop empty
sep_by_im = {kk:vv for kk,vv in sep_by_im.items() if len(vv)>1} # drop empty

## Analyze scatter:
#coo_errs = {}
coo_errs = []
for ii,(kk, vv) in enumerate(sep_by_im.items()):
    combo = np.array(vv)
    _, avg_raw_err, avg_win_err = np.average(combo, axis=0)
    _, med_raw_err, med_win_err = np.median(combo, axis=0)
    coo_errs.append((ii, kk, im2jd[kk],
        avg_raw_err, med_raw_err, avg_win_err, med_win_err))
#coo_errs = np.array(coo_errs)
coo_test = np.core.records.fromarrays(zip(*coo_errs),
        names='idx,iname,jdutc,ravg,rmed,wavg,wmed')
#[jd2im[x] for x in sep_by_jd.keys()]

plt.clf()
for col in ['ravg','rmed','wmed']: #'wavg'
    plt.scatter(coo_test['idx'], coo_test[col], lw=0, s=40, label=col)
plt.legend(loc='best')

## scatter vs. mag:
rmsflx, rmssep = zip(*sep_by_id.values())
rmsmag = fluxmag.kmag(rmsflx)

#plt.clf()
#plt.scatter(rmsmag, rmssep)
#plt.axhline(obs_floor, c='r', ls='--', label='observed floor (~130 mas)')
#plt.axhline(0.005, c='g', ls='--', label='expected floor (~5 mas)')
#plt.ylim(0.003, 5.0)
#plt.yscale('log')
#plt.grid(True)
#plt.xlabel('Instrumental Mag (approx)')
#plt.ylabel('Abs. Scatter (arcsec)')
#plt.legend(loc='upper left')
#plt.tight_layout()
#plt.savefig('noise_floors.png')



#sys.exit(0)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
## Misc:
#def log_10_product(x, pos):
#   """The two args are the value and tick position.
#   Label ticks with the product of the exponentiation."""
#   return '%.2f' % (x)  # floating-point
#
#formatter = plt.FuncFormatter(log_10_product) # wrap function for use

## Convenient, percentile-based plot limits:
def nice_limits(vec, pctiles=[1,99], pad=1.2):
    ends = np.percentile(vec[~np.isnan(vec)], pctiles)
    middle = np.average(ends)
    return (middle + pad * (ends - middle))

## Convenient plot limits for datetime/astropy.Time content:
#def nice_time_limits(tvec, buffer=0.05):
#    lower = tvec.min()
#    upper = tvec.max()
#    ndays = upper - lower
#    return ((lower - 0.05*ndays).datetime, (upper + 0.05*ndays).datetime)

## Convenient limits for datetime objects:
#def dt_limits(vec, pad=0.1):
#    tstart, tstop = vec.min(), vec.max()
#    trange = (tstop - tstart).total_seconds()
#    tpad = dt.timedelta(seconds=pad*trange)
#    return (tstart - tpad, tstop + tpad)

##--------------------------------------------------------------------------##
## KDE:
#kde_pnts, kde_vals = mk.go(data_vec)

##--------------------------------------------------------------------------##
## Plot config:

# gridspec examples:
# https://matplotlib.org/users/gridspec.html

#gs1 = gridspec.GridSpec(4, 4)
#gs1.update(wspace=0.025, hspace=0.05)  # set axis spacing

#ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3) # top-left + center + right
#ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2) # mid-left + mid-center
#ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) # mid-right + bot-right
#ax4 = plt.subplot2grid((3, 3), (2, 0))            # bot-left
#ax5 = plt.subplot2grid((3, 3), (2, 1))            # bot-center

#sys.exit(0)

##--------------------------------------------------------------------------##
## Scatterplot using patches (effectively allows point size in data units):
## from https://stackoverflow.com/questions/5009316/plot-scatter-position-and-marker-size-in-the-same-coordinates
def sizes_match(*args):
    sizes = set([len(x) for x in args])
    return (len(sizes) == 1)


def circle_scatter(ax, xvals, yvals, radii, cvals, **kwargs):
    if not sizes_match(xvals, yvals, radii, cvals):
        sys.stderr.write("\nArray size mismatch:\n")
        sys.stderr.write("len(xvals): %10d\n" % len(xvals))
        sys.stderr.write("len(yvals): %10d\n" % len(yvals))
        sys.stderr.write("len(radii): %10d\n" % len(radii))
        sys.stderr.write("len(cvals): %10d\n" % len(cvals))
        return False

    cmap = kwargs.get('cmap', plt.cm.viridis)
    norm = kwargs.get('norm', mplcolors.Normalize())
    circles = [plt.Circle((cx, cy), radius=cr, color=cc) \
            for cx,cy,cr,cc in zip(xvals, yvals, radii, cmap(norm(cvals)))]
    ax.add_collection(mcoll.PatchCollection(circles, match_original=True))
    #ax.add_collection(c)
    #globals()['lolwut'] = circles
    #for cx,cy,cr,cc in zip(xvals, yvals, radii, cmap(norm(cvals))):
    #    circle = plt.Circle((cx, cy), radius=cr, color=cc)
    #    ax.add_patch(circle)
    return True


##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (12, 10)
fig = plt.figure(1, figsize=fig_dims)
plt.gcf().clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
ax1 = fig.add_subplot(111, aspect='equal')
ax1.grid(True)
ax1.xaxis.get_major_formatter().set_useOffset(False)
ax1.yaxis.get_major_formatter().set_useOffset(False)
#ax1.scatter(expt, nsrc, c=irac)

#sys.exit(0)
mupdate = mu.MarkerUpdater()

#cmap = plt.cm.rainbow
#cmap = plt.cm.viridis

#pkw = {'cmap':mplcolors.rainbow, 'norm':mplcolors.Normalize(vmin=1.5, vmax=4.5)

lolwut = 'adsf'
__fancy__ = False
#__fancy__ = True
descrtxt = 'fancy' if __fancy__ else 'fast'
sys.stderr.write("Plotting detections (%s) ... " % descrtxt)
tik = time.time()
if __fancy__:
    # Fancy CIRCLE scatter:
    pradii = 0.0 * every_dra + 0.00005
    win = circle_scatter(ax1, xvals=every_dra, yvals=every_dde, 
            radii=pradii, cvals=every_jdutc)
else:
    #ax1.plot(every_dra, every_dde, lw=0, ms=3, color=every_jdutc)
    pkw = {'s':2.5, 'alpha':0.2}
    pkw = {'s':5.0, 'alpha':0.4}
    spts = ax1.scatter(every_dra, every_dde, lw=0, c=every_jdutc, **pkw)
    mupdate.add_ax(ax1, ['size'])   # auto-update marker size

tok = time.time()
sys.stderr.write("done. (%.3f s)\n" % (tok-tik))

## Adjust limits and orientation:
ax1.set_xlim(nice_limits(every_dra, pctiles=[1,99], pad=1.4))
ax1.set_ylim(nice_limits(every_dde, pctiles=[1,99], pad=1.4))
ax1.invert_xaxis()

## Add Gaia sources:
if context.gaia_csv:
    ax1.plot(gm._srcdata['ra'], gm._srcdata['dec'], marker='o', color='red',
            mfc='none', lw=0, ms=10)

## Disable axis offsets:
#ax1.xaxis.get_major_formatter().set_useOffset(False)
#ax1.yaxis.get_major_formatter().set_useOffset(False)

#ax1.plot(kde_pnts, kde_vals)

#blurb = "some text"
#ax1.text(0.5, 0.5, blurb, transform=ax1.transAxes)
#ax1.text(0.5, 0.5, blurb, transform=ax1.transAxes,
#      va='top', ha='left', bbox=dict(facecolor='white', pad=10.0))
#      fontdict={'family':'monospace'}) # fixed-width

#colors = cm.rainbow(np.linspace(0, 1, len(plot_list)))
#for camid, c in zip(plot_list, colors):
#    cam_data = subsets[camid]
#    xvalue = cam_data['CCDATEMP']
#    yvalue = cam_data['PIX_MED']
#    yvalue = cam_data['IMEAN']
#    ax1.scatter(xvalue, yvalue, color=c, lw=0, label=camid)

#mtickpos = [2,5,7]
#ndecades = 1.0   # for symlog, set width of linear portion in units of dex
#nonposx='mask' | nonposx='clip' | nonposy='mask' | nonposy='clip'
#ax1.set_xscale('log', basex=10, nonposx='mask', subsx=mtickpos)
#ax1.set_xscale('log', nonposx='clip', subsx=[3])
#ax1.set_yscale('symlog', basey=10, linthreshy=0.1, linscaley=ndecades)
#ax1.xaxis.set_major_formatter(formatter) # re-format x ticks
#ax1.set_ylim(ax1.get_ylim()[::-1])
#ax1.set_xlabel('whatever', labelpad=30)  # push X label down 

#ax1.set_xticks([1.0, 3.0, 10.0, 30.0, 100.0])
#ax1.set_xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'])
#for label in ax1.get_xticklabels():
#    label.set_rotax1 = fig.add_subplot(111)
#    ax1.grid(True)
#    ax1.scatter(expt, nsrc, c=irac) 
#
#    tion(30)
#    label.set_fontsize(14) 

#ax1.xaxis.label.set_fontsize(18)
#ax1.yaxis.label.set_fontsize(18)


#spts = ax1.scatter(x, y, lw=0, s=5)
#cnorm = mplcolors.Normalize(vmin=jdutc.min(), vmax=jdutc.max())
cnorm = mplcolors.Normalize(*spts.get_clim())
sm = plt.cm.ScalarMappable(norm=cnorm)
sm.set_array([])
cbar = fig.colorbar(sm, orientation='vertical')
cbar.formatter.set_useOffset(False)
cbar.update_ticks()

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')

# cyclical colormap ... cmocean.cm.phase
# cmocean: https://matplotlib.org/cmocean/




######################################################################
# CHANGELOG (slice_and_dice.py):
#---------------------------------------------------------------------
#
#  2019-10-16:
#     -- Increased __version__ to 0.0.1.
#     -- First created slice_and_dice.py.
#
