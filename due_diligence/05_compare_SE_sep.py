#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Compare extracted catalogs produced by SEP to those from SExtractor.
#
# Rob Siverd
# Created:       2019-11-12
# Last modified: 2019-11-12
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
__version__ = "0.1.0"

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
#import shutil
#import resource
import signal
import glob
import gc
import os
import sys
import time
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
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
#import window_filter as wf
#import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

import angle
reload(angle)

## Segment matching:
import segmatch1
reload(segmatch1)
smv1 = segmatch1.SegMatch()

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

#unlimited = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
#if (resource.getrlimit(resource.RLIMIT_DATA) == unlimited):
#    resource.setrlimit(resource.RLIMIT_DATA,  (3e9, 6e9))
#if (resource.getrlimit(resource.RLIMIT_AS) == unlimited):
#    resource.setrlimit(resource.RLIMIT_AS, (3e9, 6e9))

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

### Measure memory used so far:
#def check_mem_usage_MB():
#    max_kb_used = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#    return max_kb_used / 1000.0

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
    import astropy.io.ascii as aia
    import astropy.io.fits as pf
    import astropy.io.votable as av
    import astropy.table as apt
    import astropy.time as astt
    import astropy.wcs as awcs
#    from astropy import constants as aconst
#    from astropy import coordinates as coord
#    from astropy import units as uu
except ImportError:
    logger.error("astropy module not found!  Install and retry.")
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
def signal_handler(signum, frame):
    sys.stderr.write("\nInterrupted!\n\n")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

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
#class MyParser(argparse.ArgumentParser):
#    def error(self, message):
#        sys.stderr.write('error: %s\n' % message)
#        self.print_help()
#        sys.exit(2)
#
### Enable raw text AND display of defaults:
#class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
#                        argparse.RawDescriptionHelpFormatter):
#    pass
#
### Parse the command line:
#if __name__ == '__main__':
#
#    # ------------------------------------------------------------------
#    prog_name = os.path.basename(__file__)
#    descr_txt = """
#    PUT DESCRIPTION HERE.
#    
#    Version: %s
#    """ % __version__
#    parser = argparse.ArgumentParser(
#            prog='PROGRAM_NAME_HERE',
#            prog=os.path.basename(__file__),
#            #formatter_class=argparse.RawTextHelpFormatter)
#            description='PUT DESCRIPTION HERE.')
#            #description=descr_txt)
#    parser = MyParser(prog=prog_name, description=descr_txt)
#                          #formatter_class=argparse.RawTextHelpFormatter)
#    # ------------------------------------------------------------------
#    parser.set_defaults(thing1='value1', thing2='value2')
#    # ------------------------------------------------------------------
#    parser.add_argument('firstpos', help='first positional argument')
#    parser.add_argument('-w', '--whatever', required=False, default=5.0,
#            help='some option with default [def: %(default)s]', type=float)
#    parser.add_argument('-s', '--site',
#            help='Site to retrieve data for', required=True)
#    parser.add_argument('-n', '--number_of_days', default=1,
#            help='Number of days of data to retrieve.')
#    parser.add_argument('-o', '--output_file', 
#            default='observations.csv', help='Output filename.')
#    parser.add_argument('--start', type=str, default=None, 
#            help="Start time for date range query.")
#    parser.add_argument('--end', type=str, default=None,
#            help="End time for date range query.")
#    parser.add_argument('-d', '--dayshift', required=False, default=0,
#            help='Switch between days (1=tom, 0=today, -1=yest', type=int)
#    parser.add_argument('-e', '--encl', nargs=1, required=False,
#            help='Encl to make URL for', choices=all_encls, default=all_encls)
#    parser.add_argument('-s', '--site', nargs=1, required=False,
#            help='Site to make URL for', choices=all_sites, default=all_sites)
#    parser.add_argument('remainder', help='other stuff', nargs='*')
#    # ------------------------------------------------------------------
#    # ------------------------------------------------------------------
#    #iogroup = parser.add_argument_group('File I/O')
#    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
#    #        help='Output filename', type=str)
#    #iogroup.add_argument('-R', '--ref_image', default=None, required=True,
#    #        help='KELT image with WCS')
#    # ------------------------------------------------------------------
#    # ------------------------------------------------------------------
#    ofgroup = parser.add_argument_group('Output format')
#    fmtparse = ofgroup.add_mutually_exclusive_group()
#    fmtparse.add_argument('--python', required=False, dest='output_mode',
#            help='Return Python dictionary with results [default]',
#            default='pydict', action='store_const', const='pydict')
#    bash_var = 'ARRAY_NAME'
#    bash_msg = 'output Bash code snippet (use with eval) to declare '
#    bash_msg += 'an associative array %s containing results' % bash_var
#    fmtparse.add_argument('--bash', required=False, default=None,
#            help=bash_msg, dest='bash_array', metavar=bash_var)
#    fmtparse.set_defaults(output_mode='pydict')
#    # ------------------------------------------------------------------
#    # Miscellany:
#    miscgroup = parser.add_argument_group('Miscellany')
#    miscgroup.add_argument('--debug', dest='debug', default=False,
#            help='Enable extra debugging messages', action='store_true')
#    miscgroup.add_argument('-q', '--quiet', action='count', default=0,
#            help='less progress/status reporting')
#    miscgroup.add_argument('-v', '--verbose', action='count', default=0,
#            help='more progress/status reporting')
#    # ------------------------------------------------------------------
#
#    context = parser.parse_args()
#    context.vlevel = 99 if context.debug else (context.verbose-context.quiet)
#    context.prog_name = prog_name
#
##--------------------------------------------------------------------------##

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Where to find data:
cat_dir = {}
cat_dir['sex_orig'] = "/bigraid/rsiverd/ucd_scat_orig"
cat_dir['sex_ccln'] = "/bigraid/rsiverd/ucd_scat_ccln"
cat_dir['sep_fcat'] = "/bigraid/rsiverd/ucd_fcat"

ec_dir = cat_dir['sep_fcat']
se_dir = cat_dir['sex_orig']

#se_dir = {}
#se_dir['orig'] = cat_dir['sex_orig']
#se_dir['ccln'] = cat_dir['sex_ccln']

## List input files:
#cat_files = {kk:sorted(glob.glob(os.path.join(cpath, "SPIT*fits*"))) \
#                for kk,cpath in cat_dir.items()}

## List of SEP fcat files:
#fcat_list = sorted(glob.glob(os.path.join(cat_dir['sep_fcat'], "SPIT*fits")))
fcat_list = sorted(glob.glob(os.path.join(ec_dir, "SPIT*fits")))

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Cross-identification using seg-match:
def crossmatch_seg(x1, y1, x2, y2, tol=2):
    smv1.set_catalog1(x=x1, y=y1)
    smv1.set_catalog2(x=x2, y=y2)
    len_tol, len_bins = np.log10(1.10), 2
    ang_tol, ang_bins = 2., 1.
    len_rng = smv1.bintol_range(len_bins, len_tol)
    ang_rng = smv1.bintol_range(ang_bins, ang_tol)
    tdivs = (3,)
    use_ranges = (len_rng, ang_rng)
    use_nbins  = (len_bins, ang_bins)
    sys.stderr.write("use_ranges: %s\n" % str(use_ranges))
    sys.stderr.write("use_nbins:  %s\n" % str(use_nbins))
    best_pars = smv1.dither_hist_best_fit(use_ranges, use_nbins,
            tdivs, mode='weighted')
    sys.stderr.write("best_pars: %s\n" % str(best_pars))
    #for tx,ty in zip(x1, y1):
    #    pass

## Brute-force ross-identification by pixel position:
def crossmatch_xy(x1, y1, x2, y2, tol=1.0, verbose=False):
    ix1 = np.arange(x1.size)
    ix2 = np.arange(x2.size)
    matches = []
    for i1,tx,ty in zip(ix1, x1, y1):
        if verbose:
            sys.stderr.write("obj1 %d of %d ... " % (i1+2, ix1.size))
        #dx = x2 - tx
        ds = np.hypot(tx - x2, ty - y2)
        #sys.stderr.write("ds: %s\n" % str(ds))
        near = (ds < tol)
        near_i2 = ix2[near]
        near_ds =  ds[near]
        #sys.stderr.write("near: %s\n" % str(near))
        #nhit = np.sum(near)
        #sys.stderr.write("nhit: %d\n" % np.sum(near))
        hits = np.sum(near)
        if (hits > 1):
            sys.stderr.write("multiple hits??\n")
            sys.exit(1)
        if (hits == 1):
            if verbose:
                sys.stderr.write("nearby: %s" % str(ds[near]))
            found = (i1, near_i2[0], near_ds[0])
            if verbose:
                sys.stderr.write("  (%d <--> %d [%.3f])" % found)
            matches.append(found)
        if verbose:
            sys.stderr.write("\n")
        pass
    return matches

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
## Extended catalog object:
ec_data = ec.ExtendedCatalog()

## Iterate over the SEP fcat list:
ntodo = 150
nfiles = len(fcat_list)
diff_stats = {'msew_avg':[], 'msew_med':[],
              'mecw_avg':[], 'mecw_med':[],
              'usep_use_avg':[], 'usep_use_med':[],
              'wsep_wse_avg':[], 'wsep_wse_med':[],
              'nmatch':[]}
for ii,fcat_path in enumerate(fcat_list, 1):
    sys.stderr.write("fcat_path: %s\n" % fcat_path)
    fbase = os.path.basename(fcat_path)
    ftext = '_'.join(fbase.split('_')[:5])
    #sys.stderr.write("fbase: %s\n" % fbase)
    #sys.stderr.write("ftext: %s\n" % ftext)
    #se_cpath = glob.glob("%s/%s*fits*" % (sex_dir['ccln'], ftext))
    #se_opath = glob.glob("%s/%s*fits*" % (sex_dir['orig'], ftext))
    scat_list = glob.glob("%s/%s*fits*" % (se_dir, ftext))
    if (len(scat_list) != 1):
        sys.stderr.write("No/multiple matches???\n")
        sys.stderr.write("scat_list: %s\n" % str(scat_list))
        sys.exit(1)
    scat_path = scat_list[0]

    # load both data sets:
    se_data = pf.getdata(scat_path)
    se_objs = ec_data._unfitsify_recarray(se_data)
    ec_data.load_from_fits(fcat_path)
    ec_objs = ec_data.get_catalog()
    sx, sy = se_objs['XWIN_IMAGE'], se_objs['YWIN_IMAGE']
    ex, ey = ec_objs['x'], ec_objs['y']

    results = crossmatch_xy(sx, sy, ex, ey)
    if not results:
        continue
    i1, i2, ds = zip(*results)

    mse_objs = se_objs[i1,]
    mec_objs = ec_objs[i2,]

    diff_stats['nmatch'].append(len(i1))

    # SExtractor -- windowed vs. non:
    mse_wsep = np.hypot(mse_objs['X_IMAGE'] - mse_objs['XWIN_IMAGE'],
                        mse_objs['Y_IMAGE'] - mse_objs['YWIN_IMAGE'])
    diff_stats['msew_avg'].append(np.average(mse_wsep))
    diff_stats['msew_med'].append(np.median(mse_wsep))
    
    # SEP -- windowed vs. non:
    mec_wsep = np.hypot(mec_objs['x'] - mec_objs['wx'],
                        mec_objs['y'] - mec_objs['wy'])
    diff_stats['mecw_avg'].append(np.average(mec_wsep))
    diff_stats['mecw_med'].append(np.median(mec_wsep))

    # SExtractor vs. SEP (unwindowed):
    use_sep = np.hypot(mse_objs['X_IMAGE'] - mec_objs['x'],
                      mse_objs['Y_IMAGE'] - mec_objs['y'])
    diff_stats['usep_use_avg'].append(np.average(use_sep))
    diff_stats['usep_use_med'].append(np.median(use_sep))

    # SExtractor vs. SEP (windowed):
    wse_sep = np.hypot(mse_objs['XWIN_IMAGE'] - mec_objs['wx'],
                      mse_objs['YWIN_IMAGE'] - mec_objs['wy'])
    diff_stats['wsep_wse_avg'].append(np.average(wse_sep))
    diff_stats['wsep_wse_med'].append(np.median(wse_sep))

    sys.stderr.write("\n")
    if (ntodo > 0) and (ii >= ntodo):
        break
    

pkeys = list(diff_stats.keys())
pkeys.remove('nmatch')
plt.clf()
for item in pkeys:
    plt.plot(diff_stats[item], label=item)
plt.grid()
plt.legend(loc='upper right')
plt.savefig('various_quantities.png', bbox='tight')

plt.clf()
plt.plot(diff_stats['nmatch'], label='nmatch')
plt.grid()
plt.legend(loc='upper left')
plt.savefig('number_of_matches.png', bbox='tight')

## Quick ASCII I/O:
#data_file = 'data.txt'
#gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
#gftkw.update({'names':True, 'autostrip':True})
#gftkw.update({'delimiter':'|', 'comments':'%0%0%0%0'})
#gftkw.update({'loose':True, 'invalid_raise':False})
#all_data = np.genfromtxt(data_file, dtype=None, **gftkw)
#all_data = aia.read(data_file)

#all_data = pd.read_csv(data_file)
#all_data = pd.read_table(data_file, delim_whitespace=True)
#all_data = pd.read_table(data_file, sep='|')
#fields = all_data.dtype.names
#if not fields:
#    x = all_data[:, 0]
#    y = all_data[:, 1]
#else:
#    x = all_data[fields[0]]
#    y = all_data[fields[1]]

#vot_file = 'neato.xml'
#vot_data = av.parse_single_table(vot_file)
#vot_data = av.parse_single_table(vot_file).to_table()

##--------------------------------------------------------------------------##
## Misc:
#def log_10_product(x, pos):
#   """The two args are the value and tick position.
#   Label ticks with the product of the exponentiation."""
#   return '%.2f' % (x)  # floating-point
#
#formatter = plt.FuncFormatter(log_10_product) # wrap function for use

## Convenient, percentile-based plot limits:
#def nice_limits(vec, pctiles=[1,99], pad=1.2):
#    ends = np.percentile(vec[~np.isnan(vec)], pctiles)
#    middle = np.average(ends)
#    return (middle + pad * (ends - middle))

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
## Solve prep:
#ny, nx = img_vals.shape
#x_list = (0.5 + np.arange(nx)) / nx - 0.5            # relative (centered)
#y_list = (0.5 + np.arange(ny)) / ny - 0.5            # relative (centered)
#xx, yy = np.meshgrid(x_list, y_list)                 # relative (centered)
#xx, yy = np.meshgrid(nx*x_list, ny*y_list)           # absolute (centered)
#xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))   # absolute
#yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij') # absolute
#yy, xx = np.nonzero(np.ones_like(img_vals))          # absolute
#yy, xx = np.mgrid[0:ny,   0:nx].astype('uint16')     # absolute (array)
#yy, xx = np.mgrid[1:ny+1, 1:nx+1].astype('uint16')   # absolute (pixel)

## 1-D vectors:
#x_pix, y_pix, ivals = xx.flatten(), yy.flatten(), img_vals.flatten()
#w_vec = np.ones_like(ivals)            # start with uniform weights
#design_matrix = np.column_stack((np.ones(x_pix.size), x_pix, y_pix))

## Image fitting (statsmodels etc.):
#data = sm.datasets.stackloss.load()
#ols_res = sm.OLS(ivals, design_matrix).fit()
#rlm_res = sm.RLM(ivals, design_matrix).fit()
#rlm_model = sm.RLM(ivals, design_matrix, M=sm.robust.norms.HuberT())
#rlm_res = rlm_model.fit()
#data = pd.DataFrame({'xpix':x_pix, 'ypix':y_pix})
#rlm_model = sm.RLM.from_formula("ivals ~ xpix + ypix", data)

##--------------------------------------------------------------------------##
## Theil-Sen line-fitting (linear):
#model = ts.linefit(xvals, yvals)
#icept, slope = ts.linefit(xvals, yvals)

## Theil-Sen line-fitting (loglog):
#xvals, yvals = np.log10(original_xvals), np.log10(original_yvals)
#xvals, yvals = np.log10(df['x'].values), np.log10(df['y'].values)
#llmodel = ts.linefit(np.log10(xvals), np.log10(yvals))
#icept, slope = ts.linefit(xvals, yvals)
#fit_exponent = slope
#fit_multiplier = 10**icept
#bestfit_x = np.arange(5000)
#bestfit_y = fit_multiplier * bestfit_x**fit_exponent

## Log-log evaluator:
#def loglog_eval(xvals, model):
#    icept, slope = model
#    return 10**icept * xvals**slope
#def loglog_eval(xvals, icept, slope):
#    return 10**icept * xvals**slope

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
ax1 = fig.add_subplot(111)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
ax1.grid(True)
#ax1.axis('off')

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
#    label.set_rotation(30)
#    label.set_fontsize(14) 

#ax1.xaxis.label.set_fontsize(18)
#ax1.yaxis.label.set_fontsize(18)

#ax1.set_xlim(nice_limits(xvec, pctiles=[1,99], pad=1.2))
#ax1.set_ylim(nice_limits(yvec, pctiles=[1,99], pad=1.2))

#spts = ax1.scatter(x, y, lw=0, s=5)
##cbar = fig.colorbar(spts, orientation='vertical')   # old way
#cbnorm = mplcolors.Normalize(*spts.get_clim())
#sm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
#sm.set_array([])
#cbar = fig.colorbar(sm, orientation='vertical')
#cbar = fig.colorbar(sm, ticks=cs.levels, orientation='vertical') # contours
#cbar.formatter.set_useOffset(False)
#cbar.update_ticks()

fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')


######################################################################
# CHANGELOG (05_compare_SE_sep.py):
#---------------------------------------------------------------------
#
#  2019-11-12:
#     -- Increased __version__ to 0.1.0.
#     -- First created 05_compare_SE_sep.py.
#
