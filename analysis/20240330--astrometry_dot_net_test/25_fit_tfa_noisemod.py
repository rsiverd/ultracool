#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Load per-object data sets from the by_object folder, fit astrometric
# solutions to individual objects, perform TFA detrending, and fit a noise
# model to the results.
#
# Rob Siverd
# Created:       2024-12-09
# Last modified: 2024-12-09
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

## Because obviously:
#import warnings
#if not sys.warnoptions:
#    warnings.simplefilter("ignore", category=DeprecationWarning)
#    warnings.simplefilter("ignore", category=UserWarning)
#    warnings.simplefilter("ignore")
#    warnings.simplefilter('error')    # halt on warnings
#with warnings.catch_warnings():
#    some_risky_activity()
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
#    import problem_child1, problem_child2

## Astrometry fitting module:
import astrom_test_2
reload(astrom_test_2)
at2 = astrom_test_2
#af  = at2.AstFit()  # used for target
afn = at2.AstFit()  # used for neighbors

## Detrending facility:
import detrending
reload(detrending)


# Magnitude to flux conversion:
def kadu(mag, zeropt=25.0):
    return 10.0**(0.4 * (zeropt - mag))

# Flux to magnitude conversion:
def kmag(adu, zeropt=25.0):
    return (zeropt - 2.5 * np.log10(adu))

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

### Measure memory used so far:
#def check_mem_usage_MB():
#    max_kb_used = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#    return max_kb_used / 1000.0

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

## Extract gaia ID from filename:
def gaia_id_from_filename(filename):
    return int(os.path.basename(filename).split('_')[1].split('.')[0])

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Get list of CSV files with object data:
targ_dir = 'by_object'
csv_files = glob.glob('%s/gm_*.csv' % targ_dir)

## Promote files list to dictionary:
targ_files = {gaia_id_from_filename(x):x for x in csv_files}

## Load data into dictionaries:
sys.stderr.write("Loading data ... ") 
targ_data = {}
tik = time.time()
targ_data = {gg:pd.read_csv(cc) for gg,cc in targ_files.items()}
tok = time.time()
sys.stderr.write("done. Took %.3f seconds.\n" % (tok-tik))

## Measure size of each data set:
targ_npts = {gg:len(dd) for gg,dd in targ_data.items()}

## Select some large ones:
min_pts = 1000
biggies = {gg:nn for gg,nn in targ_npts.items() if nn > min_pts}
sys.stderr.write("Have %d sources with N>%d.\n" % (len(biggies), min_pts))
proc_objs = list(biggies.keys())
proc_data = {sid:targ_data[sid] for sid in proc_objs}


##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

sys.stderr.write("Creating fitters ... ")
fitters = {sid:at2.AstFit() for sid in proc_objs}
sys.stderr.write("done.\n")

## Run processing:
num_todo = 0
sig_thresh = 3
save_fitters  = {}
save_bestpars = {}
maxiters = 30
pruned_results = {}
for ii,targ in enumerate(proc_objs, 1):
    sys.stderr.write("%s\n" % fulldiv)
    sys.stderr.write("Initial fit of: %s\n" % targ)
    afn = at2.AstFit()
    _data = proc_data[targ].to_records()
    #afn.setup(_data, ra_key='calc_ra', de_key='calc_de')
    #afn.setup(_data, ra_key='dra', de_key='dde')
    #afn.setup(_data, ra_key='mean_anet_ra', de_key='mean_anet_de')
    afn.setup(_data, ra_key='jntupd_ra', de_key='jntupd_de')
    bestpars = afn.fit_bestpars(sigcut=sig_thresh)
    if not isinstance(bestpars, np.ndarray):
        sys.stderr.write("Error performing fit!\n")
        sys.exit(1)
    iterpars = bestpars.copy()
    for i in range(maxiters):
        sys.stderr.write("Iteration %d ...\n" % i)
        iterpars = afn.iter_update_bestpars(iterpars)
        if afn.is_converged():
            sys.stderr.write("Converged!  Iteration: %d\n" % i)
            break
    save_bestpars[targ] = afn.nice_units(iterpars)
    save_fitters[targ] = afn
    pruned_results[targ] = afn.collect_result_dataset(prune_outliers=True)
    if (num_todo > 0) and (ii >= num_todo):
        break
    pass

## Initial gathering of data for undetrended RMS plot (plus trend selection):
full_data = []
for ii,targ in enumerate(proc_objs, 1):
    #afn = save_fitters[targ]
    #_data = afn.collect_result_dataset(prune_outliers=True)
    #errs_ra, errs_de = afn.get_radec_minus_model_mas(cos_dec_mult=True)
    #ra_deltas.append(errs_ra)
    #de_deltas.append(errs_de)
    #meas_flux.append(afn.dataset['flux'])
    #meas_filt.append(afn.dataset['filter'])
    #tdata = pd.DataFrame.from_records(_data)
    #tdata = pd.DataFrame.from_records(
    #            afn.collect_result_dataset(prune_outliers=True))
    tdata = pd.DataFrame.from_records(pruned_results[targ])
    #tdata = pd.DataFrame.from_records(afn.dataset)
    #tdata['ra_deltas_mas'] = errs_ra
    #tdata['de_deltas_mas'] = errs_de
    #tdata['tot_delta_mas'] = np.hypot(errs_ra, errs_de)
    tdata['tot_delta_mas'] = np.hypot(tdata['fit_resid_ra_mas'], tdata['fit_resid_de_mas'])
    tdata['fake_fwhm'] = 2. * np.sqrt(tdata['a'] * tdata['b'])
    tdata['instmag'] = kmag(tdata['flux'])
    full_data.append(tdata)
    pass

full_data = pd.concat(full_data, ignore_index=True)

jwhich = (full_data['filter'] == 'J')
hwhich = (full_data['filter'] == 'H2')

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Trend star selection:
tr_npts = 1500
trcount = {'UL':0, 'UR':0, 'LL':0, 'LR':0}
tr_qmax = 2
trend_targlist = []
for ii,targ in enumerate(proc_objs, 1):
    #vals = save_fitters[targ].dataset
    #vals = save_fitters[targ].collect_result_dataset(prune_outliers=True)
    vals = pruned_results[targ]
    # avoid skimpy data sets:
    if len(vals) < tr_npts:
        continue
    avgx = np.average(vals['x'])
    avgy = np.average(vals['y'])
    # Lower-left:
    if (trcount['LL'] < tr_qmax):
        if (avgx <  500.0) and (avgy <  500.0):
            trend_targlist.append(targ)
            trcount['LL'] += 1
    # Lower-right:
    if (trcount['LR'] < tr_qmax):
        if (avgx > 1000.0) and (avgy <  500.0):
            trend_targlist.append(targ)
            trcount['LR'] += 1
    # Upper-left:
    if (trcount['UL'] < tr_qmax):
        if (avgx <  500.0) and (avgy > 1000.0):
            trend_targlist.append(targ)
            trcount['UL'] += 1
    # Upper-right:
    if (trcount['UR'] < tr_qmax):
        if (avgx > 1000.0) and (avgy > 1000.0):
            trend_targlist.append(targ)
            trcount['UR'] += 1

## Collect residual vectors from trend targets:
trend_resid_vecs = {}   # JDTDB, RA, DE, instr
want_trend_cols = ('jdtdb', 'fit_resid_ra_mas', 'fit_resid_de_mas', 'instrument')
for targ in trend_targlist:
    #this_fit   = save_fitters[targ]
    #this_jdtdb = this_fit.dataset['jdtdb']
    #this_instr = this_fit.dataset['instrument']
    #ra_errs_mas, de_errs_mas = \
    #        this_fit.get_radec_minus_model_mas(cos_dec_mult=True)
    #trend_resid_vecs[targ] = \
    #        (this_jdtdb, ra_errs_mas, de_errs_mas, this_instr)
    #tdata = save_fitters[targ].
    #_data = save_fitters[targ].collect_result_dataset(prune_outliers=True)
    _data = pruned_results[targ]
    trend_resid_vecs[targ] = [_data[x] for x in want_trend_cols]
    pass

## Detrend residuals:
#ICD_RA = detrending.InstCooDetrend()
#ICD_DE = detrending.InstCooDetrend()
save_dtr_ra = {}
save_dtr_de = {}
want_ra_cols = ('jdtdb', 'fit_resid_ra_mas', 'instrument')
want_de_cols = ('jdtdb', 'fit_resid_ra_mas', 'instrument')
for targ in proc_objs:
    sys.stderr.write("%s\n" % fulldiv)
    sys.stderr.write("Target: %s\n" % targ)
    this_ICD_RA = detrending.InstCooDetrend()
    this_ICD_DE = detrending.InstCooDetrend()
    this_ICD_RA.reset()
    this_ICD_DE.reset()
    others = [x for x in trend_targlist if x!= targ]

    # load object data into detrender:
    #_data = save_fitters[targ].collect_result_dataset(prune_outliers=True)
    _data = pruned_results[targ]
    #this_ICD_RA.set_data(_data['jdtdb'], _data['fit_resid_ra_mas'], _data['instrument'])
    this_ICD_RA.set_data(*[_data[x] for x in want_ra_cols])
    this_ICD_RA.set_data(*[_data[x] for x in want_de_cols])

    ## load object data into detrender:
    #this_fit = save_fitters[targ]
    #this_jdtdb = this_fit.dataset['jdtdb']
    #this_instr = this_fit.dataset['instrument']
    #ra_errs_mas, de_errs_mas = \
    #        this_fit.get_radec_minus_model_mas(cos_dec_mult=True)
    #this_ICD_RA.set_data(this_jdtdb, ra_errs_mas, this_instr)
    #this_ICD_DE.set_data(this_jdtdb, de_errs_mas, this_instr)

    # load trend data into detrender:
    for trtarg in others:
        tr_jdtdb, tr_ra_errs, tr_de_errs, tr_inst = trend_resid_vecs[trtarg]
        this_ICD_RA.add_trend(trtarg, tr_jdtdb, tr_ra_errs, tr_inst)
        this_ICD_DE.add_trend(trtarg, tr_jdtdb, tr_de_errs, tr_inst)
        pass
    # clean it up:
    this_ICD_RA.detrend()
    this_ICD_DE.detrend()
    # save for later:
    save_dtr_ra[targ] = this_ICD_RA
    save_dtr_de[targ] = this_ICD_DE

## Gather data for analysis after detrending:
full_data = []
for ii,targ in enumerate(proc_objs, 1):
    #this_fit = save_fitters[targ]
    #tdata = pd.DataFrame.from_records(this_fit.dataset)
    #errs_ra, errs_de = this_fit.get_radec_minus_model_mas(cos_dec_mult=True)

    this_dtr_ra = save_dtr_ra[targ]
    this_dtr_de = save_dtr_de[targ]
    clean_ra_resids = this_dtr_ra.get_results()[1]
    clean_de_resids = this_dtr_de.get_results()[1]
    clean_tot_resid = np.hypot(clean_ra_resids, clean_de_resids)


    tdata = pd.DataFrame.from_records(pruned_results[targ])
    tdata['raw_tot_delta_mas'] = np.hypot(tdata['fit_resid_ra_mas'], tdata['fit_resid_de_mas'])

    #meas_flux.append(afn.dataset['flux'])
    #meas_filt.append(afn.dataset['filter'])
    #tdata = pd.DataFrame.from_records(afn.dataset)
    #tdata['raw_ra_deltas_mas'] = errs_ra
    #tdata['raw_de_deltas_mas'] = errs_de
    #tdata['raw_tot_delta_mas'] = np.hypot(errs_ra, errs_de)
    tdata['cln_ra_deltas_mas'] = clean_ra_resids
    tdata['cln_de_deltas_mas'] = clean_de_resids
    tdata['cln_tot_delta_mas'] = np.hypot(clean_ra_resids, clean_de_resids)
    tdata['fake_fwhm'] = 2. * np.sqrt(tdata['a'] * tdata['b'])
    tdata['instmag'] = kmag(tdata['flux'])
    full_data.append(tdata)
    pass

full_data = pd.concat(full_data, ignore_index=True)
#ra_deltas = full_data['ra_deltas_mas']
#de_deltas = full_data['de_deltas_mas']
#tot_delta = np.hypot(ra_deltas, de_deltas)
#inst_mags = full_data['inst_mag']
#tot_delta = full_data

jwhich = (full_data['filter'] == 'J')
hwhich = (full_data['filter'] == 'H2')

##--------------------------------------------------------------------------##
## Binning to aid the plot:

bwid = 0.25

jmin = 6.5
hmin = 9.5
jsubset = full_data[jwhich].copy()
jsubset['bin'] = np.int_((jsubset['instmag'] - jmin) / bwid)
jb_avg = jsubset.groupby("bin").mean()
jb_med = jsubset.groupby("bin").median()
del jsubset

hsubset = full_data[hwhich].copy()
hsubset['bin'] = np.int_((hsubset['instmag'] - hmin) / bwid)
hb_avg = hsubset.groupby("bin").mean()
hb_med = hsubset.groupby("bin").median()

#sys.exit(0)
##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig_dims = (11, 9)
fig = plt.figure(1, figsize=fig_dims)
#plt.gcf().clf()
fig.clf()

skw = {'lw':0, 's':5}
ax1 = fig.add_subplot(111)
ax1.scatter(full_data['x'], full_data['y'], **skw)
plot_name = 'raw_xyposns.png'
fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
plt.draw()
fig.savefig(plot_name, bbox_inches='tight')



# ----------------------------------------------------------------------- 
# RAW RMS PLOT:
fig.clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
for ax in (ax1, ax2):
    ax.patch.set_facecolor((0.8, 0.8, 0.8))
    ax.grid(True)
#ax1.axis('off')

full_plot = 'raw_scatter_full.png'
crop_plot = 'raw_scatter_crop.png'

skw = {'lw':0, 's':5}
ax1.scatter(full_data['instmag'][jwhich], 
            full_data['raw_tot_delta_mas'][jwhich], label='J', **skw)
ax1.plot(jb_avg['instmag'], jb_avg['raw_tot_delta_mas'], c='r', label='J per-bin avg')
ax1.plot(jb_med['instmag'], jb_med['raw_tot_delta_mas'], c='g', label='J per-bin med')
ax2.scatter(full_data['instmag'][hwhich],
            full_data['raw_tot_delta_mas'][hwhich], label='H2', **skw)
ax2.plot(hb_avg['instmag'], hb_avg['raw_tot_delta_mas'], c='r', label='H per-bin avg')
ax2.plot(hb_med['instmag'], hb_med['raw_tot_delta_mas'], c='g', label='H per-bin med')
ax2.set_xlabel('instrumental mag')
#ax1.set_ylabel('total residual [mas]')
fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
[ax.legend(loc='best') for ax in (ax1,ax2)]
fig.savefig(full_plot, bbox_inches='tight')
for ax in (ax1, ax2):
    ax.set_ylim(-10, 250)
    ax.set_ylabel('total residual [mas]')
#ax1.set_ylim(-10, 250)
ax1.set_xlim(6.5, 11.5)
plt.draw()
fig.savefig(crop_plot, bbox_inches='tight')

# ----------------------------------------------------------------------- 
# TFA RMS PLOT:
fig.clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
for ax in (ax1, ax2):
    ax.patch.set_facecolor((0.8, 0.8, 0.8))
    ax.grid(True)

full_plot = 'tfa_scatter_full.png'
crop_plot = 'tfa_scatter_crop.png'

skw = {'lw':0, 's':5}
ax1.scatter(full_data['instmag'][jwhich], 
            full_data['cln_tot_delta_mas'][jwhich], label='J', **skw)
ax1.plot(jb_avg['instmag'], jb_avg['cln_tot_delta_mas'], c='r', label='J per-bin avg')
ax1.plot(jb_med['instmag'], jb_med['cln_tot_delta_mas'], c='g', label='J per-bin med')
ax2.scatter(full_data['instmag'][hwhich],
            full_data['cln_tot_delta_mas'][hwhich], label='H2', **skw)
ax2.plot(hb_avg['instmag'], hb_avg['cln_tot_delta_mas'], c='r', label='H per-bin avg')
ax2.plot(hb_med['instmag'], hb_med['cln_tot_delta_mas'], c='g', label='H per-bin med')
ax2.set_xlabel('instrumental mag')
#ax1.set_ylabel('total residual [mas]')
fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
[ax.legend(loc='best') for ax in (ax1,ax2)]
fig.savefig(full_plot, bbox_inches='tight')
for ax in (ax1, ax2):
    ax.set_ylim(-10, 250)
    ax.set_ylabel('total residual [mas]')
#ax1.set_ylim(-10, 250)
ax1.set_xlim(6.5, 11.5)
plt.draw()
fig.savefig(crop_plot, bbox_inches='tight')

## Polar scatter:
#skw = {'lw':0, 's':15}
#ax1.scatter(azm_rad, zdist_deg, **skw)

## For polar axes:
#ax1.set_rmin( 0.0)                  # if using altitude in degrees
#ax1.set_rmax(90.0)                  # if using altitude in degrees
#ax1.set_theta_direction(-1)         # clockwise
#ax1.set_theta_direction(+1)         # counterclockwise
#ax1.set_theta_zero_location("N")    # North-up
#ax1.set_rlabel_position(-30.0)      # move labels 30 degrees

## Disable axis offsets:
#ax1.xaxis.get_major_formatter().set_useOffset(False)
#ax1.yaxis.get_major_formatter().set_useOffset(False)

#ax1.plot(kde_pnts, kde_vals)

#ax1.pcolormesh(xx, yy, ivals)

#blurb = "some text"
#ax1.text(0.5, 0.5, blurb, transform=ax1.transAxes)
#ax1.text(0.5, 0.5, blurb, transform=ax1.transAxes,
#      va='top', ha='left', bbox=dict(facecolor='white', pad=10.0))
#      fontdict={'family':'monospace'}) # fixed-width
#      fontdict={'fontsize':24}) # larger typeface

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

#ax1.legend(loc='best', prop={'size':24})

#spts = ax1.scatter(x, y, lw=0, s=5)
##cbar = fig.colorbar(spts, orientation='vertical')   # old way
#cbnorm = mplcolors.Normalize(*spts.get_clim())
#scm = plt.cm.ScalarMappable(norm=cbnorm, cmap=spts.cmap)
#scm.set_array([])
#cbar = fig.colorbar(scm, orientation='vertical')
#cbar = fig.colorbar(scm, ticks=cs.levels, orientation='vertical') # contours
#cbar.formatter.set_useOffset(False)
#cbar.update_ticks()

#fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
#plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')

# cyclical colormap ... cmocean.cm.phase
# cmocean: https://matplotlib.org/cmocean/




######################################################################
# CHANGELOG (25_fit_tfa_noisemod.py):
#---------------------------------------------------------------------
#
#  2024-12-09:
#     -- Increased __version__ to 0.0.1.
#     -- First created 25_fit_tfa_noisemod.py.
#
