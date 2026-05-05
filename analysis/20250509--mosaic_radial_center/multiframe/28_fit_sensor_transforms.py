#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Attempt to fit an 18-parameter model linking the NW, SE, and SW sensor
# pixel grids to that of the NE sensor.
#
# Rob Siverd
# Created:       2026-04-07
# Last modified: 2026-04-14
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.1"

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
#import resource
#import signal
import glob
#import io
import gc
import os
import ast
import sys
import time
import pprint
import pickle
#import ephem
import numpy as np
#from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#import scipy.linalg as sla
#import scipy.signal as ssig
#import scipy.ndimage as ndi
import scipy.optimize as opti
#import scipy.interpolate as stp
#import scipy.spatial.distance as ssd
#import scipy.stats as scst
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import matplotlib.cm as cm
#import matplotlib.ticker as mt
#import matplotlib._pylab_helpers as hlp
#from matplotlib.colors import LogNorm
#import matplotlib.colors as mplcolors
#import matplotlib.collections as mcoll
#import matplotlib.gridspec as gridspec
from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
np.set_printoptions(suppress=True, linewidth=160)
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import theil_sen as ts
import itertools as itt
_have_np_vers = float('.'.join(np.__version__.split('.')[:2]))

## Angular math routines:
import angle
reload(angle)

## Gaia catalog matching:
import gaia_match
reload(gaia_match)
gm  = gaia_match.GaiaMatch()

## Storage structure for analysis results:
try:
    import extended_catalog
    reload(extended_catalog)
    ecl = extended_catalog.ExtendedCatalog()
except ImportError:
    logger.error("failed to import extended_catalog module!")
    sys.exit(1)

## Coordinate solve helpers:
import slv_helper
reload(slv_helper)
slvh = slv_helper

## Solution parameter helpers:
import slv_par_tools
reload(slv_par_tools)
spt = slv_par_tools
quads = spt._quads


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
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Extract RUNID from filename:
def runid_from_filename(filename):
    return os.path.basename(filename).split('_')[1]

## Extract observing year from RUNID:
def year_from_runid(runid):
    return int(runid[:2])

## Load parameter set from file:
def load_parameters(filename):
    with open(filename, 'r') as fff:
        return ast.literal_eval(fff.read())

## Extract CD matrix and CRPIX from parameter list:
def get_cdm_crpix(parameters):
    return np.array(parameters[:24]).reshape(4, -1)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


## QRUNID NOTES:
## For this first attempt, I am restricting myself to the joint solutions
## of QRUNIDs from 2011 - 2016, inclusive. After 2016, jointly fitted WCS
## for the various detectors do not agree as well. It is unclear whether this
## is because of poor initial guesses, bad sensor behavior (i.e., artifacts
## creating false matches), or genuinely different geometry. Because of this,
## I am suspicious of the Gaia match quality in the updated catalogs for
## QRUNIDs starting in 2017 and plan to exclude them from this initial work.

## List of available J-only joint parameter files:
par_flist = sorted(glob.glob('joint_pars/jpars_??????_J.txt'))
par_runid = [runid_from_filename(x) for x in par_flist]
par_files = dict(zip(par_runid, par_flist))

## Clean these out based on year/semester:
ythresh = 11 # 16
ythresh = 18 # 16
#ythresh = 16
ythresh = 25
keepers = {}
for rr,fname in par_files.items():
    if year_from_runid(rr) <= ythresh:
    #if year_from_runid(rr) >= ythresh:
        keepers[rr] = fname
par_files = keepers

## Load those files:
#raw_params = {kk:load_parameters(vv) for kk,vv in par_files.items()}
raw_params = {}
raw_inames = {}
for runid,fname in par_files.items():
    raw_params[runid], raw_inames[runid] = load_parameters(fname)

qrun_list = list(raw_params.keys())

##--------------------------------------------------------------------------##
##------------------         Load Pickled Catalogs          ----------------##
##--------------------------------------------------------------------------##

## Only keep columns from gstars we actually use to save space:
gstcols = ['x', 'y', 'flux', 'flag', 
           'fwhm', 'dumbsnr', 'realerr', 'gid', 'gra', 'gde']

## Iterate over QRUNIDs in use:
all_data = {}       # diags data from single-image best-fit (not used)
all_gstr = {}       # star catalog with updated Gaia matches
all_pars = {}       # initial, single-image best-fit parameters (not used)
for qrun in qrun_list:
    qrun_data, qrun_gstr, qrun_pars = {}, {}, {}    # per-QRUNID of above
    qrun_dir = 'results/%s' % qrun
    sys.stderr.write("qrun: %s\n" % qrun)
    sys.stderr.write("qrun_dir: %s\n" % qrun_dir)
    if not os.path.isdir(qrun_dir):
        sys.stderr.write("Error: folder not found: %s\n" % qrun_dir)
        sys.exit(1)
    #pickles = ['{}/{}.pickle'.format(qrun_dir, x) for x in raw_inames[qrun]]
    use_pickles = {x:qrun_dir+'/'+x+'.pickle' for x in raw_inames[qrun]}
    if not all([os.path.isfile(x) for x in use_pickles.values()]):
        sys.stderr.write("Error: pickled catalog(s) missing ...\n")
        sys.exit(1)
    # Load pickles:
    for pbase,ppath in use_pickles.items():
        _gstr, _data, _pars = load_pickled_object(ppath)
        trim_gstr = {qq:gg[gstcols].copy() for qq,gg in _gstr.items()}
        #qrun_gstr[pbase] = _gstr
        #qrun_gstr[pbase] = _gstr[gstcols].copy()    # keep useful columns
        qrun_gstr[pbase] = trim_gstr
        #qrun_data[pbase] = _data
        qrun_pars[pbase] = _pars
    # Stash finished dictionaries:
    all_gstr[qrun] = qrun_gstr
    #all_data[qrun] = qrun_data
    all_pars[qrun] = qrun_pars
    del qrun_gstr, qrun_data, qrun_pars
    del _gstr, _data, _pars

## Old all_gstr (full size): ~11.9 GiB
## New trim_gstr (useful cols): ~2.0 GiB

gcounts = {}
gc_avgs = {}
gc_meds = {}
for qrunid,dataset in all_gstr.items():
    imqstars = {img:{qq:len(cat) for qq,cat in vvv.items()} \
            for img,vvv in dataset.items()}
    gcounts[qrunid] = imqstars
    avgcount = np.mean([list(x.values()) for x in imqstars.values()], axis=0)
    medcount = np.median([list(x.values()) for x in imqstars.values()], axis=0)
    gc_avgs[qrunid] = dict(zip(spt._quads, avgcount))
    gc_meds[qrunid] = dict(zip(spt._quads, medcount))
    pass
#{qq:len(cat) for qq,cat in dataset['wircam_J_2887487p'].items()}

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Initial guesses for the sensor transformation:
#nw_xf_guess = spt.mkaffine([1.0, 0.0, 0.0, 1.0, 2184.0,    -9.0])
#se_xf_guess = spt.mkaffine([1.0, 0.0, 0.0, 1.0,   -4.0, -2192.5])
#sw_xf_guess = spt.mkaffine([1.0, 0.0, 0.0, 1.0, 2186.5, -2205.0])
#nw_xf_guess = np.array([1.0, 0.0, 0.0, 1.0, 2184.0,    -9.0])
#se_xf_guess = np.array([1.0, 0.0, 0.0, 1.0,   -4.0, -2192.5])
#sw_xf_guess = np.array([1.0, 0.0, 0.0, 1.0, 2186.5, -2205.0])

## First attempt:
xf_guess = {
#       'NW'  :  np.array([1.0, 0.0, 0.0, 1.0, 2184.0,    -9.0]),
        'NW'  :  np.array([1.0, 0.0, 0.0, 1.0, 2183.9,    -8.6]),
#       'SE'  :  np.array([1.0, 0.0, 0.0, 1.0,   -4.0, -2192.5]),
        'SE'  :  np.array([1.0, 0.0, 0.0, 1.0,   -0.7, -2195.5]),
#       'SW'  :  np.array([1.0, 0.0, 0.0, 1.0, 2186.5, -2205.0]),
        'SW'  :  np.array([1.0, 0.0, 0.0, 1.0, 2192.0, -2204.1]),
}

# all_gstr['11AQ15']['wircam_J_1319377p']['NW']
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

qrun = '11AQ15'
sensor = 'NW'
quads_to_fit = ['NW', 'SE', 'SW']
qrun_batch = qrun_list
#qrun_batch = qrun_list[-3:]

## Attempt a solve with levmar ...
#slvkw = {'method':'trf', 'xtol':1e-14, 'ftol':1e-14}

#slvkw = {'method':'trf', 'xtol':1e-8, 'ftol':1e-8}
slvkw = {'method':'lm', 'xtol':1e-8, 'ftol':1e-8}
#slvkw = {'method':'trf', 'xtol':1e-8, 'ftol':1e-8, 'loss':'huber'}
slvkw.update({'max_nfev':10})
reskw = {'unsquared':True}
total_result = {}
for qrun in qrun_batch:
    runid_result = {}
    for fitq in quads_to_fit:
        sys.stderr.write("Attempting minimization (%s, %s) ...\n"
                         % (qrun, fitq))
        tik = time.time()
        minimize_this = partial(spt.multi_squared_residuals_foc2ccd_rdist_xform,
                                    params=np.array(raw_params.get(qrun)),
                                    mdataset=all_gstr.get(qrun),
                                    imlist=raw_inames.get(qrun), 
                                    sensor=fitq)
    
        #minimize_this = partial(spt.multi_squared_residuals_foc2ccd_rdist,
        #                        mdataset=all_gstr, imlist=iname_order)
        tguess = xf_guess[fitq]
        answer = opti.least_squares(minimize_this, tguess, 
                                    kwargs=reskw, **slvkw)
        sys.stderr.write("Ended up with: %s\n" % str(answer))
        sys.stderr.write("Ended up with: %s\n" % str(answer['x']))
        tok = time.time()
        lsq_taken = tok - tik
        sys.stderr.write("Solve took %.2f seconds.\n" % lsq_taken)
        
        sys.stderr.write("%s\n%s\n" % (fulldiv, fulldiv))
        runid_result[fitq] = answer['x']
        pass
    total_result[qrun] = runid_result    

##--------------------------------------------------------------------------##
## Dump median results to screen:
for fitq in quads_to_fit:
    med_pars = np.median([x.get(fitq) for x in total_result.values()], axis=0)
    sys.stderr.write("%s: %s\n" % (fitq, str(med_pars)))

## Median transform:
medtxform = {qq:np.median([x.get(qq) for x in total_result.values()],axis=0) \
                    for qq in quads_to_fit}
##--------------------------------------------------------------------------##

## Clean parameter formatting:
def qfmt(runid, quad, pars):
    #outfmt = ' %12.9f'*4 + ' %16.9f'*2
    outfmt = ' %16.9f'*6
    valtxt = outfmt % tuple(pars)
    return '%6s  %2s  %s' % (runid, quad, valtxt)

def qqqfmt(prefix, txforms):
    #outfmt = ' %12.9f'*4 + ' %16.9f'*2
    #outfmt = ' %16.9f'*6
    return '\n'.join([qfmt(prefix, *tt) for tt in txforms.items()])
    

## Print and save median:
outfile = 'txpars_med.txt'
with open(outfile, 'w') as fff:
    #lines = '\n'.join([qfmt('median', *stuff) for stuff in medtxform.items()])
    lines = qqqfmt('median', medtxform)
    sys.stderr.write(lines + '\n')
    fff.write(lines + '\n')

## Print and save by QRUNID:
outfile = 'txpars_all.txt'
with open(outfile, 'w') as fff:
    fff.write('\n'.join([qqqfmt(*qqtt) \
            for qqtt in total_result.items()])+'\n')

sys.exit(0)

######################################################################
# CHANGELOG (28_fit_sensor_transforms.py):
#---------------------------------------------------------------------
#
#  2026-04-07:
#     -- Increased __version__ to 0.1.0.
#     -- First created 28_fit_sensor_transforms.py.
#
