#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Fine-tune the constant-within-RUNID coordinate solution parameters
# using the Gaia-matched star catalogs and best-fit solutions of individual
# images. 
#
# This script can optionally restrict itself to J-only, H2-only, or load
# images from both filters for a joint solution.
#
# Rob Siverd
# Created:       2026-02-10
# Last modified: 2026-02-10
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
import argparse
#import shutil
#import resource
#import signal
import glob
import gc
import os
import sys
import time
import copy
import pprint
import pickle
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
#np.set_printoptions(suppress=True, linewidth=160)
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
#import theil_sen as ts
#import window_filter as wf
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

## Helpers for this investigation:
import helpers
reload(helpers)


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
    Perform fine-tune solution on all coordinate solutions for a given
    RUNID using J, H2, or all images.
    
    Version: %s
    """ % __version__
    parser = MyParser(prog=prog_name, description=descr_txt,
                          formatter_class=argparse.RawTextHelpFormatter)
    # ------------------------------------------------------------------
    #parser.set_defaults(thing1='value1', thing2='value2')
    parser.set_defaults(ignoreJ=False, ignoreH2=False)
    # ------------------------------------------------------------------
    #parser.add_argument('firstpos', help='first positional argument')
    #parser.add_argument('-w', '--whatever', required=False, default=5.0,
    #        help='some option with default [def: %(default)s]', type=float)
    #parser.add_argument('-o', '--output_file', 
    #        default='observations.csv', help='Output filename.')
    #parser.add_argument('remainder', help='other stuff', nargs='*')
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    iogroup = parser.add_argument_group('File I/O')
    #iogroup.add_argument('-o', '--output_file', default=None, required=True,
    #        help='Output filename', type=str)
    iogroup.add_argument('-R', '--runid', default=None, required=True,
            help='RUNID to process', type=str)
    bpgroup = iogroup.add_mutually_exclusive_group()
    bpgroup.add_argument('--Jonly', required=False, dest='ignoreH2', 
            action='store_true', help='fit J images only (exclude H2)')
    bpgroup.add_argument('--H2only', required=False, dest='ignoreJ',
            action='store_true', help='fit H2 images only (exclude J)')
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

## Bail if insane:
if context.ignoreJ and context.ignoreH2:
    sys.stderr.write("SHOULD NOT BE POSSIBLE ...\n")
    sys.exit(1)

## Filters we keep:
use_filters = ['J', 'H2']
if context.ignoreJ:
    use_filters.remove('J')
if context.ignoreH2:
    use_filters.remove('H2')

## Filter checker:
def filter_is_ok(filename, filt_list):
    fbase = os.path.basename(filename)
    return any([fbase.startswith('wircam_'+x) for x in filt_list])

## Tokenize the name:
def iname_from_filename(filename):
    return os.path.basename(filename).split('.')[0]

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


## Make sure input folder exists:
runid_dir = 'results/%s' % context.runid
if not os.path.isdir(runid_dir):
    sys.stderr.write("Error: folder not found: %s\n" % runid_dir)
    sys.exit(1)

## Look for pickled data:
#pickles = sorted(glob.glob('%s/wircam*.pickle' % runid_dir))
#pickles_H2 = sorted(glob.glob('%s/wircam_H2*.pickle' % runid_dir))
#pickles_J  = sorted(glob.glob('%s/wircam_J*.pickle' % runid_dir))
have_pickles = {}
have_pickles['H2'] = sorted(glob.glob('%s/wircam_H2*.pickle' % runid_dir))
have_pickles[ 'J'] = sorted(glob.glob('%s/wircam_J*.pickle' % runid_dir))

## Use pickles that match the filter list:
#use_pickles = []
#for fff in use_filters:
#    use_pickles += have_pickles[fff]
keep_pickles = sorted(itt.chain(*[have_pickles[x] for x in use_filters]))

## Keep subset:
#use_pickles = [x for x in pickles if filter_is_ok(x, filters)]
iname_order = [iname_from_filename(x) for x in keep_pickles]
#use_pickles = {iname_from_filename(x):x for x in keep_pickles}
use_pickles = dict(zip(iname_order, keep_pickles))
sys.stderr.write("Identified %d pickled data products.\n" % len(iname_order))


##--------------------------------------------------------------------------##
## Load all images:
all_data = {}
all_gstr = {}
all_pars = {}
for pbase,ppath in use_pickles.items():
    _gstr, _data, _pars = load_pickled_object(ppath)
    #_data, _pars = payload
    all_gstr[pbase] = _gstr
    all_data[pbase] = _data
    all_pars[pbase] = _pars

    ## also split by sensor:
    #qgroups = _data.groupby('qq')
    #imq_data[pbase] = {qq:subset.copy() for qq,subset in qgroups}
    pass

##--------------------------------------------------------------------------##
## Iterate over parameters and separate into CD matrices and CRPIX:
every_cdmat = {qq:[] for qq in quads}
every_crpix = {qq:[] for qq in quads}
every_crval = {}
for pbase,pars in all_pars.items():
    spars = spt.sift_params(pars)
    _cdms = spars['cdmat']
    _crpx = spars['crpix']
    for qq in quads:
        every_cdmat[qq].append(_cdms[qq])
        every_crpix[qq].append(_crpx[qq])
    every_crval[pbase] = spars['crval']
    pass

## Average and stddev of CRPIX values (not super):
avg_cdm, std_cdm = {}, {}
avg_crp, std_crp = {}, {}
for qq in quads:
    cdm_stack = np.array(every_cdmat[qq])
    crp_stack = np.array(every_crpix[qq])
    avg_cdm[qq] = np.average(cdm_stack, axis=0)
    avg_crp[qq] = np.average(crp_stack, axis=0)
    std_cdm[qq] = np.std(cdm_stack, axis=0)
    std_crp[qq] = np.std(crp_stack, axis=0)

## Note for reference that a single-image parameter bundle includes:
## * 4 CD matrix values for each sensor
## * 2 CRPIXn values for each sensor
## * 2 CRVALn values, shared among sensors

## Make a parameter bundle that covers the group. This will include:
## * a single set of CD matrix values for each sensor (4 quad X 4 pars)
## * a single set of CRPIXn values for each sensor (4 quad X 2 pars)
## * one pair of CRVALn params per image (N images x 2 pars)

## We will adopt:
## * the parameter-wise average of the CD matrix stack for the bundle
## * the parameter-wise average of the CRPIXn stack for the bundle
## * the best-fit CRVALn from each image for the bundle
## UPSHOT: the only real difference between a single-image parameter
## bundle and a multi-image bundle in my scheme is the number of CRVALs

## Using sift/unsift methods should make life easier. Because the number of
## CD matrix and CRPIX parameters is fixed, they should come first in the
## unsifted parameter array.

## Build a sifted parameter dictionary directly from our lists/dicts:
tmp_sifted = {
            'cdmat'   : copy.deepcopy(avg_cdm),
            'crpix'   : copy.deepcopy(avg_crp),
            'crval'   : copy.deepcopy(every_crval),
        }

## Enforce 21AQ18 best guess:
tmp_sifted['cdmat'] = {
        'NE': np.array([-0.00008508,  0.00000052,  0.00000052,  0.00008508]),
        'NW': np.array([-0.00008509,  0.00000071,  0.00000071,  0.0000851 ]),
        'SE': np.array([-0.00008509,  0.00000064,  0.00000063,  0.00008508]),
        'SW': np.array([-0.00008506,  0.00000073,  0.00000073,  0.00008504])}
tmp_sifted['crpix'] = {
        'NE': np.array([2124.0630655 ,  -91.96674513]),
        'NW': np.array([ -59.94684968,  -83.17030954]),
        'SE': np.array([2127.98554353, 2100.62029866]),
        'SW': np.array([ -62.49889049, 2113.12048686])}


### Flatten the parameter list:
#def unsift_params_multi(sifted):
#    parvec = []
#    for qq in quads:
#        parvec.extend(tmp_sifted['cdmat'][qq])
#        parvec.extend(tmp_sifted['crpix'][qq])
#    for iname in iname_order:
#        parvec.extend(every_crval[iname])
#    return np.array(parvec)
#
### Organize the parameter list:
#def sift_params_multi(params):
#    parsleft = params.copy()
#    cdmcrpix = parsleft[:24].reshape(-1, 6)
#    parsleft = parsleft[24:]
#    tmp_cdms = {qq:vv for qq,vv in zip(quads, cdmcrpix[:, :4])}
#    tmp_crpx = {qq:vv for qq,vv in zip(quads, cdmcrpix[:, 4:])}
#    #parsleft.reshape(-1, 2)
#    tmp_crvs = dict(zip(iname_order, parsleft.reshape(-1, 2)))
#    #tmp_crvs = dict(zip(iname_order, parsleft.reshape(-1, 2).tolist()))
#    return {'cdmat':tmp_cdms, 'crpix':tmp_crpx, 'crval':tmp_crvs}

flat_params = spt.unsift_params_multi(tmp_sifted, iname_order)


def multi_squared_residuals_foc2ccd_rdist(params, mdataset, imlist, diags=False,
                                          unsquared=False, snrweight=False):
    # parse parameters:
    msifted = spt.sift_params_multi(params, imlist)
    brute_cdmat = msifted['cdmat']
    brute_crpix = msifted['crpix']
    image_crval = msifted['crval']

    test_distmod = spt.guess_distmod

    # iterate over images:
    xres, yres = [], []
    diag_data = {}
    for iname in imlist:
        dataset = mdataset.get(iname)
        using_crval = image_crval.get(iname)
        #import pdb; pdb.set_trace()
        
        # note average star count for normalization:
        avg_nstars = np.average([len(x) for x in dataset.values()])

        # iterate over sensors:
        im_xres, im_yres = [], []
        im_diag = {}
        for qq,gst in dataset.items():
            nstar_scale_factor = np.sqrt(avg_nstars / float(len(gst)))
            tcpx1, tcpx2 = brute_crpix.get(qq)
            gxx, gyy = gst['x'], gst['y']
            cdmcrv = np.array(brute_cdmat.get(qq).tolist() + using_crval)
            test_xrel, test_yrel = helpers.inverse_tan_cdmcrv(cdmcrv,
                                    dataset[qq]['gra'], dataset[qq]['gde'])
            xnudge, ynudge = spt.calc_rdist_corrections(test_xrel, test_yrel, test_distmod)
            #import pdb; pdb.set_trace()
            test_xccd = test_xrel + xnudge + tcpx1
            test_yccd = test_yrel + ynudge + tcpx2
            x_error = test_xccd - gxx.values
            y_error = test_yccd - gyy.values
            scaled_xerr = x_error * nstar_scale_factor
            scaled_yerr = y_error * nstar_scale_factor
            if snrweight:
                scaled_xerr /= gst['realerr']
                scaled_yerr /= gst['realerr']
            im_xres.extend(scaled_xerr)
            im_yres.extend(scaled_yerr)
            if diags:
                im_diag[qq] = {     "gid":gst['gid'],
                                  "xmeas":gxx,
                                  "ymeas":gyy,
                                  "xcalc":test_xccd,
                                  "ycalc":test_yccd,
                                  'rdist':np.hypot(test_xrel, test_yrel),
                                 'xnudge':xnudge,
                                 'ynudge':ynudge,
                                 'xerror':x_error,
                                 'yerror':y_error,
                                 'rerror':np.hypot(x_error, y_error),
                                 'scaled_xerror':scaled_xerr,
                                 'scaled_yerror':scaled_yerr,
                                 'scaled_rerror':np.hypot(scaled_xerr, scaled_yerr),
                                   'flux':gst['flux'],
                                   'fwhm':gst['fwhm'],
                                  'flags':gst['flag'],
                                'dumbsnr':gst['dumbsnr'],
                                'realerr':gst['realerr'],
                                }

            pass    # end of loop over sensors

        xres.extend(im_xres)
        yres.extend(im_yres)
        diag_data[iname] = im_diag
        pass        # end of loop over IMAGES

    if diags:
        return diag_data
    if unsquared:
        return np.concatenate((xres, yres))
    return np.concatenate((xres, yres))**2
    
    return

def multi_fmin_squared_residuals_foc2ccd_rdist(params, mdataset, 
                                               imlist, **kwargs):
    return np.sum(multi_squared_residuals_foc2ccd_rdist(params, mdataset,
                                                    imlist, **kwargs))

#multi_squared_residuals_foc2ccd_rdist(flat_params, all_gstr, iname_order)

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##


## Attempt a solve with levmar ...
sys.stderr.write("Attempting a least squares minimization ...\n")
tik = time.time()
#slvkw = {'method':'trf', 'xtol':1e-14, 'ftol':1e-14}
#slvkw = {'method':'trf', 'xtol':1e-8, 'ftol':1e-8}
slvkw = {'method':'lm', 'xtol':1e-8, 'ftol':1e-8}
#slvkw = {'method':'trf', 'xtol':1e-8, 'ftol':1e-8, 'loss':'huber'}
slvkw.update({'max_nfev':10})
reskw = {'unsquared':True}
minimize_this = partial(multi_squared_residuals_foc2ccd_rdist, 
                        mdataset=all_gstr, imlist=iname_order)
answer = opti.least_squares(minimize_this, flat_params, kwargs=reskw, **slvkw)
sys.stderr.write("Ended up with: %s\n" % str(answer))
sys.stderr.write("Ended up with: %s\n" % str(answer['x']))
tok = time.time()
lsq_taken = tok - tik
sys.stderr.write("Solve took %.2f seconds.\n" % lsq_taken)

sys.stderr.write("%s\n%s\n" % (fulldiv, fulldiv))

## Crank out diag data at the solution:
yay_diags = multi_squared_residuals_foc2ccd_rdist(answer['x'], 
                          mdataset=all_gstr, imlist=iname_order, diags=True)


## Quickly estimate the typical error ...
runid_xerror, runid_yerror = {}, {}
runid_xpixel, runid_ypixel = {}, {}
for qq in spt._quads:
    tmp_xerror, tmp_yerror = [], []
    tmp_xpixel, tmp_ypixel = [], []
    for iname in iname_order:
        _data = yay_diags[iname]
        tmp_xerror.extend(_data[qq]['xerror'])
        tmp_yerror.extend(_data[qq]['yerror'])
        tmp_xpixel.extend(_data[qq]['xmeas'])
        tmp_ypixel.extend(_data[qq]['ymeas'])
    runid_xerror[qq] = np.array(tmp_xerror)
    runid_yerror[qq] = np.array(tmp_yerror)
    runid_xpixel[qq] = np.array(tmp_xpixel)
    runid_ypixel[qq] = np.array(tmp_ypixel)


sys.exit(0)
## Attempt a solve with fmin...
sys.stderr.write("\n\n%s\n%s\n" % (fulldiv, fulldiv))
sys.stderr.write("Try to minimize with fmin (could be slow) ....\n")
tik = time.time()
#fmkw = {'full_output':True, 'xtol':1e-14, 'ftol':1e-14}
fmkw = {'full_output':False} #, 'xtol':1e-14, 'ftol':1e-14}
shrink_this = partial(multi_fmin_squared_residuals_foc2ccd_rdist,
                      mdataset=all_gstr, imlist=iname_order,
                      unsquared=False, snrweight=False)
fanswer = opti.fmin(shrink_this, flat_params, **fmkw)
tok = time.time()
fmin_taken = tok - tik
sys.stderr.write("Solve took %.2f seconds.\n" % fmin_taken)

## Crank out diag data at the solution:
yay_fdiags = multi_squared_residuals_foc2ccd_rdist(answer['x'], 
                          mdataset=all_gstr, imlist=iname_order, diags=True)

## Quickly estimate the typical error ...
runid_xerror, runid_yerror = {}, {}
runid_xpixel, runid_ypixel = {}, {}
for qq in spt._quads:
    tmp_xerror, tmp_yerror = [], []
    tmp_xpixel, tmp_ypixel = [], []
    for iname in iname_order:
        _data = yay_fdiags[iname]
        tmp_xerror.extend(_data[qq]['xerror'])
        tmp_yerror.extend(_data[qq]['yerror'])
        tmp_xpixel.extend(_data[qq]['xmeas'])
        tmp_ypixel.extend(_data[qq]['ymeas'])
    runid_xerror[qq] = np.array(tmp_xerror)
    runid_yerror[qq] = np.array(tmp_yerror)
    runid_xpixel[qq] = np.array(tmp_xpixel)
    runid_ypixel[qq] = np.array(tmp_ypixel)

sys.exit(0)
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
fig_dims = (11, 9)
fig = plt.figure(1, figsize=fig_dims)
plt.gcf().clf()
#fig, axs = plt.subplots(2, 2, sharex=True, figsize=fig_dims, num=1, clear=True)
# sharex='col' | sharex='row'
#fig.frameon = False # disable figure frame drawing
#fig.subplots_adjust(left=0.07, right=0.95)
#ax1 = plt.subplot(gs[0, 0])
#ax1 = fig.add_subplot(111)
#ax1 = fig.add_subplot(111, polar=True)
#ax1 = fig.add_axes([0, 0, 1, 1])
#ax1.patch.set_facecolor((0.8, 0.8, 0.8))
#ax1.grid(True)
#ax1.axis('off')

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
#ax1.xaxis.set_major_formatter(fptformat) # re-format x ticks
#ax1.set_ylim(ax1.get_ylim()[::-1])
#ax1.set_xlabel('whatever', labelpad=30)  # push X label down 

#ax1.set_xticks([1.0, 3.0, 10.0, 30.0, 100.0])
#ax1.xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'])
#ax1.xticks([1, 2, 3], ['Jan', 'Feb', 'Mar'], rotation=45)
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


######################################################################
# CHANGELOG (23_fine_tune_RUNID_WCS.py):
#---------------------------------------------------------------------
#
#  2026-02-10:
#     -- Increased __version__ to 0.0.1.
#     -- First created 23_fine_tune_RUNID_WCS.py.
#
