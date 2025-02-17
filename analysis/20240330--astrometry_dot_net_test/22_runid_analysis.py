#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Analyze individual runids using ast.net-based Gaia matching.
#
# Rob Siverd
# Created:       2024-08-15
# Last modified: 2024-08-15
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.0.1"

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
import glob
import gc
import os
import sys
import time
import math
import pickle
#import ephem
import numpy as np
from numpy.lib.recfunctions import append_fields
#import datetime as dt
#from dateutil import parser as dtp
#import scipy.linalg as sla
#import scipy.signal as ssig
#import scipy.ndimage as ndi
import scipy.optimize as opti
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
from functools import partial
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

## Tangent projection:
import tangent_proj as tp

## WIRCam polynomial routines:
try:
    import wircam_poly
    reload(wircam_poly)
    wcp = wircam_poly.WIRCamPoly()
except ImportError:
    logger.error("failed to import wircam_poly module!")
    sys.exit(1)

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
try:
    import robust_stats
    reload(robust_stats)
    rs = robust_stats
except ImportError:
    logger.error("module robust_stats not found!  Install and retry.")
    sys.stderr.write("\nError!  robust_stats module not found!\n"
           "Please install and try again ...\n\n")
    sys.exit(1)

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

## WCS defaults:
crpix1 = 2122.690779
crpix2 =  -81.678888
pscale =   0.30601957084155673

##--------------------------------------------------------------------------##

## Fitting procedure:
def calc_tan_radec(pscale, pa_deg, cv1, cv2, xrel, yrel):
    this_cdmat = tp.make_cdmat(pa_deg, pscale)
    return tp.xycd2radec(this_cdmat, xrel, yrel, cv1, cv2)

def eval_tan_params(pscale, pa_deg, cv1, cv2, xrel, yrel, true_ra, true_de, expo=1):
    calc_ra, calc_de = calc_tan_radec(pscale, pa_deg, cv1, cv2, xrel, yrel)
    deltas = angle.dAngSep(calc_ra, calc_de, true_ra, true_de)
    return np.sum(deltas**expo)

def evaluator_pacrv(pacrv, pscale, xrel, yrel, true_ra, true_de, expo=1):
    pa_deg, cv1, cv2 = pacrv
    return eval_tan_params(pscale, pa_deg, cv1, cv2,
            xrel, yrel, true_ra, true_de, expo=expo)

def eval_cdmcrv(cdm_crv, xrel, yrel):
    this_cdmat = cdm_crv[:4]
    cv1, cv2   = cdm_crv[4:]
    return tp.xycd2radec(this_cdmat, xrel, yrel, cv1, cv2)
    #tra, tde   = tp.xycd2radec(this_cdmat, xrel, yrel, cv1, cv2)
    #return (tra % 360.0, tde)

def evaluator_cdmcrv(cdm_crv, xrel, yrel, true_ra, true_de, expo=1):
    calc_ra, calc_de = eval_cdmcrv(cdm_crv, xrel, yrel)
    #this_cdmat = cdm_crv[:4]
    #cv1, cv2   = cdm_crv[4:]
    #calc_ra, calc_de = tp.xycd2radec(this_cdmat, xrel, yrel, cv1, cv2)
    deltas = angle.dAngSep(calc_ra, calc_de, true_ra, true_de)
    return np.sum(deltas**expo)

def ls_evaluator_cdmcrv(cdm_crv, xrel, yrel, true_ra, true_de):
    calc_ra, calc_de = eval_cdmcrv(cdm_crv, xrel, yrel)
    ra_diff = np.mod(true_ra - calc_ra - 180.0, 360.0) - 180.0
    de_diff = true_de - calc_de
    return np.concatenate((ra_diff * np.cos(np.radians(true_de)), de_diff))


## Fitted parameters:
## [cd11, cd12, cd21, cd22, crval1, crval2]

## Joint eval routine:
#def joint_eval_cdmcrv(

## Joint RUNID fitter:
def joint_runid_evaluator_cdmcrv(cdm_crv, xrel_by_img, yrel_by_img, 
        true_ra_by_img, true_de_by_img):
    this_cdmat = cdm_crv[:4]
    crv_by_img = cdm_crv[4:].reshape(-1, 2)

    res = []
    for ipars,xx,yy,gra,gde in zip(crv_by_img, xrel_by_img, yrel_by_img, 
            true_ra_by_img, true_de_by_img):
        cdmcrv = np.concatenate((this_cdmat, ipars))
        res.append(ls_evaluator_cdmcrv(cdmcrv, xx, yy, gra, gde))

    return np.concatenate(res)

## Joint RUNID fitter (faster?):
def tjoint_runid_evaluator_cdmcrv(cdm_crv, xrel_by_img, yrel_by_img, 
        true_ra_by_img, true_de_by_img):
    this_cdmat = cdm_crv[:4]
    crv_by_img = cdm_crv[4:].reshape(-1, 2)

    res = []
    for ipars,xx,yy,gra,gde in zip(crv_by_img, xrel_by_img, yrel_by_img, 
            true_ra_by_img, true_de_by_img):
        cdmcrv = np.concatenate((this_cdmat, ipars))
        res.append(ls_evaluator_cdmcrv(cdmcrv, xx, yy, gra, gde))

    return np.concatenate(res)



## -----------------------------------------------------------------------

# Analyze CD matrix from header:
_cd_keys = ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')
def get_cdmatrix_pa_scale(header):
    orig_cdm = np.array([header[x] for x in _cd_keys]).reshape(2, 2)
    cd_xyscl = np.sqrt(np.sum(orig_cdm**2, axis=1))
    norm_cdm = orig_cdm / cd_xyscl
    norm_rot = np.dot(tp.xflip_mat, norm_cdm)
    flat_rot = norm_rot.flatten()
    pa_guess = [math.acos(flat_rot[0]), -math.asin(flat_rot[1]),
                        math.asin(flat_rot[2]), math.acos(flat_rot[3])]
    pos_ang  = np.degrees(np.average(pa_guess))
    pixscale = np.average(cd_xyscl)
    return pos_ang, pixscale

##--------------------------------------------------------------------------##


## Grab the ibase from a filename:
def ibase_from_filename(fits_path):
    return os.path.basename(fits_path).split('.')[0]


##--------------------------------------------------------------------------##
## Where to look for catalogs with Gaia matches:
load_root = 'matched'
#load_root = 'matched_v2'
if not os.path.isdir(load_root):
    sys.stderr.write("Folder not found: %s\n" % load_root)
    sys.stderr.write("Run 21_gaia_matching.py first ...\n")
    sys.exit(1)

## Where to save catalogs with recalculated RA/DE:
save_root = 'jointupd'
if not os.path.isdir(save_root):
    os.mkdir(save_root)

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

#use_runids = ['11AQ15']
##--------------------------------------------------------------------------##
dist_mod = 'dl12'
_xdw_key = 'xdw_dl12'
_ydw_key = 'ydw_dl12'
_wxdw_key = 'wxdw_dl12'
_wydw_key = 'wydw_dl12'
_crval_keys = ['CRVAL1', 'CRVAL2']
_param_keys = list(_cd_keys) + _crval_keys

model_dir = 'models'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

def dump_wcspars(filename, pars, delim=','):
    cols = ['cd11', 'cd12', 'cd21', 'cd22', 'crval1', 'crval2']
    with open(filename, 'w') as fff:
        fff.write(delim.join(cols) + '\n')
        fff.write(','.join([str(x) for x in pars]) + '\n')


fit_results = {}
ra_residuals = {}
de_residuals = {}
big_results = {}  # jnt_full_result for each runid
jresid_stddev = {}
jresid_maddev = {}
jresid_points = {}
flux_cut = 1000.  # ~15th percentile
#flux_cut = 10000.  # ~15th percentile
for this_runid in use_runids:
    have_files = runid_files[this_runid]
    cats = {}
    srcs = {}
    cdxx = []
    crvs = []
    fmin_best_pars = []
    ls_best_pars   = []
    save_ra_res = []
    save_de_res = []
    every_guess = []
    by_img_gra  = []
    by_img_gde  = []
    by_img_xrel = []
    by_img_yrel = []
    by_img_xraw = []
    by_img_yraw = []
    by_img_filt = []
    by_res_filt = []
    by_res_size = []
    by_res_rade = []  # whether residual is RA or DE
    by_res_isrc = []  # source ibase
    by_res_indx = []  # index in detections catalog
    jresid_stddev[this_runid] = {}
    jresid_maddev[this_runid] = {}
    jresid_points[this_runid] = {}

    # ensure output folder for joint-updated catalogs:
    save_jdir = os.path.join(save_root, this_runid)
    if not os.path.isdir(save_jdir):
        os.mkdir(save_jdir)

    #jresid_src
    for ii,this_fcat in enumerate(have_files, 1):
        ibase = ibase_from_filename(this_fcat)
        sys.stderr.write("Loading %s ... " % this_fcat)
        ccc.load_from_fits(this_fcat)
        stars = ccc.get_catalog()
        cats[ibase] = stars
        sys.stderr.write("done.\n")
        which_gaia = (stars['gaia_id'] > 0)
        brightish  = (stars['flux'] > flux_cut)
        matches    = stars[which_gaia & brightish]
        srcs[ibase] = matches

        # Header mining:
        header = ccc.get_header()
        #cdxx.append([header[x] for x in _cd_keys])
        guess = np.array([header[x] for x in _param_keys])
        cdxx.append(guess[:4])
        every_guess.append(guess)
        crvs.append(guess[4:])

        # shortcut notations:
        gaia_ra, xrel = matches['gaia_ra'], matches[_xdw_key] - crpix1
        gaia_de, yrel = matches['gaia_de'], matches[_ydw_key] - crpix2
        #by_img_gra.append(gaia_ra)
        #by_img_gde.append(gaia_de)
        #by_img_xrel.append(xrel)
        #by_img_yrel.append(yrel)
        ##by_img_xraw.append(matches['x'])
        ##by_img_yraw.append(matches['y'])
        by_img_filt.append(matches['filter'])
        by_res_filt.append(matches['filter'])   # once for RA
        by_res_filt.append(matches['filter'])   # again for DE
        fakeFWHM = 2*np.sqrt(matches['a'] * matches['b'])
        by_res_size.append(fakeFWHM)
        by_res_rade.append(['ra' for x in gaia_ra])
        by_res_rade.append(['de' for x in gaia_ra])
        by_res_isrc.append([ibase for x in gaia_ra])    # once for RA
        by_res_isrc.append([ibase for x in gaia_ra])    # again for DE
        by_res_indx.append(np.arange(len(gaia_ra)))     # once for RA
        by_res_indx.append(np.arange(len(gaia_ra)))     # again for DE

        ## fmin solution, absolute residuals:
        #minimize_this = partial(evaluator_cdmcrv, expo=1,
        ##minimize_this = partial(ls_evaluator_cdmcrv, 
        #        true_ra=gaia_ra, true_de=gaia_de, xrel=xrel, yrel=yrel)
        #tik = time.time()
        #fmin_answer = opti.fmin(minimize_this, guess, xtol=1e-6)
        #tok = time.time()
        #tsolve = tok - tik
        #sys.stderr.write("fmin solved in %.4f seconds\n" % tsolve)
        #fmin_best_pars.append(fmin_answer)
        #fmin_best_ra, fmin_best_de = eval_cdmcrv(fmin_answer, xrel, yrel)
        #fmin_best_ra = fmin_best_ra % 360.0
        #fmin_ra_resid = (fmin_best_ra - gaia_ra) * 3600. * np.cos(np.radians(gaia_de))
        #fmin_de_resid = (fmin_best_de - gaia_de) * 3600.
        #fmin_tot_resid = np.hypot(fmin_ra_resid, fmin_de_resid)

        # least-squares solution, signed residuals:
        minimize_this = partial(ls_evaluator_cdmcrv, 
                true_ra=gaia_ra, true_de=gaia_de, xrel=xrel, yrel=yrel)
        tik = time.time()
        ls_full_result = opti.least_squares(minimize_this, guess)
        ls_answer = ls_full_result['x']
        ls_best_pars.append(ls_answer)
        tok = time.time()
        tsolve = tok - tik
        sys.stderr.write("least_squares solved in %.4f seconds\n" % tsolve)
        ls_best_ra, ls_best_de = eval_cdmcrv(ls_answer, xrel, yrel)
        ls_best_ra = ls_best_ra % 360.0
        ls_ra_resid = (ls_best_ra - gaia_ra) * 3600.0 * np.cos(np.radians(gaia_de))
        ls_de_resid = (ls_best_de - gaia_de) * 3600.0
        ls_tot_resid = np.hypot(ls_ra_resid, ls_de_resid)

        # select inliers and refit:
        #sys.stderr.write("pass1 len(gaia_ra): %d\n" % len(gaia_ra))
        useful = rs.pick_inliers(ls_tot_resid, 4)
        gaia_ra, gaia_de = gaia_ra[useful], gaia_de[useful]
        #sys.stderr.write("pass2 len(gaia_ra): %d\n" % len(gaia_ra))
        xrel, yrel = xrel[useful], yrel[useful]
        minimize_this = partial(ls_evaluator_cdmcrv, 
                true_ra=gaia_ra, true_de=gaia_de, xrel=xrel, yrel=yrel)
        tik = time.time()
        ls_full_result = opti.least_squares(minimize_this, ls_answer)
        ls_answer = ls_full_result['x']
        ls_best_pars.append(ls_answer)
        tok = time.time()
        tsolve = tok - tik
        sys.stderr.write("least_squares solved in %.4f seconds (pass 2)\n" % tsolve)
        ls_best_ra, ls_best_de = eval_cdmcrv(ls_answer, xrel, yrel)
        ls_best_ra = ls_best_ra % 360.0
        ls_ra_resid = (ls_best_ra - gaia_ra) * 3600.0 * np.cos(np.radians(gaia_de))
        ls_de_resid = (ls_best_de - gaia_de) * 3600.0
        ls_tot_resid = np.hypot(ls_ra_resid, ls_de_resid)

        # Stick with the higher-quality data points:
        by_img_gra.append(gaia_ra)
        by_img_gde.append(gaia_de)
        by_img_xrel.append(xrel)
        by_img_yrel.append(yrel)
        #sys.exit(0)
        #if (ii > 444):
        #    break

    #break
    # Summary statistics:
    total_dets = sum([len(x) for x in cats.values()])
    match_dets = sum([len(x) for x in srcs.values()])
    match_pctg = 100.0 * match_dets / total_dets
    sys.stderr.write("Total dets: %d\n" % total_dets)
    sys.stderr.write("Match dets: %d\n" % match_dets)
    sys.stderr.write("Gaia match: %.2f%%\n" % match_pctg)


    #def joint_runid_evaluator_cdmcrv(cdm_crv, xrel_by_img, yrel_by_img,
    #    true_ra_by_img, true_de_by_img):

    fit_results[this_runid] = np.array(ls_best_pars)

    # Initial joint guess is the average CD matrix plus header CRVALs:
    every_guess = np.array(every_guess)
    #avg_cdm = np.average(cdxx, axis=0)
    im_crvs = np.concatenate(crvs)
    #avg_cdm     = np.average(every_guess)

    # Joint initial guess:
    avg_cdm = np.average(ls_best_pars, axis=0)[:4]
    ls_crvs = np.array(ls_best_pars)[:, 4:].flatten()
    j_guess = np.concatenate((avg_cdm, ls_crvs))

    
    joint_runid_evaluator_cdmcrv(j_guess, by_img_xrel, by_img_yrel,
            by_img_gra, by_img_gde)

    jnt_minimize_this = partial(joint_runid_evaluator_cdmcrv,
            xrel_by_img=by_img_xrel, yrel_by_img=by_img_yrel,
            true_ra_by_img=by_img_gra, true_de_by_img=by_img_gde)

    sys.stderr.write("Big, nasty, joint fit ...\n")
    tik = time.time()
    jnt_full_result = opti.least_squares(jnt_minimize_this, j_guess)
    jnt_answer = jnt_full_result['x']
    jnt_resids = jnt_minimize_this(jnt_answer)
    #jnt_best_pars.append(ls_answer)
    tok = time.time()
    tsolve = tok - tik
    sys.stderr.write("least_squares solved in %.4f seconds\n" % tsolve)
    jresid_stddev[this_runid]['all'] = np.std(jnt_resids)
    jnt_resid_med, jnt_resid_iqrn = rs.calc_ls_med_MAD(jnt_resids)
    jresid_maddev[this_runid]['all'] = jnt_resid_iqrn
    jresid_points[this_runid]['all'] = len(jnt_resids)

    #big_results[this_runid] = jnt_full_result

    # ----------------------------------------------------------------------- 
    # Recalculate positions for input catalogs:
    jnt_cdm = jnt_answer[:4]
    jnt_crvals = jnt_answer[4:].reshape(-1, 2)

    # Ensure output models dir for this runid:
    runid_model_dir = os.path.join(model_dir, this_runid)
    if not os.path.isdir(runid_model_dir):
        os.mkdir(runid_model_dir)

    for idx,this_fcat in enumerate(have_files, 0):
        sys.stderr.write("Loading %s ... " % this_fcat)
        #fbase = os.path.basename(this_fcat)
        #save_fcat = os.path.join(save_root, this_runid, fbase)
        fbase = os.path.basename(this_fcat)
        save_fcat = os.path.join(save_jdir, fbase)
        ccc.load_from_fits(this_fcat)
        stars = ccc.get_catalog()

        save_mod = os.path.join(runid_model_dir, fbase + '.txt')
        sys.stderr.write("%s ... " % save_mod)


        astpars = jnt_cdm.tolist() + jnt_crvals[idx].tolist()
        dump_wcspars(save_mod, astpars)
        #sys.stderr.write("\n")
        # unwindowed positions:
        xrel = stars[_xdw_key] - crpix1
        yrel = stars[_ydw_key] - crpix2
        jntupd_ra, jntupd_de = eval_cdmcrv(astpars, xrel, yrel)
        jntupd_ra = jntupd_ra % 360.0
        # windowed positions:
        wxrel = stars[_wxdw_key] - crpix1
        wyrel = stars[_wydw_key] - crpix2
        wjntupd_ra, wjntupd_de = eval_cdmcrv(astpars, wxrel, wyrel)
        wjntupd_ra = wjntupd_ra % 360.0
        # augment catalog and save:
        newcat = stars.copy()
        newcat = append_fields(newcat,
                ('jntupd_ra', 'jntupd_de', 'wjntupd_ra', 'wjntupd_de'),
                (jntupd_ra, jntupd_de, wjntupd_ra, wjntupd_de),
                usemask=False)
        ccc.set_catalog(newcat)
        ccc.save_as_fits(save_fcat, overwrite=True)
        #break
        pass

    continue

    ### ----------------------------------------------------------------------- 
    ### Breakout by filter:
    ##jresid_rade = np.concatenate(by_res_rade)
    ##jresid_indx = np.concatenate(by_res_indx)
    ##jresid_isrc = np.concatenate(by_res_isrc)
    ##jresid_filt = np.concatenate(by_res_filt)
    ##is_Hband    = jresid_filt == 'H2'
    ##is_Jband    = jresid_filt == 'J'
    ##jnt_resids_H = jnt_resids[is_Hband]
    ##jnt_resids_J = jnt_resids[is_Jband]

    ### lookie catalog:
    ##is_outlier  = (jnt_resids * 3600.0) > 0.3
    ##which_Jbad  = is_outlier & is_Jband
    ##Jbad_ibase  = jresid_isrc[which_Jbad]
    ##Jbad_index  = jresid_indx[which_Jbad]
    ###examine_me  = []
    ###for ibase,idx in zip(Jbad_ibase, Jbad_index):
    ###    examine_me.append(np.atleast_1d(cats[ibase][idx]))
    ###examine_me  = np.concatenate(examine_me)

    ##jresid_stddev[this_runid]['H2'] = np.std(jnt_resids_H)
    ##jnt_resid_med_H, jnt_resid_iqrn_H = rs.calc_ls_med_MAD(jnt_resids_H)
    ##jresid_maddev[this_runid]['H2'] = jnt_resid_iqrn_H
    ##jresid_points[this_runid]['H2'] = len(jnt_resids_H)

    ##jresid_stddev[this_runid]['J'] = np.std(jnt_resids_J)
    ##jnt_resid_med_J, jnt_resid_iqrn_J = rs.calc_ls_med_MAD(jnt_resids_J)
    ##jresid_maddev[this_runid]['J'] = jnt_resid_iqrn_J
    ##jresid_points[this_runid]['J'] = len(jnt_resids_J)

    ###sys.exit(0)
    ##continue

    #break
    ###break
    ### nudge the xrel/yrel (adjust CRPIX) and see if solution improves:
    #rtracker = []
    #xnudge = np.arange(3) - 1.0
    #ynudge = np.arange(3) - 1.0
    ##xnudge = np.arange(4) + 2.0
    ##ynudge = np.arange(4) + 2.0
    #for dx in xnudge:
    #    for dy in ynudge:
    #        sys.stderr.write("dx,dy: %.1f,%.1f\n" % (dx, dy))
    #        # Update CRPIX1/2 in distortion module and recompute nudges:
    #        adjusted_crpix1 = crpix1 + dx
    #        adjusted_crpix2 = crpix2 + dy
    #        wcp._crpix1 = adjusted_crpix1
    #        wcp._crpix2 = adjusted_crpix2

    #        # Iterate over per-image raw X,Y and compute new relative X,Y:
    #        sys.stderr.write("Adjusting xrel/yrel ... ")
    #        nudged_img_xrel = []
    #        nudged_img_yrel = []
    #        for tx,ty in zip(by_img_xraw, by_img_yraw):
    #            xcorr, ycorr = wcp.calc_xy_nudges(tx, ty, dist_mod)
    #            nudged_img_xrel.append(tx + xcorr - adjusted_crpix1)
    #            nudged_img_yrel.append(ty + ycorr - adjusted_crpix2)
    #        sys.stderr.write("done.\n")

    #        #nudged_img_xrel = [x+dx for x in by_img_xrel]
    #        #nudged_img_yrel = [y+dy for y in by_img_yrel]
    #        sys.stderr.write("Solving after nudge ...\n")
    #        ndg_minimize_this = partial(joint_runid_evaluator_cdmcrv,
    #                xrel_by_img=nudged_img_xrel, yrel_by_img=nudged_img_yrel,
    #                true_ra_by_img=by_img_gra, true_de_by_img=by_img_gde)
    #        tik = time.time()
    #        ndg_full_result = opti.least_squares(ndg_minimize_this, jnt_answer)
    #        ndg_answer = ndg_full_result['x']
    #        ndg_resids = ndg_minimize_this(ndg_answer)
    #        rtracker.append((dx, dy, np.std(ndg_resids), ndg_resids))
    #        tok = time.time()
    #        tsolve = tok - tik
    #        sys.stderr.write("least_squares solved in %.4f seconds (%.1f, %.1f)\n"
    #                % (tsolve, dx, dy))
    #    #break



    #break
    #continue

    ## average CD matrix:
    ##avg_cdm = np.average(cdxx, axis=0)

    ## concatenate data arrays (Gaia-matched subset):
    #match_gra = np.concatenate([x[ 'gaia_ra'] for x in srcs.values()])
    #match_gde = np.concatenate([x[ 'gaia_de'] for x in srcs.values()])
    #match_xdw = np.concatenate([x['xdw_dl12'] for x in srcs.values()])
    #match_ydw = np.concatenate([x['ydw_dl12'] for x in srcs.values()])
    #match_vecs = {'true_ra' : match_gra,
    #              'true_de' : match_gde,
    #                 'xrel' : match_xdw,
    #                 'yrel' : match_ydw,}
    #minimize_this = partial(evaluator_cdmcrv, expo=1, **match_vecs)

## Stash my cumulative results for later:
save_file = 'big_results.pickle'
#stash_as_pickle(save_file, big_results)

# first = cats[sorted(cats.keys())[0]]
#anet_ra_all = np.concatenate([x['mean_anet_ra'] for x in cats.values()])
#anet_de_all = np.concatenate([x['mean_anet_de'] for x in cats.values()])
#gaia_ra_all = np.concatenate([x[     'gaia_ra'] for x in cats.values()])
#gaia_de_all = np.concatenate([x[     'gaia_de'] for x in cats.values()])
#gaia_keeper = ~np.isnan(gaia_ra_all)
#ra_diffs = (anet_ra_all - gaia_ra_all) * np.cos(np.radians(gaia_de_all))
#de_diffs = (anet_de_all - gaia_de_all)

##--------------------------------------------------------------------------##
## Quick ASCII I/O:
#data_file = 'data.txt'
#gftkw = {'encoding':None} if (_have_np_vers >= 1.14) else {}
#gftkw.update({'names':True, 'autostrip':True})
#gftkw.update({'delimiter':'|', 'comments':'%0%0%0%0'})
#gftkw.update({'loose':True, 'invalid_raise':False})
#all_data = np.genfromtxt(data_file, dtype=None, **gftkw)
#all_data = np.atleast_1d(np.genfromtxt(data_file, dtype=None, **gftkw))
#all_data = np.genfromtxt(fix_hashes(data_file), dtype=None, **gftkw)
#all_data = aia.read(data_file)

#all_data = append_fields(all_data, ('ra', 'de'), 
#         np.vstack((ra, de)), usemask=False)
#all_data = append_fields(all_data, cname, cdata, usemask=False)

#pdkwargs = {'skipinitialspace':True, 'low_memory':False}
#pdkwargs.update({'delim_whitespace':True, 'sep':'|', 'escapechar':'#'})
#all_data = pd.read_csv(data_file)
#all_data = pd.read_csv(data_file, **pdkwargs)
#all_data = pd.read_table(data_file)
#all_data = pd.read_table(data_file, **pdkwargs)
#nskip, cnames = analyze_header(data_file)
#all_data = pd.read_csv(data_file, names=cnames, skiprows=nskip, **pdkwargs)
#all_data = pd.DataFrame.from_records(npy_data)
#all_data = pd.DataFrame(all_data.byteswap().newbyteorder()) # for FITS tables

### Strip leading '#' from column names:
#def colfix(df):
#    df.rename(columns={kk:kk.lstrip('#') for kk in df.keys()}, inplace=True)
#colfix(all_data)

#all_data.rename(columns={'old_name':'new_name'}, inplace=True)
#all_data.reset_index()
#firstrow = all_data.iloc[0]
#for ii,row in all_data.iterrows():
#    pass

#vot_file = 'neato.xml'
#vot_data = av.parse_single_table(vot_file)
#vot_data = av.parse_single_table(vot_file).to_table()

##--------------------------------------------------------------------------##
## Timestamp modification:
#def time_warp(jdutc, jd_offset, scale):
#    return (jdutc - jd_offset) * scale

## Self-consistent time-modification for plotting:
#tfudge = partial(time_warp, jd_offset=tstart.jd, scale=24.0)    # relative hrs
#tfudge = partial(time_warp, jd_offset=tstart.jd, scale=1440.0)  # relative min

##--------------------------------------------------------------------------##
## Quick FITS I/O:
#data_file = 'image.fits'
#img_vals = pf.getdata(data_file)
#hdr_keys = pf.getheader(data_file)
#img_vals, hdr_keys = pf.getdata(data_file, header=True)
#img_vals, hdr_keys = pf.getdata(data_file, header=True, uint=True) # USHORT
#img_vals, hdr_keys = fitsio.read(data_file, header=True)

#date_obs = hdr_keys['DATE-OBS']
#site_lat = hdr_keys['LATITUDE']
#site_lon = hdr_keys['LONGITUD']

sys.stderr.write("Stop here, before plotting.\n")
sys.exit(0)

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
#ax1.set_theta_direction(-1)         # counterclockwise
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



######################################################################
# CHANGELOG (22_runid_analysis.py):
#---------------------------------------------------------------------
#
#  2024-08-15:
#     -- Increased __version__ to 0.0.1.
#     -- First created 22_runid_analysis.py.
#
