#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Vet Abby's master list of Gaia matches using astrometry.net-derived
# solutions (believed to be clean).
#
# Rob Siverd
# Created:       2024-09-05
# Last modified: 2024-09-05
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
#import pickle
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
#import matplotlib.pyplot as plt
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
#import tangent_proj as tp


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

## WCS defaults:
crpix1 = 2122.690779
crpix2 =  -81.678888
pscale =   0.30601957084155673

##--------------------------------------------------------------------------##

## Abby's final_matches CSV:
abby_csv = '/home/acolclas/final_matches_new.csv'
if not os.path.isfile(abby_csv):
    sys.stderr.write("File not found: %s\n" % abby_csv)
    sys.exit(1)

## Load the CSV file:
sys.stderr.write("Loading %s ... " % abby_csv)
tik = time.time()
pdkwargs = {'skipinitialspace':True, 'low_memory':False}
pdkwargs['index_col'] = 0
abby_data = pd.read_csv(abby_csv, **pdkwargs)
tok = time.time()
taken = tok - tik
sys.stderr.write("done. Took %.3f seconds.\n" % taken)


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
## Where to look for update catalogs:
save_root = 'matched'
if not os.path.isdir(save_root):
    sys.stderr.write("Folder not found: %s\n" % save_root)
    sys.stderr.write("Run 21_gaia_matching.py first ...\n")
    sys.exit(1)

## Get a list of runid folders and RUNIDs themselves:
runids_list = sorted(glob.glob('%s/??????' % save_root))
runid_dirs  = {os.path.basename(x):x for x in runids_list}
runid_files = {kk:sorted(glob.glob('%s/wir*fcat'%dd)) \
                            for kk,dd in runid_dirs.items()}

## Append ibase column to Abby's dataset:
abby_data['ibase'] = [ibase_from_filename(x) for x in abby_data['Image Name']]

## Pick one for now:
use_runids = ['12BQ01']
use_runids = ['12BQ01', '14AQ08']
#use_runids = ['14AQ08']
#use_runids = sorted(runid_dirs.keys())
use_runids = sorted(list(set(abby_data['RunID'])))

##--------------------------------------------------------------------------##
dist_mod = 'dl12'
_xdw_key = 'xdw_dl12'
_ydw_key = 'ydw_dl12'
_crval_keys = ['CRVAL1', 'CRVAL2']
_param_keys = list(_cd_keys) + _crval_keys


boguses = {}
missing = {}
for this_runid in use_runids:
    have_files = runid_files[this_runid]
    #cats = {}
    srcs = {}
    cdxx = []
    crvs = []
    #every_guess = []
    #by_img_gra  = []
    #by_img_gde  = []
    #by_img_xrel = []
    #by_img_yrel = []
    #by_img_xraw = []
    #by_img_yraw = []
    abby_subset = abby_data[abby_data['RunID'].isin([this_runid])]
    for ii,this_fcat in enumerate(have_files, 1):
        ibase = ibase_from_filename(this_fcat)
        sys.stderr.write("Loading %s ... " % this_fcat)
        ccc.load_from_fits(this_fcat)
        stars = ccc.get_catalog()
        #cats[ibase] = stars
        sys.stderr.write("done.\n")
        which_gaia = (stars['gaia_id'] > 0)
        matches    = stars[which_gaia]
        srcs[ibase] = matches
        ai_subset  = abby_subset[abby_subset.ibase == ibase]
        nsrc_abby  = len(ai_subset)
        sys.stderr.write("For %s ...\n" % ibase)
        sys.stderr.write("abby has %d entries\n" % nsrc_abby)
        sys.stderr.write("anet has %d entries\n" % len(matches))
        if nsrc_abby < 1:
            sys.stderr.write("SKIP!\n")
            continue
        ai_strxy = set(['%.0f_%.0f'%x for x in \
                zip(100*ai_subset['X Pixel'], 100*ai_subset['Y Pixel'])])
        an_strxy = set(['%.0f_%.0f'%x for x in zip(100*matches['x'], 100*matches['y'])])

        abby_missing = an_strxy - ai_strxy
        abby_boguses = ai_strxy - an_strxy
        sys.stderr.write("len(abby_missing): %d\n" % len(abby_missing))
        sys.stderr.write("len(abby_boguses): %d\n" % len(abby_boguses))
        #ai_xyset = set(zip(ai_subset['X Pixel'], ai_subset['Y Pixel']))
        #an_xyset = set(zip(matches
        missfrac = float(len(abby_missing)) / float(nsrc_abby)
        bogufrac = float(len(abby_boguses)) / float(nsrc_abby)
        missing[ibase] = (abby_missing, len(ai_subset), missfrac)
        boguses[ibase] = (abby_boguses, len(ai_subset), bogufrac)



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