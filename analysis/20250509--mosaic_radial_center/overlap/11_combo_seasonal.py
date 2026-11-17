#!/usr/bin/env python3
# vim: set fileencoding=utf-8 ts=4 sts=4 sw=4 et tw=80 :
#
# Put 'seasonal' multiframe calib1 and calib2 data on the same plot.
#
# Rob Siverd
# Created:       2026-10-20
# Last modified: 2026-10-20
#--------------------------------------------------------------------------
#**************************************************************************
#--------------------------------------------------------------------------

## Current version:
__version__ = "0.1.0"

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
#import math
import ast
#import io
import gc
import os
import sys
import time
#import pprint
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
#from functools import partial
#from collections import OrderedDict
#from collections.abc import Iterable
#import multiprocessing as mp
#np.set_printoptions(suppress=True, linewidth=160)
#import operator
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from statsmodels.regression.quantile_regression import QuantReg
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
#class Unbuffered(object):
#   def __init__(self, stream):
#       self.stream = stream
#   def write(self, data):
#       self.stream.write(data)
#       self.stream.flush()
#   def __getattr__(self, attr):
#       return getattr(self.stream, attr)
#
#sys.stdout = Unbuffered(sys.stdout)
#sys.stderr = Unbuffered(sys.stderr)

##--------------------------------------------------------------------------##

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

## Save object as string:
def stash_as_stringrep(filename, thing):
    with open(filename, 'w') as sasf:
        sasf.write(str(thing) + '\n')
        #pprint.pprint(thing, stream=sasf, width=120)  # pretty!
    return

## Stringified object load (ast):
def load_stringrep_object(filename):
    with open(filename, 'r') as srof: 
        return ast.literal_eval(srof.read())

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

## Extract RUNID from filename:
def runid_from_filename(filename):
    return os.path.basename(filename).split('_')[1]

## Load parameter set from file:
def load_parameters(filename):
    with open(filename, 'r') as fff:
        return ast.literal_eval(fff.read())

## Extract CD matrix and CRPIX from parameter list:
def get_cdm_crpix(parameters):
    return np.array(parameters[:24]).reshape(4, -1)

##--------------------------------------------------------------------------##
##------------------         CD Matrix / PA Breakout        ----------------##
##--------------------------------------------------------------------------##

## Breakdown from:
## https://arxiv.org/html/2602.04041v1
def analyze(cddata):
    cd11, cd12, cd21, cd22 = cddata.ravel()[:4]
    xscale = np.sqrt(cd11*cd11 + cd12*cd12)
    yscale = np.sqrt(cd21*cd21 + cd22*cd22)
    yangle = np.degrees(np.arctan2(cd12, cd22))
    xangle = np.degrees(np.arctan2(cd11, cd21))
    #yangle = np.degrees(np.arctan(cd12 / cd22))
    #xangle = np.degrees(np.arctan(cd11 / cd21))
    axskew = yangle - xangle - 90.0
    return xscale, yscale, yangle, axskew

##--------------------------------------------------------------------------##
##--------------------------------------------------------------------------##

## Where the parameter files live:
field_dir = { 'c1' : '../multiframe',
              'c2' : '../multiframe_calib2' }
#pars_dir = { 'c1' : '../multiframe/joint_pars',
#             'c2' : '../multiframe_calib2/joint_pars' }
pars_dir = {ff:dd+'/joint_pars' for ff,dd in field_dir.items()}

## List of available J-only joint parameter files (calib1):
par_flist = {ff:sorted(glob.glob(dd + '/jpars_??????_J.txt')) \
                                    for ff,dd in pars_dir.items()}

## Field keys for later use:
#fields = field_dir.keys()

##--------------------------------------------------------------------------##
##------------------       Load and Analyze Parameters      ----------------##
##--------------------------------------------------------------------------##


## RUNIDs and files:
raw_params, raw_inames = {}, {}
fpar_stack = {}
fne_pstack, fnw_pstack, fse_pstack, fsw_pstack = {}, {}, {}, {}
fne_crpix1, fnw_crpix1, fse_crpix1, fsw_crpix1 = {}, {}, {}, {}
fne_crpix2, fnw_crpix2, fse_crpix2, fsw_crpix2 = {}, {}, {}, {}
fne_cdinfo, fnw_cdinfo, fse_cdinfo, fsw_cdinfo = {}, {}, {}, {}
frunid_list = {}
for ff,pdir in pars_dir.items():
    par_flist = sorted(glob.glob(pdir + '/jpars_??????_J.txt'))
    runid_list = [runid_from_filename(x) for x in par_flist]
    par_files = dict(zip(runid_list, par_flist))
    t_params, t_inames = {}, {}
    for runid,fname in par_files.items():
        t_params[runid], t_inames[runid] = load_parameters(fname)
    t_stack = np.dstack([get_cdm_crpix(t_params[x]) for x in runid_list])
    fne_pstack[ff], fnw_pstack[ff], fse_pstack[ff], fsw_pstack[ff] = t_stack
    raw_params[ff] = t_params
    raw_inames[ff] = t_inames
    fpar_stack[ff] = t_stack
    frunid_list[ff] = runid_list

    # CRPIX breakout:
    fne_crpix1[ff], fne_crpix2[ff] = fne_pstack[ff][4:]
    fnw_crpix1[ff], fnw_crpix2[ff] = fnw_pstack[ff][4:]
    fse_crpix1[ff], fse_crpix2[ff] = fse_pstack[ff][4:]
    fsw_crpix1[ff], fsw_crpix2[ff] = fsw_pstack[ff][4:]

    # CD analysis:
    fne_cdinfo = np.array([analyze(x) for x in fne_pstack[ff][:4].T])
    fnw_cdinfo = np.array([analyze(x) for x in fnw_pstack[ff][:4].T])
    fse_cdinfo = np.array([analyze(x) for x in fse_pstack[ff][:4].T])
    fsw_cdinfo = np.array([analyze(x) for x in fsw_pstack[ff][:4].T])

##--------------------------------------------------------------------------##
##------------------         Load fcats and seasonal        ----------------##
##--------------------------------------------------------------------------##

## Coordinates:
cfht_latNdeg =   19.825252
cfht_lonEdeg = -155.468876

## Files:
fcat_base = 'fcat_paths.csv'
season_base = 'seasonal_data.csv'
#fcat_lists = {ff:'%s/%s'%(dd, fcat_list) for ff,dd in field_dir.items()}
#seasnp.average(rro['CRVAL1'])on_files = {ff:'%s/%s'%(dd, season_data) for ff,dd in field_dir.items()}

## Load fcat tables and seasonal data:
pdkwargs = {'skipinitialspace':True, 'low_memory':False}
cat_tables = {}
seasonals = {}
for ff,dd in field_dir.items():
    fcat_list = os.path.join(dd, fcat_base)
    season_data = os.path.join(dd, season_base)
    sys.stderr.write("Loading '%s' ... " % fcat_list)
    cat_tables[ff] = pd.read_csv(fcat_list, **pdkwargs)
    sys.stderr.write("done.\n")
    sys.stderr.write("Loading '%s' ... " % season_data)
    seasonals[ff] = pd.read_csv(season_data, **pdkwargs)
    sys.stderr.write("done.\n")

## An array with all of the JDTDB values in it:
every_jdtdb = np.concatenate([x['JDTDB'].values for x in seasonals.values()])

##--------------------------------------------------------------------------##
##------------------         Seasonal Helper Routines       ----------------##
##--------------------------------------------------------------------------##

twopi = 2.0 * np.pi

## Convert HHMMSS.SS to degrees:
def hms2deg(hmstxt):
    th, tm, ts = hmstxt.split(':')
    is_neg = hmstxt.startswith('-')
    mm = float(ts) / 60.0 + float(tm)
    hh = float(th) - mm/60. if is_neg else float(th) + mm/60.
    return 15.0*hh

## RA/Dec to Cartesian:
def rade2xyz(rad_ra, rad_de):
    x = np.cos(rad_de) * np.cos(rad_ra)
    y = np.cos(rad_de) * np.sin(rad_ra)
    z = np.sin(rad_de)
    return np.vstack((x, y, z))

## Convert Cartesian points to RA, Dec:
def xyz2equ(xyz_pts):
    # Shape/dimension sanity check:
    if ((xyz_pts.ndim != 2) or (xyz_pts.shape[0] != 3)):
        sys.stderr.write("XYZ points have wrong shape!\n")
        return (0,0)
    tx = np.array(xyz_pts[0]).flatten()
    ty = np.array(xyz_pts[1]).flatten()
    tz = np.array(xyz_pts[2]).flatten()
    ra = np.arctan2(ty, tx)
    de = np.arcsin(tz)
    equ_coo = np.vstack((ra, de))
    return equ_coo

## Vector length:
def veclen(vec):
    #return np.sqrt(np.sum(vec*vec, axis=0))
    return np.sqrt(np.sum(vec*vec, axis=1))


##--------------------------------------------------------------------------##
##------------------       Deal with LST/HA and similar     ----------------##
##--------------------------------------------------------------------------##


## The following work because 
## - LST doesn't get close to 360
## - HA is +/- and doesn't wrap
for ff,seasonal in seasonals.items():
    # mid-image LST:
    lst1 = np.array([hms2deg(x) for x in seasonal.SIDTIME])
    lst2 = np.array([hms2deg(x) for x in seasonal.LSTEND])
    lst_deg = 0.5 * (lst1 + lst2)

    # mid-iamge HA:
    ha1 = np.array([hms2deg(x) for x in seasonal.HA])
    ha2 = np.array([hms2deg(x) for x in seasonal.HAEND])
    ha_deg = 0.5 * (ha1 + ha2)

    # parallactic angle:
    r_dec = np.radians(seasonal.CRVAL2)
    numer = np.sin(np.radians(ha_deg))
    denom = np.tan(np.radians(cfht_latNdeg)) * np.cos(r_dec) \
            - np.sin(r_dec) * np.cos(np.radians(ha_deg))
    parang_rad = np.arctan2(numer, denom) % twopi
    parang_deg = np.degrees(parang_rad)
    seasonal['PARANG'] = parang_deg

##--------------------------------------------------------------------------##
##------------------         Mean/stddev of parameters      ----------------##
##--------------------------------------------------------------------------##

## Compute per-RUNID averages and stddevs for various seasonal quantities:
## 'MCTR_RA', 'MCTR_DEC'
statcols = ['JDTDB', 'AIRMASS', 'TELALT', 'TELAZ', 'CRVAL1', 'CRVAL2',
            'OBS_X', 'OBS_Y', 'OBS_Z', 'OBS_VX', 'OBS_VY', 'OBS_VZ', 'PARANG']

frstats = {}
maxdursec = 3600.0
for ff,seasonal in seasonals.items():
    rstats = {}
    for runid,imlist in raw_inames[ff].items():
        subset = seasonal[seasonal.qrunid == runid]
        dursec = np.ptp(subset.JDTDB)*86400.
        #sys.stderr.write("%s %s lasted %.1f sec\n" % (ff, runid, dursec))
        things = {}
        for col in statcols:
            things[col] = (np.average(subset[col]), np.std(subset[col]))
        things['dursec'] = (dursec, dursec)
        rstats[runid] = things
        pass
    frstats[ff] = rstats

## Reorganize stat averages into arrays ordered by runid:
statcols.append('dursec')
frstats_runid_order = {}
for ff,rstats in frstats.items():
    frstats_runid_order[ff] = \
            {ss:np.array([rstats[rr][ss][0] for rr in frunid_list[ff]]) \
                            for ss in statcols}
    pass

##--------------------------------------------------------------------------##
##------------------         Point Orders for Plotting      ----------------##
##--------------------------------------------------------------------------##

## Orderings for fractional year, az/alt, parang:
j2k_epoch = 2451545.0
frst_yfrac, fyfr_order = {}, {}
fazm_order, falt_order = {}, {}
fsub_night, fpar_order = {}, {}
for ff,rro in frstats_runid_order.items():
    frst_yfrac[ff] = (rro['JDTDB'] - j2k_epoch) % 365.25
    fyfr_order[ff] = np.argsort(frst_yfrac[ff])   # fractional year
    fazm_order[ff] = np.argsort(rro['TELAZ'])     # telescope azimuth
    falt_order[ff] = np.argsort(rro['TELALT'])    # telescope azimuth
    fpar_order[ff] = np.argsort(rro['PARANG'])    # parallactic angle
    fsub_night[ff] = (rro['JDTDB'] - j2k_epoch - 10/24.) % 1.0
    #fsub_night[ff] = (rro['JDTDB'] + 10/24.

## Earth velocity ordering (perpendicular to aberration):
fvel_order, fvdot_vals = {}, {}
for ff,rro in frstats_runid_order.items():
    xyzpos = np.column_stack([rro[x] for x in ['OBS_X', 'OBS_Y', 'OBS_Z']])
    xyzvel = np.column_stack([rro[x] for x in ['OBS_VX', 'OBS_VY', 'OBS_VZ']])
    rtotal = np.sum(xyzpos**2, axis=1)**0.5
    vtotal = np.sum(xyzvel**2, axis=1)**0.5
    u_xyzvel = xyzvel / veclen(xyzvel)[:, None]	# velocity unit vector

    # dot product of target direction and Earth velocity:
    avg_ra_rad = np.radians(np.average(rro['CRVAL1']))
    avg_de_rad = np.radians(np.average(rro['CRVAL2']))
    target_xyz = np.squeeze(rade2xyz(avg_ra_rad, avg_de_rad))
    vdottarget = np.degrees(np.arccos([np.dot(target_xyz, x) for x in u_xyzvel]))
    fvdot_vals[ff] = vdottarget
    fvel_order[ff] = np.argsort(vdottarget)


##--------------------------------------------------------------------------##
##------------------         Plotting for the Win!          ----------------##
##--------------------------------------------------------------------------##

## Plot window config:
fig_dims = (11, 9)

## Plotting windows and axes. What they contain:
## * fg1 - reproduces fg1 of previous seasonal plots (CD11/CD12 vs all)
fg1, axs1 = plt.subplots(nrows=4, ncols=2, num=1, clear=True, figsize=fig_dims,
#fig, axs = plt.subplots(nrows=2, ncols=6, num=1, clear=True, figsize=fig_dims,
                        sharex=False, squeeze=True)
fg2, axs2 = plt.subplots(nrows=4, ncols=2, num=2, clear=True, figsize=fig_dims,
#fig, axs = plt.subplots(nrows=2, ncols=6, num=1, clear=True, figsize=fig_dims,
                        sharex=False, squeeze=True)

## Master fig/axis lists:
every_axs = [axs1, axs2]
every_fig = [ fg1,  fg2]

## Grid lines:
for axs in every_axs:
    for ax in axs.ravel():
        ax.grid(True)

##--------------------------------------------------------------------------##

## CD11 and CD22 vs:
## * day of year
## * vdot
## * par angle
for ff,rro in frstats_runid_order.items():
    #cdinfo = fne_cdinfo
    xscale, yscale, yangle, skew = fne_cdinfo.T
    pstack = fpar_stack.get(ff)
    cd11, cd22 = pstack[0,0,:], pstack[0,3,:]
    #cd11, cd22 = pstack[0,1,:], pstack[0,2,:]
    multi = rro['dursec'] > 3600.0  # these are multi-night
    which = ~np.isinf(cd11)         # select everything
    which = ~multi

    # day-of-year ordering
    order = fyfr_order.get(ff)[which]
    yfrac = frst_yfrac.get(ff)
    axs1[0,0].scatter(yfrac[order], cd11[order], label=ff)
    axs1[0,1].scatter(yfrac[order], cd22[order], label=ff)
    [axs1[0,x].set_xlabel("Day of Year") for x in (0,1)]

    # velocity ordering:
    vdata = fvdot_vals.get(ff)
    order = fvel_order.get(ff)[which]
    axs1[1,0].scatter(vdata[order], cd11[order], label=ff)
    axs1[1,1].scatter(vdata[order], cd22[order], label=ff)
    [axs1[1,x].set_xlabel("$v_E \cdot RA/DE$") for x in (0,1)]
 
    # parallactic angle:
    pdata = rro['PARANG']
    #pdata = rro['TELAZ']
    #pdata = rro['TELALT']
    order = np.argsort(pdata)[which]
    axs1[2,0].scatter(pdata[order], cd11[order], label=ff)
    axs1[2,1].scatter(pdata[order], cd22[order], label=ff)
    #[axs1[2,x].set_xlabel("TELALT (deg)") for x in (0,1)]
    #[axs1[2,x].set_xlabel("TELAZ (deg)") for x in (0,1)]
    [axs1[2,x].set_xlabel("PARANG (deg)") for x in (0,1)]

    ## by fractional night progress
    ndata = fsub_night.get(ff)
    #axs1[3,0].scatter(ndata[multi], cd11[multi], c='r', s=255)
    #axs1[3,1].scatter(ndata[multi], cd22[multi], c='r', s=255)
    axs1[3,0].scatter(ndata[which], cd11[which], label=ff)
    axs1[3,1].scatter(ndata[which], cd22[which], label=ff)
    #axs1[3,0].scatter(ndata[which], xscale[which], label=ff)
    #axs1[3,1].scatter(ndata[which], yscale[which], label=ff)
    ##[axs1[3,x].set_xlabel("PARANG (deg)") for x in (0,1)]
    [axs1[3,x].set_xlabel("Night fraction") for x in (0,1)]
    [axs1[3,x].set_xlim(0, 1) for x in (0,1)]

## Special handling (joint) for RUNID:


[ax.set_ylabel("CD11") for ax in axs1[:, 0]]
[ax.set_ylabel("CD22") for ax in axs1[:, 1]]

axs1[0,0].legend(loc='lower left')
axs1[1,0].legend(loc='upper left')

## CD11 and CD22 in velocity order:
#for ff,vdot in fvdot_data.items():


##--------------------------------------------------------------------------##

for fig in every_fig:
    fig.tight_layout()
plt.draw()

sys.exit(0)
##--------------------------------------------------------------------------##
#plt.style.use('bmh')   # Bayesian Methods for Hackers style
fig = plt.figure(1, figsize=fig_dims)
plt.gcf().clf()
#fig, axs = plt.subplots(nrows=2, ncols=2, num=1, clear=True, figsize=fig_dims,
#                        sharex=True, squeeze=False)
# sharex='col' | sharex='row' | squeeze=False
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

#lnorm = LogNorm(vmin=0.05, vmax=1.0)

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

#fig.align_labels()
#fig.align_xlabels()
#fig.align_ylabels()
#fig.tight_layout() # adjust boundaries sensibly, matplotlib v1.1+
#plt.draw()
#fig.savefig(plot_name, bbox_inches='tight')

# cyclical colormap ... cmocean.cm.phase
# cmocean: https://matplotlib.org/cmocean/




######################################################################
# CHANGELOG (11_combo_seasonal.py):
#---------------------------------------------------------------------
#
#  2026-10-20:
#     -- Increased __version__ to 0.1.0.
#     -- First created 11_combo_seasonal.py.
#
